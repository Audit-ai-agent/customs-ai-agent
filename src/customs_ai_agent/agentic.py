from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Iterable, Literal, Protocol

from customs_ai_agent.adapters.llm import (
    DEFAULT_CLAUDE_MODEL,
    ClaudeDocumentAnalyzer,
    LLMError,
    StructuredAgenticResult,
    anthropic_output_schema_sha256,
    text_output_schema_sha256,
)
from customs_ai_agent.build_info import (
    load_packaged_build_manifest,
    package_code_snapshot_sha256,
)
from customs_ai_agent.contracts import schema_sha256
from customs_ai_agent.prompts import (
    CASE_ANSWER_PROMPT_ID,
    NARRATIVE_REPORT_PROMPT_ID,
    REVIEW_COORDINATOR_PROMPT_ID,
    load_case_answer_system_prompt,
    load_document_analysis_system_prompt,
    load_narrative_report_system_prompt,
    load_review_coordinator_system_prompt,
)
from customs_ai_agent.release_profile import load_release_profile
from customs_ai_agent.security_guard import (
    PROMPT_INJECTION_REASON_CODE,
    prompt_injection_unmet_reason_codes,
)
from customs_ai_agent.schemas.models import ReviewRunResultV2
from customs_ai_agent.schemas.v3 import (
    ActionTraceV1,
    AgenticCallAuditV1,
    CitationReferenceV1,
    CompletionEligibilityV1,
    FactLedgerEntryV1,
    FactLedgerV1,
    GroundedAnswerV1,
    GroundedQuestionV1,
    NarrativeClaimV1,
    NarrativeReportV1,
    PlannedToolActionV1,
    RequiredActionV1,
    ReviewGapV1,
    ReviewPlanProposalV1,
    ReviewRunResultV3,
    RuntimeReleaseIdentityV1,
    ServiceReviewRunRequestV3,
    SupportingDocumentV1,
    ToolObservationV1,
    canonical_json_sha256,
    review_result_v3_sha256,
    stable_object_id,
)
from customs_ai_agent.assertion_trust import AssertionTrustStoreV1, load_assertion_trust_store
from customs_ai_agent.v3_semantics import (
    ACTIVE_FAILURE_REASON_CODES,
    SAFE_CLAIM_FRAMES,
    allowed_framed_claim_texts,
    citable_regulatory_evidence_ids,
    expected_required_actions,
    has_citable_regulatory_evidence,
    reconstruct_v3_current_semantics,
    render_framed_claim,
)


AGENT_POLICY_ID = "bounded-adaptive-customs-review/v1"
TOOL_REGISTRY_ID = "customs-agent-tools/text-v1"
MAX_AGENT_CYCLES = 2
MAX_TOOL_CALLS = 6


class BoundedToolPolicyError(RuntimeError):
    """Raised before an unregistered or unauthorised Tool can execute."""


@dataclass(frozen=True)
class BoundedToolResult:
    status: Literal["succeeded", "failed"]
    reason_code: str
    output_reference_ids: tuple[str, ...] = ()


class SpanRereadHandler(Protocol):
    def __call__(self, reference_ids: tuple[str, ...]) -> BoundedToolResult: ...


class TrustedAssertionVerifier(Protocol):
    """Runtime-owned trust bridge for connector receipts or Backend decisions.

    Request JSON cannot provide this verifier.  A Production composition must
    implement detached-signature verification or an immutable authenticated
    Backend decision lookup and pin that implementation in the code release.
    """

    def __call__(
        self,
        *,
        assertion: Any,
        document: SupportingDocumentV1,
        policy: dict[str, Any],
        case_id: str,
        assertion_cutoff_at: datetime,
    ) -> bool: ...


class EvidenceRetryHandler(Protocol):
    def __call__(self, reference_ids: tuple[str, ...]) -> BoundedToolResult: ...


@dataclass(frozen=True)
class DeterministicReevaluationResult:
    """Result of rerunning the code-owned stages affected by one observation."""

    status: Literal["succeeded", "failed"]
    reason_code: str
    resolved_gap_ids: tuple[str, ...] = ()
    output_reference_ids: tuple[str, ...] = ()


class DeterministicStageReevaluator(Protocol):
    def __call__(
        self,
        *,
        action: PlannedToolActionV1,
        observation: BoundedToolResult,
        affected_stages: tuple[str, ...],
    ) -> DeterministicReevaluationResult: ...


@dataclass(frozen=True)
class ActiveBoundedRuntimeOutcome:
    """Code-owned result returned after active observation and stage reruns."""

    result: ReviewRunResultV2
    supporting_document_rejections: tuple[str, ...]
    action_trace: tuple[ActionTraceV1, ...]
    observations: tuple[ToolObservationV1, ...]
    resolved_gap_ids: tuple[str, ...]
    failure_reason_codes: tuple[str, ...] = ()


class ActiveBoundedRuntimeExecutor(Protocol):
    def execute(
        self,
        *,
        plan: ReviewPlanProposalV1,
        fact_ledger: FactLedgerV1,
    ) -> ActiveBoundedRuntimeOutcome: ...


class BoundedToolBroker:
    """Closed Tool registry: no URL, SQL, file, shell, or dynamic handler name."""

    def __init__(
        self,
        *,
        span_reread: SpanRereadHandler | None = None,
        evidence_retry: EvidenceRetryHandler | None = None,
    ) -> None:
        self._span_reread = span_reread
        self._evidence_retry = evidence_retry

    def execute(
        self,
        tool_name: str,
        reference_ids: tuple[str, ...],
    ) -> BoundedToolResult:
        if not reference_ids:
            raise BoundedToolPolicyError("bounded Tool input must be source-bound")
        if tool_name == "document.reread_span/v1":
            if self._span_reread is None:
                raise BoundedToolPolicyError("span reread handler is unavailable")
            return self._span_reread(reference_ids)
        if tool_name == "evidence.retrieve/v3":
            if self._evidence_retry is None:
                raise BoundedToolPolicyError("Evidence retry handler is unavailable")
            return self._evidence_retry(reference_ids)
        raise BoundedToolPolicyError("Tool is outside the active bounded registry")


_COORDINATOR_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["suggestions"],
    "properties": {
        "suggestions": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["gap_id", "tool_name", "reason_code", "input_reference_ids"],
                "properties": {
                    "gap_id": {"type": "string", "pattern": r"^gap_[a-f0-9]{64}$"},
                    "tool_name": {
                        "type": "string",
                        "enum": [
                            "document.reread_span/v1",
                            "evidence.retrieve/v3",
                            "human.request/v1",
                        ],
                    },
                    "reason_code": {"type": "string", "minLength": 1, "maxLength": 128},
                    "input_reference_ids": {
                        "type": "array",
                        "maxItems": 100,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1, "maxLength": 256},
                    },
                },
            },
        }
    },
}
_NARRATIVE_SECTIONS = (
    "summary",
    "document_consistency",
    "regulatory_review",
    "required_actions",
)
_NARRATIVE_FRAME_IDS = (
    "plain",
    "summary_brief",
    "summary_label",
    "document_result",
    "document_review",
    "regulatory_result",
    "regulatory_review",
    "action_brief",
    "action_owner",
)
_SAFE_FRAME_SELECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["section", "source_claim_id", "frame"],
    "properties": {
        "section": {"type": "string", "enum": list(_NARRATIVE_SECTIONS)},
        "source_claim_id": {"type": "string", "pattern": r"^claim_[a-f0-9]{64}$"},
        "frame": {"type": "string", "enum": list(_NARRATIVE_FRAME_IDS)},
    },
}
_NARRATIVE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "minItems": 1,
            "maxItems": 320,
            "items": _SAFE_FRAME_SELECTION_SCHEMA,
        },
    },
}
_ANSWER_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "minItems": 1,
            "maxItems": 20,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["source_claim_id", "frame"],
                "properties": {
                    "source_claim_id": {
                        "type": "string",
                        "pattern": r"^answer_claim_[a-f0-9]{64}$",
                    },
                    "frame": {
                        "type": "string",
                        "enum": ["plain", "answer_grounded", "answer_evidence"],
                    },
                },
            },
        },
    },
}


def agentic_semantic_schema_hashes() -> dict[str, str]:
    """Return strict local schemas, distinct from provider-transmitted schemas."""

    return {
        "coordinator": canonical_json_sha256(_COORDINATOR_OUTPUT_SCHEMA),
        "narrative": canonical_json_sha256(_NARRATIVE_OUTPUT_SCHEMA),
        "chat": canonical_json_sha256(_ANSWER_OUTPUT_SCHEMA),
    }


_QUESTION_TOKEN = re.compile(r"[0-9A-Za-z가-힣]{2,}")
_KOREAN_QUESTION_TOKEN = re.compile(r"[가-힣]+")
_QUESTION_STOP_WORDS = frozenset(
    {"어떤", "무엇", "있나요", "인가요", "대해", "대한", "현재", "이번", "해당", "알려줘"}
)
_ISSUE_INTENT_KEYWORDS = frozenset({"문제", "오류", "불일치", "위험", "finding", "issue"})
_REGULATION_INTENT_KEYWORDS = frozenset({"법령", "규정", "요건", "서류", "근거", "evidence"})
_OUTCOME_INTENT_KEYWORDS = frozenset({"결과", "상태", "완료", "심사", "outcome", "status"})
_FIELD_INTENT_KEYWORDS = frozenset(
    {"문서", "금액", "수량", "원산지", "송장", "선하", "품목", "hsk", "hs"}
)


def _matches_question_intent(
    question_tokens: set[str],
    keywords: frozenset[str],
) -> bool:
    """Match Korean inflections without broadening English substring matching."""

    for keyword in keywords:
        if keyword.isascii():
            if keyword in question_tokens:
                return True
            continue
        if any(
            _KOREAN_QUESTION_TOKEN.fullmatch(token) is not None and keyword in token
            for token in question_tokens
        ):
            return True
    return False


@dataclass(frozen=True, slots=True)
class _FactIndex:
    ledger: FactLedgerV1
    field_references: tuple[str, ...]
    issue_references: tuple[str, ...]
    finding_references: tuple[str, ...]
    evidence_references: tuple[str, ...]
    outcome_reference: str


def _display_value(value: Any) -> str:
    if isinstance(value, str):
        rendered = value.strip()
    else:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
    if not rendered:
        rendered = "(empty)"
    return rendered[:4000]


def _fact(
    *,
    kind: str,
    reference_id: str,
    label: str,
    value: Any,
    document_id: str | None = None,
) -> FactLedgerEntryV1:
    rendered = _display_value(value)
    canonical = {
        "kind": kind,
        "reference_id": reference_id,
        "label": label,
        "value": rendered,
        "document_id": document_id,
    }
    return FactLedgerEntryV1(
        fact_id=stable_object_id("fact", canonical),
        kind=kind,  # type: ignore[arg-type]
        reference_id=reference_id,
        label=label,
        value=rendered,
        document_id=document_id,
        source_sha256=canonical_json_sha256(canonical),
    )


def build_fact_ledger(result: ReviewRunResultV2) -> _FactIndex:
    entries: list[FactLedgerEntryV1] = []
    field_references: list[str] = []
    for document in result.document_results:
        for field in document.fields:
            reference_id = stable_object_id(
                "field",
                {
                    "document_id": document.document_id,
                    "field_path": f"fields.{field.name}",
                    "source": field.source.model_dump(mode="json"),
                },
            )
            field_references.append(reference_id)
            entries.append(
                _fact(
                    kind="field_source",
                    reference_id=reference_id,
                    label=f"{document.document_id}.fields.{field.name}",
                    value=(
                        field.normalized_value
                        if field.normalized_value is not None
                        else field.value
                    ),
                    document_id=document.document_id,
                )
            )
        for line_item in document.line_items:
            for field in line_item.fields:
                reference_id = stable_object_id(
                    "field",
                    {
                        "document_id": document.document_id,
                        "field_path": f"line_items.{line_item.item_id}.{field.name}",
                        "source": field.source.model_dump(mode="json"),
                    },
                )
                field_references.append(reference_id)
                entries.append(
                    _fact(
                        kind="field_source",
                        reference_id=reference_id,
                        label=(
                            f"{document.document_id}.line_items.{line_item.item_id}.{field.name}"
                        ),
                        value=(
                            field.normalized_value
                            if field.normalized_value is not None
                            else field.value
                        ),
                        document_id=document.document_id,
                    )
                )

    issue_references: list[str] = []
    for index, issue in enumerate(result.validation_issues):
        payload = issue.model_dump(mode="json")
        reference_id = stable_object_id("issue", {"index": index, "issue": payload})
        issue_references.append(reference_id)
        entries.append(
            _fact(
                kind="issue",
                reference_id=reference_id,
                label=issue.issue_code,
                value=issue.message,
                document_id=issue.source_document_id,
            )
        )

    finding_references: list[str] = []
    for index, finding in enumerate(result.regulatory_findings):
        payload = finding.model_dump(mode="json")
        reference_id = stable_object_id("finding", {"index": index, "finding": payload})
        finding_references.append(reference_id)
        entries.append(
            _fact(
                kind="finding",
                reference_id=reference_id,
                label=finding.finding_code,
                value=finding.message,
                document_id=(finding.source.document_id if finding.source is not None else None),
            )
        )

    evidence_references: list[str] = []
    for evidence in result.rag_evidence:
        evidence_references.append(evidence.evidence_id)
        entries.append(
            _fact(
                kind="evidence",
                reference_id=evidence.evidence_id,
                label=evidence.title,
                value=evidence.text,
            )
        )

    outcome_reference = stable_object_id(
        "outcome",
        {
            "status": result.status,
            "requires_human_review": result.requires_human_review,
            "retrieval_status": result.retrieval_status,
            "retrieval_coverage": result.retrieval_coverage,
        },
    )
    entries.append(
        _fact(
            kind="outcome",
            reference_id=outcome_reference,
            label="deterministic_outcome",
            value={
                "status": result.status,
                "requires_human_review": result.requires_human_review,
                "retrieval_status": result.retrieval_status,
                "retrieval_coverage": result.retrieval_coverage,
            },
        )
    )
    entries = sorted(entries, key=lambda item: (item.kind, item.reference_id))
    ledger = FactLedgerV1(
        ledger_id=stable_object_id(
            "ledger",
            [item.model_dump(mode="json") for item in entries],
        ),
        entries=entries,
    )
    return _FactIndex(
        ledger=ledger,
        field_references=tuple(field_references),
        issue_references=tuple(issue_references),
        finding_references=tuple(finding_references),
        evidence_references=tuple(evidence_references),
        outcome_reference=outcome_reference,
    )


def _retain_investigation_input_facts(
    current: _FactIndex,
    investigation_input: FactLedgerV1,
    input_reference_ids: set[str],
) -> _FactIndex:
    """Retain only pre-action facts actually consumed by an active plan.

    An outcome/evidence rerun may replace the current FactLedger references.
    The action trace must still resolve each immutable input reference without
    confusing it with a current fact, so missing inputs are copied with an
    explicit audit-only label into the canonical run ledger.
    """

    existing_reference_ids = {item.reference_id for item in current.ledger.entries}
    retained = [
        _fact(
            kind=item.kind,
            reference_id=item.reference_id,
            label=f"investigation_input:{item.label}",
            value=item.value,
            document_id=item.document_id,
        )
        for item in investigation_input.entries
        if item.reference_id in input_reference_ids
        and item.reference_id not in existing_reference_ids
    ]
    if not retained:
        return current
    entries = sorted(
        [*current.ledger.entries, *retained],
        key=lambda item: (item.kind, item.reference_id),
    )
    ledger = FactLedgerV1(
        ledger_id=stable_object_id(
            "ledger",
            [item.model_dump(mode="json") for item in entries],
        ),
        entries=entries,
    )
    return _FactIndex(
        ledger=ledger,
        field_references=current.field_references,
        issue_references=current.issue_references,
        finding_references=current.finding_references,
        evidence_references=current.evidence_references,
        outcome_reference=current.outcome_reference,
    )


def _gap(gap_type: str, reason_code: str, references: Iterable[str]) -> ReviewGapV1:
    source_reference_ids = sorted(set(references))
    canonical = {
        "gap_type": gap_type,
        "reason_code": reason_code,
        "source_reference_ids": source_reference_ids,
    }
    return ReviewGapV1(
        gap_id=stable_object_id("gap", canonical),
        gap_type=gap_type,  # type: ignore[arg-type]
        reason_code=reason_code,
        source_reference_ids=source_reference_ids,
        status="open",
    )


def detect_review_gaps(
    result: ReviewRunResultV2,
    facts: _FactIndex,
    *,
    extraction_release_approved: bool = False,
) -> list[ReviewGapV1]:
    gaps: list[ReviewGapV1] = []
    if result.model_errors or (
        not extraction_release_approved
        and any(item.extraction_confidence < 0.90 for item in result.document_results)
    ):
        gaps.append(
            _gap(
                "field_uncertainty",
                "extraction_confidence_below_release_threshold",
                facts.field_references[:20],
            )
        )
    if facts.issue_references:
        gaps.append(
            _gap("validation_issue", "deterministic_validation_issue", facts.issue_references)
        )
    if facts.finding_references:
        gaps.append(
            _gap("regulatory_finding", "material_regulatory_finding", facts.finding_references)
        )
    if result.retrieval_status != "ok" or result.retrieval_coverage != "complete":
        gaps.append(
            _gap(
                "evidence_incomplete",
                "evidence_not_complete",
                (*facts.evidence_references, facts.outcome_reference),
            )
        )
    if result.requires_human_review:
        gaps.append(
            _gap(
                "human_approval_required",
                "deterministic_outcome_requires_human",
                (facts.outcome_reference,),
            )
        )
    unique = {gap.gap_id: gap for gap in gaps}
    return sorted(unique.values(), key=lambda item: item.gap_id)


def _tool_for_gap(gap: ReviewGapV1) -> str:
    if gap.gap_type == "field_uncertainty":
        return "document.reread_span/v1"
    if gap.gap_type == "evidence_incomplete":
        return "evidence.retrieve/v3"
    return "human.request/v1"


def compile_shadow_plan(
    gaps: list[ReviewGapV1],
    *,
    suggestions: list[dict[str, Any]] | None = None,
    model_call: StructuredAgenticResult | None = None,
) -> ReviewPlanProposalV1:
    actions: list[PlannedToolActionV1] = []
    gap_by_id = {item.gap_id: item for item in gaps}
    proposed = suggestions or [
        {
            "gap_id": gap.gap_id,
            "tool_name": _tool_for_gap(gap),
            "reason_code": gap.reason_code,
            "input_reference_ids": gap.source_reference_ids,
        }
        for gap in gaps[: MAX_TOOL_CALLS - 1]
    ]
    for suggestion in proposed[: MAX_TOOL_CALLS - 1]:
        gap = gap_by_id.get(str(suggestion.get("gap_id") or ""))
        if gap is None:
            continue
        tool_name = str(suggestion.get("tool_name") or "")
        # A model proposal is narrower than the policy compiler. It cannot
        # change which Tool class is permitted for a Gap type.
        if tool_name != _tool_for_gap(gap):
            continue
        requested_references = suggestion.get("input_reference_ids")
        if not isinstance(requested_references, list) or any(
            not isinstance(item, str) for item in requested_references
        ):
            continue
        input_references = sorted(set(requested_references))
        if not set(input_references).issubset(gap.source_reference_ids):
            continue
        reason_code = str(suggestion.get("reason_code") or gap.reason_code)[:128]
        canonical = {
            "tool_name": tool_name,
            "gap_id": gap.gap_id,
            "reason_code": reason_code,
            "input_reference_ids": input_references,
        }
        actions.append(
            PlannedToolActionV1(
                action_id=stable_object_id("action", canonical),
                tool_name=tool_name,  # type: ignore[arg-type]
                gap_id=gap.gap_id,
                reason_code=reason_code,
                input_reference_ids=input_references,
            )
        )
    if not gaps:
        payload = {
            "mode": "shadow",
            "cycle_limit": MAX_AGENT_CYCLES,
            "tool_call_limit": MAX_TOOL_CALLS,
            "actions": [],
            "provider": "anthropic" if model_call is not None else "deterministic_policy",
            "model_id": model_call.model_id if model_call is not None else None,
            "prompt_sha256": model_call.prompt_sha256 if model_call is not None else None,
            "output_schema_sha256": (
                model_call.output_schema_sha256 if model_call is not None else None
            ),
            "input_tokens": model_call.input_tokens if model_call is not None else 0,
            "output_tokens": model_call.output_tokens if model_call is not None else 0,
            "latency_ms": model_call.latency_ms if model_call is not None else 0,
        }
        return ReviewPlanProposalV1(
            plan_id=stable_object_id("plan", payload),
            mode="shadow",
            cycle_limit=MAX_AGENT_CYCLES,
            tool_call_limit=MAX_TOOL_CALLS,
            actions=[],
            provider=("anthropic" if model_call is not None else "deterministic_policy"),
            model_id=(model_call.model_id if model_call is not None else None),
            prompt_sha256=(model_call.prompt_sha256 if model_call is not None else None),
            output_schema_sha256=(
                model_call.output_schema_sha256 if model_call is not None else None
            ),
            input_tokens=(model_call.input_tokens if model_call is not None else 0),
            output_tokens=(model_call.output_tokens if model_call is not None else 0),
            latency_ms=(model_call.latency_ms if model_call is not None else 0),
        )
    explanation_gap = gaps[0]
    explanation = {
        "tool_name": "explanation.compose/v1",
        "gap_id": explanation_gap.gap_id,
        "reason_code": "compose_grounded_review_report",
        "input_reference_ids": explanation_gap.source_reference_ids,
    }
    actions.append(
        PlannedToolActionV1(
            action_id=stable_object_id("action", explanation),
            tool_name="explanation.compose/v1",
            gap_id=explanation_gap.gap_id,
            reason_code="compose_grounded_review_report",
            input_reference_ids=explanation_gap.source_reference_ids,
        )
    )
    payload = {
        "mode": "shadow",
        "cycle_limit": MAX_AGENT_CYCLES,
        "tool_call_limit": MAX_TOOL_CALLS,
        "actions": [item.model_dump(mode="json") for item in actions],
        "provider": "anthropic" if model_call is not None else "deterministic_policy",
        "model_id": model_call.model_id if model_call is not None else None,
        "prompt_sha256": model_call.prompt_sha256 if model_call is not None else None,
        "output_schema_sha256": (
            model_call.output_schema_sha256 if model_call is not None else None
        ),
        "input_tokens": model_call.input_tokens if model_call is not None else 0,
        "output_tokens": model_call.output_tokens if model_call is not None else 0,
        "latency_ms": model_call.latency_ms if model_call is not None else 0,
    }
    return ReviewPlanProposalV1(
        plan_id=stable_object_id("plan", payload),
        mode="shadow",
        cycle_limit=MAX_AGENT_CYCLES,
        tool_call_limit=MAX_TOOL_CALLS,
        actions=actions,
        provider=("anthropic" if model_call is not None else "deterministic_policy"),
        model_id=(model_call.model_id if model_call is not None else None),
        prompt_sha256=(model_call.prompt_sha256 if model_call is not None else None),
        output_schema_sha256=(model_call.output_schema_sha256 if model_call is not None else None),
        input_tokens=(model_call.input_tokens if model_call is not None else 0),
        output_tokens=(model_call.output_tokens if model_call is not None else 0),
        latency_ms=(model_call.latency_ms if model_call is not None else 0),
    )


def claude_shadow_plan(
    analyzer: ClaudeDocumentAnalyzer,
    gaps: list[ReviewGapV1],
) -> ReviewPlanProposalV1:
    context = {
        "mode": "shadow",
        "allowed_tools": [
            "document.reread_span/v1",
            "evidence.retrieve/v3",
            "human.request/v1",
        ],
        "declared_gaps": [item.model_dump(mode="json") for item in gaps],
        "limits": {"max_cycles": MAX_AGENT_CYCLES, "max_tool_calls": MAX_TOOL_CALLS},
    }
    result = analyzer.run_bounded_agentic_task(
        task_id=REVIEW_COORDINATOR_PROMPT_ID,
        system_prompt=load_review_coordinator_system_prompt(),
        untrusted_context=context,
        output_schema=_COORDINATOR_OUTPUT_SCHEMA,
        max_tokens=2048,
    )
    if result.semantic_output_schema_sha256 != agentic_semantic_schema_hashes()["coordinator"]:
        raise ValueError("coordinator semantic output schema identity mismatch")
    suggestions = result.value.get("suggestions")
    if not isinstance(suggestions, list):
        raise ValueError("coordinator suggestions are invalid")
    return compile_shadow_plan(gaps, suggestions=suggestions, model_call=result)


def _citation(kind: str, reference_id: str) -> CitationReferenceV1:
    return CitationReferenceV1(kind=kind, reference_id=reference_id)  # type: ignore[arg-type]


def _claim(text: str, citations: list[CitationReferenceV1]) -> NarrativeClaimV1:
    payload = {
        "text": text,
        "material": True,
        "citations": [item.model_dump(mode="json") for item in citations],
    }
    return NarrativeClaimV1(
        claim_id=stable_object_id("claim", payload),
        text=text,
        material=True,
        citations=citations,
    )


def deterministic_narrative(
    result: ReviewRunResultV2,
    facts: _FactIndex,
    *,
    release_approved: bool = False,
) -> NarrativeReportV1:
    outcome_citation = [_citation("outcome", facts.outcome_reference)]
    if result.validation_issues or result.regulatory_findings:
        summary_text = "문서 또는 규정 검토에서 담당자 확인이 필요한 항목이 발견되었습니다."
    elif (
        result.retrieval_status == "ok"
        and result.retrieval_coverage == "complete"
        and has_citable_regulatory_evidence(result)
    ):
        summary_text = "현재 검토 범위에서 문서 불일치와 미충족 규정 요건이 발견되지 않았습니다."
    else:
        summary_text = "규정 근거가 완전하지 않아 자동 사전심사를 확정할 수 없습니다."
    summary = [_claim(summary_text, outcome_citation)]

    document_consistency: list[NarrativeClaimV1] = []
    for issue, reference_id in zip(result.validation_issues, facts.issue_references, strict=True):
        document_consistency.append(
            _claim(
                f"{issue.issue_code}: {issue.message}",
                [_citation("issue", reference_id)],
            )
        )
    if not document_consistency:
        document_consistency.append(
            _claim("결정론적 문서 교차검증에서 불일치가 발견되지 않았습니다.", outcome_citation)
        )

    regulatory_review: list[NarrativeClaimV1] = []
    citable_evidence_ids = citable_regulatory_evidence_ids(result)
    for finding, reference_id in zip(
        result.regulatory_findings,
        facts.finding_references,
        strict=True,
    ):
        citations = [_citation("finding", reference_id)]
        citations.extend(_citation("evidence", evidence_id) for evidence_id in finding.evidence_ids)
        regulatory_review.append(_claim(f"{finding.finding_code}: {finding.message}", citations))
    if not regulatory_review:
        if citable_evidence_ids:
            regulatory_review.append(
                _claim(
                    "조회된 규정 근거 범위에서 별도의 미충족 요건 Finding이 생성되지 않았습니다.",
                    [_citation("evidence", citable_evidence_ids[0])],
                )
            )
        else:
            regulatory_review.append(
                _claim("인용 가능한 규정 근거가 없어 규정 판단을 보류합니다.", outcome_citation)
            )

    required_actions: list[NarrativeClaimV1] = []
    if result.requires_human_review:
        required_actions.append(
            _claim(
                "담당자는 근거와 추출값을 확인한 뒤 수정 또는 승인을 기록해야 합니다.",
                outcome_citation,
            )
        )
    if result.status == "blocked":
        required_actions.append(
            _claim(
                "차단 원인을 해소한 뒤 새로운 Review Run으로 다시 실행해야 합니다.",
                outcome_citation,
            )
        )
    if not required_actions:
        message = (
            "승인된 자동완료 범위에서 추가 조치가 필요한 항목이 없습니다."
            if release_approved
            else "현재 결과는 Release Gate 승인 전까지 자동완료에 사용할 수 없습니다."
        )
        required_actions.append(_claim(message, outcome_citation))
    return NarrativeReportV1(
        summary=summary,
        document_consistency=document_consistency,
        regulatory_review=regulatory_review,
        required_actions=required_actions,
        generation_mode="deterministic_fallback",
    )


def _call_audit(task_id: str, call: StructuredAgenticResult) -> AgenticCallAuditV1:
    return AgenticCallAuditV1(
        task_id=task_id,  # type: ignore[arg-type]
        model_id=call.model_id,
        prompt_sha256=call.prompt_sha256,
        output_schema_sha256=call.output_schema_sha256,
        request_id_sha256=hashlib.sha256(call.request_id.encode("utf-8")).hexdigest(),
        message_id_sha256=hashlib.sha256(call.message_id.encode("utf-8")).hexdigest(),
        input_tokens=call.input_tokens,
        output_tokens=call.output_tokens,
        latency_ms=call.latency_ms,
    )


def _render_grounded_claim(canonical: str, frame: str, *, section: str) -> str:
    """Render a code-owned claim from Claude's bounded presentation choice.

    Claude selects only an allowlisted frame identifier.  The factual sentence
    and citations never cross the model-output authority boundary: both are
    projected from the canonical code-owned claim after strict identity checks.
    """

    return render_framed_claim(canonical, frame, section=section)


def _rewrite_report(
    canonical: NarrativeReportV1,
    values: Any,
    *,
    call: StructuredAgenticResult,
) -> NarrativeReportV1:
    if not isinstance(values, list):
        raise ValueError("narrative rewrite collection is invalid")
    canonical_claims = {
        claim.claim_id: (section, claim)
        for section in _NARRATIVE_SECTIONS
        for claim in getattr(canonical, section)
    }
    selections: dict[str, tuple[str, str]] = {}
    for value in values:
        if not isinstance(value, dict) or set(value) != {"section", "source_claim_id", "frame"}:
            raise ValueError("narrative frame selection is invalid")
        section = value.get("section")
        claim_id = value.get("source_claim_id")
        frame = value.get("frame")
        if (
            not isinstance(section, str)
            or not isinstance(claim_id, str)
            or not isinstance(frame, str)
            or claim_id in selections
        ):
            raise ValueError("narrative frame selection identity is invalid")
        source = canonical_claims.get(claim_id)
        if source is None or source[0] != section:
            raise ValueError("narrative frame selection references an unknown typed claim")
        _render_grounded_claim(source[1].text, frame, section=section)
        selections[claim_id] = (section, frame)
    if set(selections) != set(canonical_claims):
        raise ValueError("narrative frame selection must cover every canonical claim exactly once")

    def project(claims: list[NarrativeClaimV1]) -> list[NarrativeClaimV1]:
        return [
            _claim(
                _render_grounded_claim(
                    claim.text,
                    selections[claim.claim_id][1],
                    section=selections[claim.claim_id][0],
                ),
                claim.citations,
            )
            for claim in claims
        ]

    return NarrativeReportV1(
        summary=project(canonical.summary),
        document_consistency=project(canonical.document_consistency),
        regulatory_review=project(canonical.regulatory_review),
        required_actions=project(canonical.required_actions),
        generation_mode="claude",
        model_call=_call_audit(NARRATIVE_REPORT_PROMPT_ID, call),
    )


def claude_narrative(
    analyzer: ClaudeDocumentAnalyzer,
    result: ReviewRunResultV2,
    facts: _FactIndex,
    *,
    release_approved: bool = False,
) -> tuple[NarrativeReportV1, StructuredAgenticResult]:
    canonical = deterministic_narrative(result, facts, release_approved=release_approved)
    canonical_claims = [
        {
            "section": section,
            "source_claim_id": claim.claim_id,
            "canonical_text": claim.text,
            "allowed_frames": list(SAFE_CLAIM_FRAMES[section]),
        }
        for section in _NARRATIVE_SECTIONS
        for claim in getattr(canonical, section)
    ]
    context = {
        "canonical_claims": canonical_claims,
        "disclaimer": ("This is a customs pre-review aid and is not a customs clearance decision."),
    }
    call = analyzer.run_bounded_agentic_task(
        task_id=NARRATIVE_REPORT_PROMPT_ID,
        system_prompt=load_narrative_report_system_prompt(),
        untrusted_context=context,
        output_schema=_NARRATIVE_OUTPUT_SCHEMA,
        max_tokens=4096,
    )
    if call.semantic_output_schema_sha256 != agentic_semantic_schema_hashes()["narrative"]:
        raise ValueError("narrative semantic output schema identity mismatch")
    report = _rewrite_report(canonical, call.value.get("claims"), call=call)
    return report, call


def _required_actions(
    result: ReviewRunResultV2,
    *,
    supporting_rejections: list[str],
    release_approved: bool,
    active_failure_codes: list[str],
) -> list[RequiredActionV1]:
    semantics = reconstruct_v3_current_semantics(result)
    return [
        RequiredActionV1.model_validate(item)
        for item in expected_required_actions(
            result,
            semantics,
            completion_eligible=release_approved,
            supporting_verification_incomplete=bool(supporting_rejections),
            active_failure_codes=active_failure_codes,
        )
    ]


def activate_bounded_plan(
    plan: ReviewPlanProposalV1,
    *,
    policy_id: str,
) -> ReviewPlanProposalV1:
    """Convert a reviewed shadow plan only under the signed packaged release.

    The release profile is deliberately loaded here instead of accepted as an
    argument, preventing a caller from injecting an in-memory approval object.
    """

    profile = load_release_profile()
    try:
        build_authorized = profile.agent_code_snapshot_sha256 == package_code_snapshot_sha256()
    except Exception:
        build_authorized = False
    if (
        policy_id != AGENT_POLICY_ID
        or profile.policy_id != AGENT_POLICY_ID
        or profile.tool_registry_id != TOOL_REGISTRY_ID
        or profile.release_state != "approved"
        or profile.kill_switch_active
        or not profile.bounded_adaptive_active_enabled
        or not build_authorized
    ):
        raise BoundedToolPolicyError("active bounded execution is not release-authorized")
    payload = {
        **plan.model_dump(mode="json", exclude={"plan_id", "mode"}),
        "mode": "active",
    }
    return ReviewPlanProposalV1(
        plan_id=stable_object_id("plan", payload),
        **payload,
    )


def execute_active_bounded_plan(
    plan: ReviewPlanProposalV1,
    *,
    broker: BoundedToolBroker,
    reevaluate: DeterministicStageReevaluator,
    policy_id: str,
) -> tuple[list[ActionTraceV1], list[ToolObservationV1], list[str]]:
    """Execute allowlisted observations and deterministic affected-stage reruns.

    This API is intentionally dependency-injected.  The broker can observe a
    source span or retry the already-pinned Evidence provider, while only the
    code-owned reevaluator may resolve a Gap.  It cannot accept or emit Outcome
    or severity mutations from Claude.
    """

    if plan.mode != "active":
        raise BoundedToolPolicyError("active executor requires an active plan")
    # Re-check the packaged signed policy at the execution boundary.  Activation
    # alone is not a reusable bearer capability after a kill-switch change.
    profile = load_release_profile()
    try:
        build_authorized = profile.agent_code_snapshot_sha256 == package_code_snapshot_sha256()
    except Exception:
        build_authorized = False
    if (
        policy_id != AGENT_POLICY_ID
        or profile.policy_id != AGENT_POLICY_ID
        or profile.tool_registry_id != TOOL_REGISTRY_ID
        or profile.release_state != "approved"
        or profile.kill_switch_active
        or not profile.bounded_adaptive_active_enabled
        or not build_authorized
    ):
        raise BoundedToolPolicyError("active bounded execution is not release-authorized")

    traces: list[ActionTraceV1] = []
    observations: list[ToolObservationV1] = []
    resolved_gap_ids: set[str] = set()
    reread_references: set[str] = set()
    calls = 0
    per_cycle = max(
        1, (plan.tool_call_limit + max(plan.cycle_limit, 1) - 1) // max(plan.cycle_limit, 1)
    )
    for action in plan.actions:
        if calls >= min(plan.tool_call_limit, MAX_TOOL_CALLS):
            break
        cycle = calls // per_cycle
        if cycle >= min(plan.cycle_limit, MAX_AGENT_CYCLES):
            break
        calls += 1
        status: Literal["succeeded", "failed", "deferred"]
        reason_code: str
        output_references: list[str]
        if action.tool_name == "human.request/v1":
            status = "succeeded"
            reason_code = "human_review_request_recorded"
            output_references = [action.gap_id]
        elif action.tool_name == "explanation.compose/v1":
            status = "succeeded"
            reason_code = "explanation_rendered_after_deterministic_reevaluation"
            output_references = [action.gap_id]
        elif action.tool_name == "case.answer/v1":
            status = "deferred"
            reason_code = "case_answer_is_not_an_active_review_tool"
            output_references = [action.gap_id]
        elif action.tool_name == "document.reread_span/v1" and reread_references.intersection(
            action.input_reference_ids
        ):
            status = "deferred"
            reason_code = "field_reread_limit_reached"
            output_references = [action.gap_id]
        else:
            if action.tool_name == "document.reread_span/v1":
                reread_references.update(action.input_reference_ids)
            try:
                observed = broker.execute(
                    action.tool_name,
                    tuple(action.input_reference_ids),
                )
            except BoundedToolPolicyError:
                status = "failed"
                reason_code = "bounded_tool_policy_rejected"
                output_references = [action.gap_id]
            else:
                if observed.status != "succeeded":
                    status = "failed"
                    reason_code = observed.reason_code
                    output_references = [*observed.output_reference_ids, action.gap_id]
                else:
                    affected_stages = (
                        (
                            "extraction",
                            "normalization",
                            "pairing",
                            "validation",
                            "compliance",
                            "outcome",
                        )
                        if action.tool_name == "document.reread_span/v1"
                        else ("evidence", "compliance", "outcome")
                    )
                    rerun = reevaluate(
                        action=action,
                        observation=observed,
                        affected_stages=affected_stages,
                    )
                    if rerun.status == "succeeded":
                        status = "succeeded"
                        reason_code = rerun.reason_code
                        resolved_gap_ids.update(rerun.resolved_gap_ids)
                    else:
                        status = "failed"
                        reason_code = rerun.reason_code
                    output_references = sorted(
                        {
                            *observed.output_reference_ids,
                            *rerun.output_reference_ids,
                        }
                    )
                    if not output_references:
                        output_references = [action.gap_id]
        trace = ActionTraceV1(
            action_id=action.action_id,
            cycle=cycle,
            tool_name=action.tool_name,
            reason_code=action.reason_code,
            input_reference_ids=action.input_reference_ids,
            output_reference_ids=output_references,
            status=status,
            duration_ms=0,
            provider="deterministic_policy",
        )
        traces.append(trace)
        observation_payload = {
            "action_id": action.action_id,
            "cycle": cycle,
            "tool_name": action.tool_name,
            "status": status,
            "reason_code": reason_code,
            "input_reference_ids": action.input_reference_ids,
            "output_reference_ids": output_references,
        }
        observations.append(
            ToolObservationV1(
                observation_id=stable_object_id("observation", observation_payload),
                action_id=action.action_id,
                cycle=cycle,
                tool_name=action.tool_name,
                status=status,
                reason_code=reason_code,
                input_reference_ids=action.input_reference_ids,
                output_reference_ids=output_references,
            )
        )
    return traces, observations, sorted(resolved_gap_ids)


def execute_bounded_plan(
    plan: ReviewPlanProposalV1,
    narrative: NarrativeReportV1,
    narrative_call: StructuredAgenticResult | None,
) -> tuple[list[ActionTraceV1], list[ToolObservationV1]]:
    """Execute only locally safe tools; defer anything requiring pipeline mutation.

    The mandatory workflow has already completed.  Evidence or field rereads that
    could change its state require a new deterministic child run and are therefore
    converted to a human request instead of being applied post hoc.
    """

    traces: list[ActionTraceV1] = []
    observations: list[ToolObservationV1] = []
    reread_references: set[str] = set()
    for index, action in enumerate(plan.actions[:MAX_TOOL_CALLS]):
        cycle = min(index // max(1, MAX_TOOL_CALLS // MAX_AGENT_CYCLES), MAX_AGENT_CYCLES - 1)
        explanation = action.tool_name == "explanation.compose/v1"
        repeated_reread = action.tool_name == "document.reread_span/v1" and bool(
            reread_references.intersection(action.input_reference_ids)
        )
        if action.tool_name == "document.reread_span/v1":
            reread_references.update(action.input_reference_ids)
        status: Literal["succeeded", "failed", "deferred"]
        if explanation:
            status = "succeeded"
            reason_code = "canonical_claims_rendered"
            output_references = [claim.claim_id for claim in narrative.summary]
        elif action.tool_name == "human.request/v1":
            status = "succeeded"
            reason_code = "human_review_request_recorded"
            output_references = [action.gap_id]
        elif repeated_reread:
            status = "deferred"
            reason_code = "field_reread_limit_reached"
            output_references = [action.gap_id]
        else:
            status = "deferred"
            reason_code = "new_child_run_required_for_mandatory_pipeline_reexecution"
            output_references = [action.gap_id]
        traces.append(
            ActionTraceV1(
                action_id=action.action_id,
                cycle=cycle,
                tool_name=action.tool_name,
                reason_code=action.reason_code,
                input_reference_ids=action.input_reference_ids,
                output_reference_ids=output_references,
                status=status,
                duration_ms=(narrative_call.latency_ms if explanation and narrative_call else 0),
                provider=(
                    "anthropic" if explanation and narrative_call else "deterministic_policy"
                ),
                model_id=(narrative_call.model_id if explanation and narrative_call else None),
                prompt_sha256=(
                    narrative_call.prompt_sha256 if explanation and narrative_call else None
                ),
                input_tokens=(narrative_call.input_tokens if explanation and narrative_call else 0),
                output_tokens=(
                    narrative_call.output_tokens if explanation and narrative_call else 0
                ),
            )
        )
        observation_payload = {
            "action_id": action.action_id,
            "cycle": cycle,
            "tool_name": action.tool_name,
            "status": status,
            "reason_code": reason_code,
            "input_reference_ids": action.input_reference_ids,
            "output_reference_ids": output_references,
        }
        observations.append(
            ToolObservationV1(
                observation_id=stable_object_id("observation", observation_payload),
                action_id=action.action_id,
                cycle=cycle,
                tool_name=action.tool_name,
                status=status,
                reason_code=reason_code,
                input_reference_ids=action.input_reference_ids,
                output_reference_ids=output_references,
            )
        )
    return traces, observations


def runtime_release_identity(
    *,
    model_id: str,
    evidence_snapshot_id: str | None,
    evidence_runtime_v3_sha256: str | None,
    assertion_trust_store: AssertionTrustStoreV1 | None = None,
) -> RuntimeReleaseIdentityV1:
    """Build the canonical public Runtime identity from one code-owned source.

    Service metadata and each V3 result call this function.  That prevents the
    preflight identity advertised to Backend from drifting from the identity
    embedded in a completed review.
    """

    try:
        package_version = version("customs-ai-agent")
    except PackageNotFoundError:
        package_version = "0.3.0"
    prompt_sha = hashlib.sha256(load_document_analysis_system_prompt().encode("utf-8")).hexdigest()
    coordinator_prompt_sha = hashlib.sha256(
        load_review_coordinator_system_prompt().encode("utf-8")
    ).hexdigest()
    narrative_prompt_sha = hashlib.sha256(
        load_narrative_report_system_prompt().encode("utf-8")
    ).hexdigest()
    chat_prompt_sha = hashlib.sha256(load_case_answer_system_prompt().encode("utf-8")).hexdigest()
    profile = load_release_profile()
    assertion_store = assertion_trust_store or load_assertion_trust_store()
    build_manifest = load_packaged_build_manifest(required=False)
    exact_model_revision = (
        model_id if re.fullmatch(r"claude-[A-Za-z0-9._-]*[0-9]{8}", model_id) else None
    )
    return RuntimeReleaseIdentityV1(
        agent_package_version=package_version,
        agent_git_sha=(build_manifest.git_sha if build_manifest is not None else None),
        agent_source_snapshot_sha256=(
            build_manifest.source_snapshot_sha256 if build_manifest is not None else None
        ),
        agent_code_snapshot_sha256=package_code_snapshot_sha256(),
        model_id=model_id,
        model_revision=exact_model_revision,
        extraction_prompt_sha256=prompt_sha,
        coordinator_prompt_sha256=coordinator_prompt_sha,
        narrative_prompt_sha256=narrative_prompt_sha,
        chat_prompt_sha256=chat_prompt_sha,
        extraction_schema_sha256=text_output_schema_sha256(),
        coordinator_schema_sha256=anthropic_output_schema_sha256(_COORDINATOR_OUTPUT_SCHEMA),
        narrative_schema_sha256=anthropic_output_schema_sha256(_NARRATIVE_OUTPUT_SCHEMA),
        chat_schema_sha256=anthropic_output_schema_sha256(_ANSWER_OUTPUT_SCHEMA),
        result_schema_sha256=schema_sha256("backend", "v3", "review-result"),
        policy_id=AGENT_POLICY_ID,
        tool_registry_id=TOOL_REGISTRY_ID,
        evidence_snapshot_id=evidence_snapshot_id,
        evidence_runtime_v3_sha256=evidence_runtime_v3_sha256,
        release_profile_id=profile.profile_id,
        release_profile_state=profile.release_state,
        release_profile_sha256=profile.profile_sha256,
        release_profile_subject_sha256=profile.review_subject_sha256,
        release_profile_evaluation_report_sha256=profile.evaluation_report_sha256,
        reviewer_trust_store_sha256=profile.reviewer_trust_store_sha256,
        assertion_trust_store_sha256=assertion_store.trust_store_sha256,
        assertion_trust_store_state=assertion_store.release_state,
        kill_switch_active=profile.kill_switch_active,
    )


def _runtime_release(
    result: ReviewRunResultV2,
    *,
    evidence_runtime_v3_sha256: str | None,
    assertion_trust_store: AssertionTrustStoreV1 | None = None,
) -> RuntimeReleaseIdentityV1:
    observed_model = result.model_calls[0].model_id if result.model_calls else DEFAULT_CLAUDE_MODEL
    return runtime_release_identity(
        model_id=observed_model,
        evidence_snapshot_id=result.knowledge_snapshot_id,
        evidence_runtime_v3_sha256=evidence_runtime_v3_sha256,
        assertion_trust_store=assertion_trust_store,
    )


def _detect_text_language(text: str | None) -> str:
    """Conservatively verify the text script used by the first text-only release."""

    if not text:
        return "unknown"
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return "unknown"
    has_hangul = any("\uac00" <= character <= "\ud7a3" for character in letters)
    if has_hangul:
        has_non_hangul_letters = any(
            not ("\uac00" <= character <= "\ud7a3") for character in letters
        )
        return "mixed" if has_non_hangul_letters else "ko"
    if all(character.isascii() for character in letters):
        return "en"
    return "unknown"


def automatic_completion_unmet_reason_codes(
    result: ReviewRunResultV2,
    request: ServiceReviewRunRequestV3,
    runtime_release: RuntimeReleaseIdentityV1,
    *,
    supporting_rejections: list[str],
    require_completed_outcome: bool = True,
) -> list[str]:
    """Return exact release/scope blockers without granting policy authority.

    The active runtime uses ``require_completed_outcome=False`` only while it is
    deciding whether the signed extraction calibration may be supplied to the
    code-owned outcome policy.  Every other release, evidence, document, model,
    supporting-document, issue, and finding Gate remains mandatory.
    """

    profile = load_release_profile()
    reasons: set[str] = set()
    reasons.update(
        prompt_injection_unmet_reason_codes(document.content_text for document in request.documents)
    )
    if profile.release_state != "approved":
        reasons.add("claude_release_profile_unapproved")
        reasons.add("domain_reviewers_unavailable")
    if profile.kill_switch_active or not profile.auto_completion_enabled:
        reasons.add("automatic_completion_kill_switch_active")
    identity_pairs = (
        (profile.agent_code_snapshot_sha256, runtime_release.agent_code_snapshot_sha256),
        (profile.model_id, runtime_release.model_id),
        (profile.model_revision, runtime_release.model_revision),
        (profile.extraction_prompt_sha256, runtime_release.extraction_prompt_sha256),
        (profile.coordinator_prompt_sha256, runtime_release.coordinator_prompt_sha256),
        (profile.narrative_prompt_sha256, runtime_release.narrative_prompt_sha256),
        (profile.chat_prompt_sha256, runtime_release.chat_prompt_sha256),
        (profile.extraction_schema_sha256, runtime_release.extraction_schema_sha256),
        (profile.coordinator_schema_sha256, runtime_release.coordinator_schema_sha256),
        (profile.narrative_schema_sha256, runtime_release.narrative_schema_sha256),
        (profile.chat_schema_sha256, runtime_release.chat_schema_sha256),
        (
            profile.coordinator_semantic_schema_sha256,
            agentic_semantic_schema_hashes()["coordinator"],
        ),
        (
            profile.narrative_semantic_schema_sha256,
            agentic_semantic_schema_hashes()["narrative"],
        ),
        (
            profile.chat_semantic_schema_sha256,
            agentic_semantic_schema_hashes()["chat"],
        ),
        (profile.policy_id, runtime_release.policy_id),
        (profile.tool_registry_id, runtime_release.tool_registry_id),
        (
            profile.request_schema_sha256,
            schema_sha256("backend", "v3", "review-request"),
        ),
        (profile.result_schema_sha256, runtime_release.result_schema_sha256),
        (profile.profile_sha256, runtime_release.release_profile_sha256),
        (profile.review_subject_sha256, runtime_release.release_profile_subject_sha256),
        (
            profile.evaluation_report_sha256,
            runtime_release.release_profile_evaluation_report_sha256,
        ),
        (
            profile.reviewer_trust_store_sha256,
            runtime_release.reviewer_trust_store_sha256,
        ),
        (
            profile.assertion_trust_store_sha256,
            runtime_release.assertion_trust_store_sha256,
        ),
    )
    if profile.release_state == "approved" and any(
        expected != observed for expected, observed in identity_pairs
    ):
        reasons.add("claude_release_identity_mismatch")
    if runtime_release.model_revision is None:
        reasons.add("claude_exact_model_revision_unavailable")
    if runtime_release.assertion_trust_store_state != "active":
        reasons.add("supporting_document_trust_store_unreleased")
    expected_document_ids = {document.document_id for document in request.documents}
    model_call_document_ids = [call.document_id for call in result.model_calls]
    if (
        not result.model_calls
        or len(model_call_document_ids) != len(set(model_call_document_ids))
        or set(model_call_document_ids) != expected_document_ids
        or any(call.input_modality != "text" for call in result.model_calls)
        or any(
            call.model_id != runtime_release.model_id
            or call.prompt_sha256 != runtime_release.extraction_prompt_sha256
            or call.output_schema_sha256 != runtime_release.extraction_schema_sha256
            for call in result.model_calls
        )
    ):
        reasons.add("claude_model_call_identity_incomplete")
    if result.model_errors or result.model_conflicts:
        reasons.add("claude_extraction_not_release_clean")
    observed_languages = {
        _detect_text_language(document.content_text) for document in request.documents
    }
    if (
        request.document_language == "unknown"
        or request.document_language not in profile.approved_languages
        or observed_languages != {request.document_language}
    ):
        reasons.add("document_language_outside_approved_scope")
    if request.trade_type not in profile.approved_trade_types:
        reasons.add("trade_type_outside_approved_scope")
    document_types = {item.document_type for item in result.document_results}
    observed_types_by_id = {
        item.document_id: item.document_type for item in result.document_results
    }
    if any(
        document.declared_type is not None
        and observed_types_by_id.get(document.document_id) != document.declared_type
        for document in request.documents
    ):
        reasons.add("declared_document_type_mismatch")
    if (
        not document_types
        or "unknown" in document_types
        or not document_types.issubset(set(profile.approved_document_types))
    ):
        reasons.add("document_type_outside_approved_scope")
    resolved_hsk10s = {
        item.resolved_hsk10 for item in result.retrieval_results if item.resolved_hsk10
    }
    if not resolved_hsk10s or not resolved_hsk10s.issubset(set(profile.approved_hsk10)):
        reasons.add("hsk10_outside_approved_scope")
    if runtime_release.evidence_runtime_v3_sha256 is None:
        reasons.add("evidence_runtime_v3_unreleased")
    if result.retrieval_status != "ok" or result.retrieval_coverage != "complete":
        reasons.add("evidence_not_complete")
    if not has_citable_regulatory_evidence(result):
        # A successful transport/coverage response is not itself affirmative
        # regulatory Evidence.  Until Data publishes a separately verified
        # no-requirement negative assertion, zero citable records cannot
        # authorize automatic completion.
        reasons.add("evidence_citation_missing")
    if supporting_rejections:
        reasons.add("supporting_document_verification_incomplete")
    if require_completed_outcome and (result.status != "completed" or result.requires_human_review):
        reasons.add("deterministic_review_requires_human")
    if result.validation_issues or result.regulatory_findings:
        reasons.add("material_issue_or_finding_present")
    return sorted(reasons)


def _completion_eligibility(
    result: ReviewRunResultV2,
    request: ServiceReviewRunRequestV3,
    runtime_release: RuntimeReleaseIdentityV1,
    *,
    supporting_rejections: list[str],
) -> CompletionEligibilityV1:
    """Evaluate only exact, independently approved automatic-completion scope."""

    profile = load_release_profile()
    reasons = automatic_completion_unmet_reason_codes(
        result,
        request,
        runtime_release,
        supporting_rejections=supporting_rejections,
    )
    document_types = {item.document_type for item in result.document_results}
    resolved_hsk10s = {
        item.resolved_hsk10 for item in result.retrieval_results if item.resolved_hsk10
    }
    scope = [
        "text",
        "KR",
        request.trade_type,
        f"language:{request.document_language}",
        *(f"document:{value}" for value in sorted(document_types)),
        *(f"hsk10:{value}" for value in sorted(resolved_hsk10s)),
    ]
    if reasons:
        return CompletionEligibilityV1(
            eligible=False,
            scope=scope,
            unmet_reason_codes=sorted(reasons),
        )
    return CompletionEligibilityV1(
        eligible=True,
        profile_id=profile.profile_id,
        scope=scope,
        unmet_reason_codes=[],
    )


def _with_release_identity_blockers(
    eligibility: CompletionEligibilityV1,
    reason_codes: Iterable[str],
) -> CompletionEligibilityV1:
    combined = sorted({*eligibility.unmet_reason_codes, *reason_codes})
    if not combined:
        return eligibility
    return CompletionEligibilityV1(
        eligible=False,
        profile_id=None,
        scope=eligibility.scope,
        unmet_reason_codes=combined,
    )


_ADAPTIVE_RECOVERABLE_RELEASE_REASONS = frozenset(
    {
        # These facts may cause an investigation or an explicit human request,
        # but do not grant Claude authority over Outcome or severity.  Every
        # immutable release/scope/model-call identity reason remains blocking.
        "claude_extraction_not_release_clean",
        "evidence_citation_missing",
        "evidence_not_complete",
        "material_issue_or_finding_present",
        "supporting_document_verification_incomplete",
    }
)


def _adaptive_execution_authorized(
    result: ReviewRunResultV2,
    request: ServiceReviewRunRequestV3,
    runtime_release: RuntimeReleaseIdentityV1,
    *,
    supporting_rejections: list[str],
) -> bool:
    reasons = automatic_completion_unmet_reason_codes(
        result,
        request,
        runtime_release,
        supporting_rejections=supporting_rejections,
        require_completed_outcome=False,
    )
    return not (set(reasons) - _ADAPTIVE_RECOVERABLE_RELEASE_REASONS)


def _coordinator_identity_blockers(
    plan: ReviewPlanProposalV1,
    runtime_release: RuntimeReleaseIdentityV1,
) -> list[str]:
    if (
        plan.provider != "anthropic"
        or plan.model_id != runtime_release.model_id
        or plan.prompt_sha256 != runtime_release.coordinator_prompt_sha256
        or plan.output_schema_sha256 != runtime_release.coordinator_schema_sha256
    ):
        return ["coordinator_release_identity_incomplete"]
    return []


def _narrative_identity_blockers(
    narrative: NarrativeReportV1,
    runtime_release: RuntimeReleaseIdentityV1,
) -> list[str]:
    call = narrative.model_call
    if (
        narrative.generation_mode != "claude"
        or call is None
        or call.model_id != runtime_release.model_id
        or call.prompt_sha256 != runtime_release.narrative_prompt_sha256
        or call.output_schema_sha256 != runtime_release.narrative_schema_sha256
    ):
        return ["narrative_release_identity_incomplete"]
    return []


def to_review_run_result_v3(
    result: ReviewRunResultV2,
    request: ServiceReviewRunRequestV3,
    *,
    analyzer: ClaudeDocumentAnalyzer | None = None,
    evidence_runtime_v3_sha256: str | None = None,
    assertion_trust_store: AssertionTrustStoreV1 | None = None,
    supporting_document_rejections: list[str] | None = None,
    active_executor: ActiveBoundedRuntimeExecutor | None = None,
) -> ReviewRunResultV3:
    """Add release-gated adaptive inspection without granting Claude policy authority."""

    def derive(
        current: ReviewRunResultV2,
        rejections: list[str],
    ) -> tuple[
        _FactIndex,
        RuntimeReleaseIdentityV1,
        CompletionEligibilityV1,
        list[ReviewGapV1],
    ]:
        current_facts = build_fact_ledger(current)
        current_release = _runtime_release(
            current,
            evidence_runtime_v3_sha256=evidence_runtime_v3_sha256,
            assertion_trust_store=assertion_trust_store,
        )
        current_eligibility = _completion_eligibility(
            current,
            request,
            current_release,
            supporting_rejections=rejections,
        )
        current_gaps = detect_review_gaps(
            current,
            current_facts,
            extraction_release_approved=current_eligibility.eligible,
        )
        if not current_eligibility.eligible:
            current_gaps.extend(
                _gap(
                    "release_gate",
                    reason_code,
                    (current_facts.outcome_reference,),
                )
                for reason_code in current_eligibility.unmet_reason_codes
            )
        current_gaps.extend(
            _gap(
                "human_approval_required",
                reason_code,
                (current_facts.outcome_reference,),
            )
            for reason_code in rejections
        )
        return (
            current_facts,
            current_release,
            current_eligibility,
            sorted(
                {gap.gap_id: gap for gap in current_gaps}.values(),
                key=lambda item: item.gap_id,
            ),
        )

    rejection_codes = sorted(set(supporting_document_rejections or []))
    if rejection_codes:
        result = result.model_copy(deep=True)
        result.status = "needs_review"
        result.requires_human_review = True
        result.report.overall_status = "needs_review"
    facts, runtime_release, eligibility, gaps = derive(result, rejection_codes)
    investigation_input_ledger = facts.ledger
    initial_gaps = list(gaps)
    if analyzer is None:
        plan = compile_shadow_plan(gaps)
    else:
        try:
            plan = claude_shadow_plan(analyzer, gaps)
        except (LLMError, ValueError):
            plan = compile_shadow_plan(gaps)
    observed_coordinator_blockers = _coordinator_identity_blockers(plan, runtime_release)
    if plan.provider == "anthropic" and observed_coordinator_blockers:
        # A non-release run may still call Claude, but a call outside the exact
        # runtime identity cannot be published as Anthropic-authored audit.
        plan = compile_shadow_plan(gaps)
    adaptive_authorized = _adaptive_execution_authorized(
        result,
        request,
        runtime_release,
        supporting_rejections=rejection_codes,
    )
    coordinator_blockers = observed_coordinator_blockers if adaptive_authorized else []
    eligibility = _with_release_identity_blockers(eligibility, coordinator_blockers)
    gaps.extend(
        _gap("release_gate", reason_code, (facts.outcome_reference,))
        for reason_code in coordinator_blockers
    )
    gaps = sorted({gap.gap_id: gap for gap in gaps}.values(), key=lambda item: item.gap_id)
    initial_gaps = list(gaps)
    action_trace: list[ActionTraceV1] | None = None
    observations: list[ToolObservationV1] | None = None
    active_failure_codes: list[str] = []
    if active_executor is not None and adaptive_authorized and not coordinator_blockers:
        try:
            active_plan = activate_bounded_plan(plan, policy_id=AGENT_POLICY_ID)
        except BoundedToolPolicyError:
            # The normal current state: an unreleased profile and kill switch
            # retain the fully audited shadow plan without executing a Tool.
            pass
        else:
            plan = active_plan
            try:
                active_outcome = active_executor.execute(
                    plan=active_plan,
                    fact_ledger=facts.ledger,
                )
            except (BoundedToolPolicyError, LLMError, ValueError):
                active_failure_codes.append("active_bounded_execution_failed")
            except Exception:
                active_failure_codes.append("active_bounded_execution_failed")
            else:
                result = active_outcome.result
                rejection_codes = sorted(set(active_outcome.supporting_document_rejections))
                if rejection_codes:
                    result = result.model_copy(deep=True)
                    result.status = "needs_review"
                    result.requires_human_review = True
                    result.report.overall_status = "needs_review"
                action_trace = list(active_outcome.action_trace)
                observations = list(active_outcome.observations)
                active_failure_codes.extend(active_outcome.failure_reason_codes)
                facts, runtime_release, eligibility, final_gaps = derive(
                    result,
                    rejection_codes,
                )
                plan_input_reference_ids = {
                    reference_id
                    for action in plan.actions
                    for reference_id in action.input_reference_ids
                }
                facts = _retain_investigation_input_facts(
                    facts,
                    investigation_input_ledger,
                    plan_input_reference_ids,
                )
                final_gap_ids = {item.gap_id for item in final_gaps}
                planned_gap_ids = {item.gap_id for item in plan.actions}
                explicitly_resolved_gap_ids = set(active_outcome.resolved_gap_ids)
                historical = [
                    gap.model_copy(
                        update={
                            "status": (
                                "open"
                                if gap.gap_id in final_gap_ids
                                else (
                                    "resolved"
                                    if gap.gap_id in explicitly_resolved_gap_ids
                                    else "deferred"
                                )
                            )
                        }
                    )
                    for gap in initial_gaps
                    if gap.gap_id in planned_gap_ids
                ]
                gaps = sorted(
                    {gap.gap_id: gap for gap in [*historical, *final_gaps]}.values(),
                    key=lambda item: item.gap_id,
                )
    active_failure_codes = sorted(
        {
            (
                reason_code
                if isinstance(reason_code, str) and reason_code in ACTIVE_FAILURE_REASON_CODES
                else "active_bounded_execution_failed"
            )
            for reason_code in active_failure_codes
        }
    )
    for reason_code in sorted(set(active_failure_codes)):
        gaps.append(
            _gap(
                "release_gate",
                reason_code,
                (facts.outcome_reference,),
            )
        )
    gaps = sorted({gap.gap_id: gap for gap in gaps}.values(), key=lambda item: item.gap_id)
    narrative: NarrativeReportV1 | None = None
    narrative_call: StructuredAgenticResult | None = None
    if analyzer is not None:
        # One bounded retry is permitted only for schema/citation failure. The
        # deterministic report remains the fail-closed public fallback.
        for _attempt in range(2):
            try:
                candidate_narrative, candidate_call = claude_narrative(
                    analyzer,
                    result,
                    facts,
                    release_approved=eligibility.eligible,
                )
                if _narrative_identity_blockers(candidate_narrative, runtime_release):
                    continue
                narrative, narrative_call = candidate_narrative, candidate_call
                break
            except (LLMError, ValueError):
                continue
    if narrative is None:
        narrative = deterministic_narrative(
            result,
            facts,
            release_approved=eligibility.eligible,
        )
    narrative_blockers = (
        _narrative_identity_blockers(narrative, runtime_release) if eligibility.eligible else []
    )
    eligibility = _with_release_identity_blockers(eligibility, narrative_blockers)
    gaps.extend(
        _gap("release_gate", reason_code, (facts.outcome_reference,))
        for reason_code in narrative_blockers
    )
    gaps = sorted({gap.gap_id: gap for gap in gaps}.values(), key=lambda item: item.gap_id)
    if action_trace is None or observations is None:
        action_trace, observations = execute_bounded_plan(plan, narrative, narrative_call)
    open_gaps = any(gap.status != "resolved" for gap in gaps)
    action_failed = any(item.status != "succeeded" for item in observations)
    if result.status == "blocked":
        execution_status = "failed"
        disposition = "blocked"
    else:
        execution_status = (
            "partial"
            if result.model_errors or active_failure_codes or action_failed
            else "succeeded"
        )
        disposition = (
            "auto_review_complete"
            if eligibility.eligible and not open_gaps and not action_failed
            else "human_review_required"
        )
    if PROMPT_INJECTION_REASON_CODE in eligibility.unmet_reason_codes:
        assessment = "indeterminate"
    elif result.validation_issues or result.regulatory_findings:
        assessment = "issues_detected"
    elif (
        result.retrieval_status == "ok"
        and result.retrieval_coverage == "complete"
        and has_citable_regulatory_evidence(result)
    ):
        assessment = "no_issue_detected"
    else:
        assessment = "indeterminate"
    required_actions = _required_actions(
        result,
        supporting_rejections=rejection_codes,
        release_approved=eligibility.eligible,
        active_failure_codes=active_failure_codes,
    )
    payload = result.model_dump(mode="python", warnings=False)
    payload.update(
        {
            "schema_version": "3.0",
            "execution_status": execution_status,
            "assessment": assessment,
            "review_disposition": disposition,
            "assurance_level": "production" if eligibility.eligible else "production_candidate",
            "completion_eligibility": eligibility,
            "required_actions": required_actions,
            "fact_ledger": facts.ledger,
            "review_gaps": gaps,
            "review_plan": plan,
            "narrative_report": narrative,
            "action_trace": action_trace,
            "observations": observations,
            "runtime_release": runtime_release,
            "input_manifest_sha256": canonical_json_sha256(request),
            "result_sha256": "0" * 64,
        }
    )
    payload["result_sha256"] = canonical_json_sha256(payload)
    return ReviewRunResultV3.model_validate(payload)


@dataclass(frozen=True, slots=True)
class _AnswerClaim:
    claim_id: str
    text: str
    citations: tuple[CitationReferenceV1, ...]


@dataclass(frozen=True, slots=True)
class _GroundedAnswerPlan:
    status: Literal["answered", "insufficient_evidence", "out_of_scope"]
    claims: tuple[_AnswerClaim, ...] = ()
    refusal_text: str | None = None


def _answer_claim(text: str, citations: Iterable[CitationReferenceV1]) -> _AnswerClaim:
    text = " ".join(text.split())
    frozen_citations = tuple(citations)
    payload = {
        "text": text,
        "citations": [item.model_dump(mode="json") for item in frozen_citations],
    }
    return _AnswerClaim(
        claim_id=stable_object_id("answer_claim", payload),
        text=text,
        citations=frozen_citations,
    )


def _bounded_answer_claims(claims: Iterable[_AnswerClaim]) -> tuple[_AnswerClaim, ...]:
    selected: list[_AnswerClaim] = []
    rendered_size = 0
    for claim in claims:
        additional = len(claim.text) + (1 if selected else 0)
        if len(selected) >= 20 or rendered_size + additional > 7600:
            break
        selected.append(claim)
        rendered_size += additional
    return tuple(selected)


def _deterministic_grounded_answer_plan(question: GroundedQuestionV1) -> _GroundedAnswerPlan:
    result = question.review_result
    current_semantics = reconstruct_v3_current_semantics(result)
    current_facts = {
        str(entry["reference_id"]): entry for entry in current_semantics.current_fact_entries
    }
    lowered = question.question.casefold()
    out_of_scope_tokens = (
        "weather",
        "stock",
        "news",
        "날씨",
        "주식",
        "뉴스",
        "요리",
    )
    if any(token in lowered for token in out_of_scope_tokens):
        return _GroundedAnswerPlan(
            status="out_of_scope",
            refusal_text="이 질문은 현재 통관 Case와 인용된 근거의 범위를 벗어납니다.",
        )
    question_tokens = {
        token.casefold()
        for token in _QUESTION_TOKEN.findall(question.question)
        if token.casefold() not in _QUESTION_STOP_WORDS
    }
    issue_intent = _matches_question_intent(question_tokens, _ISSUE_INTENT_KEYWORDS)
    regulation_intent = _matches_question_intent(question_tokens, _REGULATION_INTENT_KEYWORDS)
    outcome_intent = _matches_question_intent(question_tokens, _OUTCOME_INTENT_KEYWORDS)
    field_intent = _matches_question_intent(question_tokens, _FIELD_INTENT_KEYWORDS)
    case_intent = issue_intent or regulation_intent or outcome_intent or field_intent
    if not case_intent:
        return _GroundedAnswerPlan(
            status="out_of_scope",
            refusal_text="이 질문은 현재 통관 Case의 문서·규정·심사 결과 범위를 벗어납니다.",
        )
    claims: list[_AnswerClaim] = []
    if issue_intent:
        for finding, reference_id in zip(
            result.regulatory_findings,
            current_semantics.finding_reference_ids,
            strict=True,
        ):
            if reference_id not in current_facts:
                continue
            citations = [_citation("finding", reference_id)]
            citations.extend(_citation("evidence", item) for item in finding.evidence_ids)
            claims.append(_answer_claim(f"{finding.finding_code}: {finding.message}", citations))
        for issue, reference_id in zip(
            result.validation_issues,
            current_semantics.issue_reference_ids,
            strict=True,
        ):
            if reference_id in current_facts:
                claims.append(
                    _answer_claim(
                        f"{issue.issue_code}: {issue.message}",
                        [_citation("issue", reference_id)],
                    )
                )
    elif regulation_intent and result.rag_evidence:
        for evidence in result.rag_evidence:
            claims.append(
                _answer_claim(
                    f"{evidence.title}: {evidence.text[:1000]}",
                    [_citation("evidence", evidence.evidence_id)],
                )
            )
    elif outcome_intent:
        claims.append(
            _answer_claim(
                f"현재 사전심사 상태는 {result.review_disposition}입니다.",
                [_citation("outcome", current_semantics.outcome_reference_id)],
            )
        )
    elif field_intent:
        specific_tokens = question_tokens.difference({"문서", "품목", "송장", "선하", "hsk", "hs"})
        field_facts = [
            entry
            for entry in current_semantics.current_fact_entries
            if entry["kind"] == "field_source"
        ]
        matching = [
            entry
            for entry in field_facts
            if (
                not specific_tokens
                or any(
                    token in f"{entry['label']} {entry['value']}".casefold()
                    for token in specific_tokens
                )
            )
        ]
        if not matching:
            return _GroundedAnswerPlan(
                status="insufficient_evidence",
                refusal_text=(
                    "현재 Review Result에 질문과 직접 연결되는 문서 필드 근거가 없습니다."
                ),
            )
        claims.extend(
            _answer_claim(
                f"{reference['label']}: {reference['value']}",
                [_citation("field_source", str(reference["reference_id"]))],
            )
            for reference in matching
        )
    selected = _bounded_answer_claims(claims)
    if not selected:
        return _GroundedAnswerPlan(
            status="insufficient_evidence",
            refusal_text=("현재 Review Result에 질문에 답할 수 있는 문서·규정 근거가 없습니다."),
        )
    return _GroundedAnswerPlan(status="answered", claims=selected)


def _render_answer_plan(
    question_id: str,
    plan: _GroundedAnswerPlan,
) -> GroundedAnswerV1:
    if plan.status != "answered":
        if plan.refusal_text is None:
            raise ValueError("grounded answer refusal text is missing")
        return GroundedAnswerV1(
            question_id=question_id,
            answer=plan.refusal_text,
            status=plan.status,
            generation_mode="deterministic_fallback",
        )
    citations = [citation for claim in plan.claims for citation in claim.citations]
    unique_citations = list(dict.fromkeys((item.kind, item.reference_id) for item in citations))
    return GroundedAnswerV1(
        question_id=question_id,
        answer="\n".join(claim.text for claim in plan.claims),
        status="answered",
        citations=[_citation(kind, reference_id) for kind, reference_id in unique_citations],
        generation_mode="deterministic_fallback",
    )


def _deterministic_grounded_answer(question: GroundedQuestionV1) -> GroundedAnswerV1:
    plan = _deterministic_grounded_answer_plan(question)
    answer = _render_answer_plan(question.question_id, plan)
    result = question.review_result
    valid = {(item.kind, item.reference_id) for item in result.fact_ledger.entries}
    if any((item.kind, item.reference_id) not in valid for item in answer.citations):
        return GroundedAnswerV1(
            question_id=question.question_id,
            answer="인용 검증을 통과하지 못해 답변을 제공할 수 없습니다.",
            status="insufficient_evidence",
            generation_mode="deterministic_fallback",
        )
    return answer


def _rewrite_grounded_answer(
    plan: _GroundedAnswerPlan,
    values: Any,
    *,
    question_id: str,
    call: StructuredAgenticResult,
) -> GroundedAnswerV1:
    if plan.status != "answered" or not isinstance(values, list):
        raise ValueError("Case answer claim collection is invalid")
    canonical = {claim.claim_id: claim for claim in plan.claims}
    selections: dict[str, str] = {}
    for value in values:
        if not isinstance(value, dict) or set(value) != {"source_claim_id", "frame"}:
            raise ValueError("Case answer frame selection is invalid")
        claim_id = value.get("source_claim_id")
        frame = value.get("frame")
        if not isinstance(claim_id, str) or not isinstance(frame, str) or claim_id in selections:
            raise ValueError("Case answer frame selection identity is invalid")
        source = canonical.get(claim_id)
        if source is None:
            raise ValueError("Case answer references an unknown canonical claim")
        _render_grounded_claim(source.text, frame, section="answer")
        selections[claim_id] = frame
    if set(selections) != set(canonical):
        raise ValueError(
            "Case answer frame selection must cover every canonical claim exactly once"
        )
    citations = [citation for claim in plan.claims for citation in claim.citations]
    unique_citations = list(dict.fromkeys((item.kind, item.reference_id) for item in citations))
    return GroundedAnswerV1(
        question_id=question_id,
        answer="\n".join(
            _render_grounded_claim(
                claim.text,
                selections[claim.claim_id],
                section="answer",
            )
            for claim in plan.claims
        ),
        status="answered",
        citations=[_citation(kind, reference_id) for kind, reference_id in unique_citations],
        generation_mode="claude",
        model_call=_call_audit(CASE_ANSWER_PROMPT_ID, call),
    )


def validate_grounded_answer_projection(
    question: GroundedQuestionV1,
    answer: GroundedAnswerV1,
) -> GroundedAnswerV1:
    """Validate a Case answer against its code-owned question/result projection.

    ``GroundedAnswerV1`` alone intentionally carries no duplicate ReviewResult.
    Therefore the service boundary (and Backend consumer) must validate the
    answer together with the exact ``GroundedQuestionV1`` request.
    """

    plan = _deterministic_grounded_answer_plan(question)
    canonical = _render_answer_plan(question.question_id, plan)
    if answer.question_id != question.question_id or answer.status != canonical.status:
        raise ValueError("Case answer identity and status must match the canonical plan")
    if answer.citations != canonical.citations:
        raise ValueError("Case answer citations must exactly match the canonical plan")
    if plan.status != "answered":
        if (
            answer.generation_mode != "deterministic_fallback"
            or answer.model_call is not None
            or answer.answer != canonical.answer
        ):
            raise ValueError("refused Case answer must use the exact deterministic response")
        return answer

    rendered_claims = answer.answer.split("\n")
    if len(rendered_claims) != len(plan.claims):
        raise ValueError("Case answer must cover every canonical claim exactly once")
    if answer.generation_mode == "deterministic_fallback":
        if rendered_claims != [claim.text for claim in plan.claims]:
            raise ValueError("deterministic Case answer must use exact canonical claim text")
        return answer

    call = answer.model_call
    runtime_release = question.review_result.runtime_release
    if (
        call is None
        or call.task_id != "customs_case_answer"
        or call.provider != "anthropic"
        or call.model_id != runtime_release.model_id
        or call.prompt_sha256 != runtime_release.chat_prompt_sha256
        or call.output_schema_sha256 != runtime_release.chat_schema_sha256
        or call.input_tokens <= 0
        or call.output_tokens <= 0
    ):
        raise ValueError("Claude Case answer requires exact runtime release identity")
    if any(
        rendered not in allowed_framed_claim_texts("answer", source.text)
        for rendered, source in zip(rendered_claims, plan.claims, strict=True)
    ):
        raise ValueError("Claude Case answer must use code-owned claim frames")
    return answer


def answer_grounded_question(
    question: GroundedQuestionV1,
    *,
    analyzer: ClaudeDocumentAnalyzer | None = None,
) -> GroundedAnswerV1:
    plan = _deterministic_grounded_answer_plan(question)
    fallback = _deterministic_grounded_answer(question)
    if analyzer is None or fallback.status != "answered":
        return validate_grounded_answer_projection(question, fallback)
    context = {
        "question": question.question,
        "canonical_claims": [
            {
                "source_claim_id": claim.claim_id,
                "canonical_text": claim.text,
                "allowed_frames": list(SAFE_CLAIM_FRAMES["answer"]),
            }
            for claim in plan.claims
        ],
        "fixed_status": fallback.status,
    }
    for _attempt in range(2):
        try:
            call = analyzer.run_bounded_agentic_task(
                task_id=CASE_ANSWER_PROMPT_ID,
                system_prompt=load_case_answer_system_prompt(),
                untrusted_context=context,
                output_schema=_ANSWER_OUTPUT_SCHEMA,
                max_tokens=2048,
            )
            runtime_release = question.review_result.runtime_release
            if (
                call.model_id != runtime_release.model_id
                or call.prompt_sha256 != runtime_release.chat_prompt_sha256
                or call.output_schema_sha256 != runtime_release.chat_schema_sha256
                or call.semantic_output_schema_sha256 != agentic_semantic_schema_hashes()["chat"]
            ):
                raise ValueError("Case answer release identity mismatch")
            answer = _rewrite_grounded_answer(
                plan,
                call.value.get("claims"),
                question_id=question.question_id,
                call=call,
            )
            return validate_grounded_answer_projection(question, answer)
        except (LLMError, ValueError):
            continue
    return validate_grounded_answer_projection(question, fallback)


def verified_requirement_codes_by_item(
    documents: list[SupportingDocumentV1],
    *,
    case_id: str,
    review_as_of: date,
    assertion_cutoff_at: datetime | None,
    retrieval_results: list[Any],
    policy_bindings: dict[str, dict[str, Any]],
    assertion_verifier: TrustedAssertionVerifier | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """Bind supporting documents to exact item, HSK, policy and trusted assertion."""

    item_hsk: dict[str, str] = {}
    for result in retrieval_results:
        item_id = (
            result.get("item_id") if isinstance(result, dict) else getattr(result, "item_id", None)
        )
        resolved_hsk10 = (
            result.get("resolved_hsk10")
            if isinstance(result, dict)
            else getattr(result, "resolved_hsk10", None)
        )
        if isinstance(item_id, str) and isinstance(resolved_hsk10, str):
            item_hsk[item_id] = resolved_hsk10
    verified: dict[str, set[str]] = {}
    rejected: list[str] = []
    for document in documents:
        in_force = document.issued_on <= review_as_of and (
            document.valid_until is None or document.valid_until >= review_as_of
        )
        policy = policy_bindings.get(document.document_code)
        if not in_force:
            rejected.append(f"{document.document_id}:supporting_document_not_in_force")
            continue
        if (
            not isinstance(policy, dict)
            or not policy.get("approved")
            or policy.get("required_document_code") != document.document_code
            or not isinstance(policy.get("policy_source_sha256"), str)
        ):
            rejected.append(f"{document.document_id}:requirement_policy_unapproved")
            continue
        valid_assertions = []
        for assertion in document.verification_assertions:
            cutoff = assertion_cutoff_at
            locally_well_formed = (
                assertion.asserted_at.tzinfo is not None
                and assertion.subject_document_id == document.document_id
                and assertion.source_reference_sha256 == document.source_sha256
                and assertion.asserted_at.date() >= document.issued_on
                and cutoff is not None
                and cutoff.tzinfo is not None
                and assertion.asserted_at <= cutoff
                and assertion.method != "test_issuer"
                and (
                    (
                        assertion.method == "trusted_connector"
                        and assertion.authority_id.startswith("connector:")
                    )
                    or (
                        assertion.method == "human_approver"
                        and re.fullmatch(
                            r"backend:principal-sha256:[a-f0-9]{64}",
                            assertion.authority_id,
                        )
                        is not None
                        and assertion.authority_release == "backend-human-decision-policy/v1"
                    )
                )
            )
            if not locally_well_formed or assertion_verifier is None or cutoff is None:
                continue
            try:
                externally_verified = assertion_verifier(
                    assertion=assertion,
                    document=document,
                    policy=policy,
                    case_id=case_id,
                    assertion_cutoff_at=cutoff,
                )
            except Exception:
                externally_verified = False
            if externally_verified is True:
                valid_assertions.append(assertion)
        policy_name = policy.get("verification_policy")
        if policy_name == "external_verification_required":
            valid_assertions = [
                item for item in valid_assertions if item.method == "trusted_connector"
            ]
        elif policy_name == "human_only":
            valid_assertions = [
                item for item in valid_assertions if item.method == "human_approver"
            ]
        elif policy_name == "content_match_required":
            valid_assertions = [
                item for item in valid_assertions if item.method == "trusted_connector"
            ]
        elif policy_name != "presence_only":
            valid_assertions = []
        if not valid_assertions:
            rejected.append(f"{document.document_id}:trusted_verification_assertion_missing")
            continue
        item_failed = False
        for item_id in document.subject_item_ids:
            resolved_hsk10 = item_hsk.get(item_id)
            if resolved_hsk10 is None or resolved_hsk10 not in document.hs_codes:
                rejected.append(
                    f"{document.document_id}:{item_id}:supporting_document_scope_mismatch"
                )
                item_failed = True
                continue
            verified.setdefault(item_id, set()).add(document.document_code)
        if item_failed:
            for item_id in document.subject_item_ids:
                verified.get(item_id, set()).discard(document.document_code)
    return (
        {item_id: sorted(codes) for item_id, codes in sorted(verified.items()) if codes},
        sorted(set(rejected)),
    )


def presented_requirement_codes_by_item(
    documents: list[SupportingDocumentV1],
    *,
    retrieval_results: list[Any],
) -> dict[str, list[str]]:
    """Project exact item/HSK-bound presence without claiming document trust."""

    item_hsk: dict[str, str] = {}
    for result in retrieval_results:
        item_id = (
            result.get("item_id") if isinstance(result, dict) else getattr(result, "item_id", None)
        )
        resolved_hsk10 = (
            result.get("resolved_hsk10")
            if isinstance(result, dict)
            else getattr(result, "resolved_hsk10", None)
        )
        if isinstance(item_id, str) and isinstance(resolved_hsk10, str):
            item_hsk[item_id] = resolved_hsk10
    presented: dict[str, set[str]] = {}
    for document in documents:
        for item_id in document.subject_item_ids:
            resolved_hsk10 = item_hsk.get(item_id)
            if resolved_hsk10 is not None and resolved_hsk10 in document.hs_codes:
                presented.setdefault(item_id, set()).add(document.document_code)
    return {item_id: sorted(codes) for item_id, codes in sorted(presented.items()) if codes}


def assert_result_hash(result: ReviewRunResultV3) -> None:
    if result.result_sha256 != review_result_v3_sha256(result):
        raise ValueError("stale or mutated V3 result")


__all__ = [
    "AGENT_POLICY_ID",
    "MAX_AGENT_CYCLES",
    "MAX_TOOL_CALLS",
    "TOOL_REGISTRY_ID",
    "TrustedAssertionVerifier",
    "answer_grounded_question",
    "assert_result_hash",
    "build_fact_ledger",
    "compile_shadow_plan",
    "detect_review_gaps",
    "deterministic_narrative",
    "execute_bounded_plan",
    "presented_requirement_codes_by_item",
    "runtime_release_identity",
    "to_review_run_result_v3",
    "verified_requirement_codes_by_item",
    "validate_grounded_answer_projection",
]
