from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Literal, Mapping

from dotenv import dotenv_values

from customs_ai_agent.adapters.llm import (
    ClaudeDocumentAnalyzer,
    DocumentAnalysis,
    ModelCallMetadata,
    StructuredAgenticResult,
)
from customs_ai_agent.schemas.models import (
    EvidenceRetrievalRequest,
    EvidenceRetrievalResponse,
    RetrievalErrorDetail,
)
from customs_ai_agent.schemas.v3 import (
    ServiceReviewRunRequestV3,
    canonical_json_sha256,
)
from tools.agent_execution_inspector.evidence_v3_conformance import (
    LoopbackEvidenceV3DiagnosticRetriever,
)
from tools.local_uat_agent_service.app import (
    LocalUatDiagnosticRuntime,
    build_local_uat_runtime,
    load_local_uat_manifest,
)


FaultMode = Literal["none", "first_partial", "always_partial"]
MAX_REQUEST_BYTES = 32 * 1024 * 1024
_SAFE_REASON_CODE = re.compile(r"^[a-z0-9_]{1,128}$")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write_private_summary_to_stdout(value: Mapping[str, Any]) -> None:
    """Write the private verification bundle only to the supervising process pipe."""

    if not _stdout_is_supervising_pipe():
        raise RuntimeError("private adaptive summary requires a supervising stdout pipe")
    encoded = _canonical_bytes(value) + b"\n"
    binary = getattr(sys.stdout, "buffer", None)
    if binary is not None:
        binary.write(encoded)
        binary.flush()
        return
    sys.stdout.write(encoded.decode("utf-8"))
    sys.stdout.flush()


def _stdout_is_supervising_pipe() -> bool:
    try:
        descriptor = sys.stdout.fileno()
        if os.name == "nt":
            import ctypes
            import msvcrt

            handle = msvcrt.get_osfhandle(descriptor)
            return ctypes.windll.kernel32.GetFileType(handle) == 3  # type: ignore[attr-defined]
        return stat.S_ISFIFO(os.fstat(descriptor).st_mode)
    except (AttributeError, OSError, ValueError):
        return False


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def _decode_request_bytes(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("stdin request is empty or exceeds the bounded input size")
    value = json.loads(
        raw,
        parse_constant=_reject_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(value, dict):
        raise ValueError("stdin request must contain one JSON object")
    return value


def _load_request_from_stdin() -> dict[str, Any]:
    stream = getattr(sys.stdin, "buffer", sys.stdin)
    raw = stream.read(MAX_REQUEST_BYTES + 1)
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return _decode_request_bytes(raw)


def _identifier_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _InstrumentedAnalyzer:
    """Record safe proof that the pinned Claude transport was actually called.

    The private supervising Harness needs an independent count of the initial
    four document calls and any bounded span rereads. ReviewResult retains only
    the final per-document call, so it cannot prove those additional calls by
    itself. This wrapper records only identities, hashes, token counts, and
    latency. It never records a document, prompt, model response, or provider
    request/message identifier in plaintext.
    """

    def __init__(self, actual: ClaudeDocumentAnalyzer) -> None:
        self._actual = actual
        self._records: list[dict[str, Any]] = []
        self._closed = False

    @property
    def model_id(self) -> str:
        return self._actual.model_id

    @property
    def call_count(self) -> int:
        return len(self._records)

    def records_since(self, index: int) -> list[dict[str, Any]]:
        if index < 0 or index > len(self._records):
            raise ValueError("Claude audit cursor is outside the recorded call range")
        return [dict(item) for item in self._records[index:]]

    def _record_metadata(
        self,
        metadata: ModelCallMetadata,
        *,
        document_id_sha256: str,
        document_input_sha256: str,
        additional: bool,
    ) -> None:
        value: dict[str, Any] = {
            "sequence": len(self._records) + 1,
            "call_kind": "document_analysis",
            "additional_model_call": additional,
            "document_id_sha256": document_id_sha256,
            "document_input_sha256": document_input_sha256,
            "model_id": metadata.model_id,
            "prompt_id": metadata.prompt_id,
            "prompt_version": metadata.prompt_version,
            "prompt_sha256": metadata.prompt_sha256,
            "output_schema_id": metadata.output_schema_id,
            "output_schema_version": metadata.output_schema_version,
            "output_schema_sha256": metadata.output_schema_sha256,
            "request_id_sha256": _identifier_sha256(metadata.request_id),
            "message_id_sha256": _identifier_sha256(metadata.message_id),
            "input_tokens": metadata.input_tokens,
            "output_tokens": metadata.output_tokens,
            "latency_ms": metadata.latency_ms,
        }
        value["record_sha256"] = canonical_json_sha256(value)
        self._records.append(value)

    def analyze(self, document: Mapping[str, Any]) -> DocumentAnalysis:
        document_id = document.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("instrumented Claude analysis requires a document_id")
        analysis = self._actual.analyze(document)
        document_id_sha256 = _identifier_sha256(document_id)
        document_input_sha256 = canonical_json_sha256(dict(document))
        self._record_metadata(
            analysis.metadata,
            document_id_sha256=document_id_sha256,
            document_input_sha256=document_input_sha256,
            additional=False,
        )
        for metadata in analysis.additional_model_calls:
            self._record_metadata(
                metadata,
                document_id_sha256=document_id_sha256,
                document_input_sha256=document_input_sha256,
                additional=True,
            )
        return analysis

    def run_bounded_agentic_task(
        self,
        *,
        task_id: str,
        system_prompt: str,
        untrusted_context: dict[str, Any],
        output_schema: dict[str, Any],
        max_tokens: int = 2048,
    ) -> StructuredAgenticResult:
        result = self._actual.run_bounded_agentic_task(
            task_id=task_id,
            system_prompt=system_prompt,
            untrusted_context=untrusted_context,
            output_schema=output_schema,
            max_tokens=max_tokens,
        )
        raw_failure_codes = untrusted_context.get("previous_verification_failures", [])
        if not isinstance(raw_failure_codes, list) or len(raw_failure_codes) > 32:
            raise ValueError("verification failure-code audit is not bounded")
        previous_verification_failure_codes = sorted(
            {
                item
                for item in raw_failure_codes
                if isinstance(item, str) and _SAFE_REASON_CODE.fullmatch(item)
            }
        )
        if len(previous_verification_failure_codes) != len(set(raw_failure_codes)):
            raise ValueError("verification failure-code audit contains an unsafe code")
        value: dict[str, Any] = {
            "sequence": len(self._records) + 1,
            "call_kind": "agentic_task",
            "task_id": task_id,
            "untrusted_context_sha256": canonical_json_sha256(untrusted_context),
            "previous_verification_failure_codes": (previous_verification_failure_codes),
            "model_id": result.model_id,
            "prompt_sha256": result.prompt_sha256,
            "output_schema_sha256": result.output_schema_sha256,
            "semantic_output_schema_sha256": result.semantic_output_schema_sha256,
            "request_id_sha256": _identifier_sha256(result.request_id),
            "message_id_sha256": _identifier_sha256(result.message_id),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
        }
        value["record_sha256"] = canonical_json_sha256(value)
        self._records.append(value)
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._actual.close()


class _ActualRetrieverFaultControl:
    """Inject a response-state fault only after a successful real HTTP POST.

    Readiness, all seventeen identity headers, snapshot validation, policy
    bindings and the actual Store query remain owned by the wrapped loopback
    retriever. The fault is a test-only observation returned to the Agent; it
    cannot change the Provider or Store.
    """

    def __init__(
        self,
        actual: LoopbackEvidenceV3DiagnosticRetriever,
        *,
        fault_mode: FaultMode,
    ) -> None:
        self._actual = actual
        self._fault_mode = fault_mode
        self._retrieve_count = 0
        self._request_ids: list[str] = []

    @property
    def expected_identity(self):
        return self._actual.expected_identity

    @property
    def successful_retrieval_post_count(self) -> int:
        return self._actual.successful_retrieval_post_count

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(self._request_ids)

    def preflight(self, *args: Any, **kwargs: Any):
        return self._actual.preflight(*args, **kwargs)

    def policy_bindings(self, request_id: str) -> dict[str, dict[str, Any]]:
        return self._actual.policy_bindings(request_id)

    def retrieve(
        self,
        request: EvidenceRetrievalRequest,
    ) -> EvidenceRetrievalResponse:
        response = self._actual.retrieve(request)
        self._request_ids.append(request.request_id)
        self._retrieve_count += 1
        inject = self._fault_mode == "always_partial" or (
            self._fault_mode == "first_partial" and self._retrieve_count == 1
        )
        if not inject:
            return response
        fault = RetrievalErrorDetail(
            code="uat_injected_partial_after_actual_http",
            message="The UAT control changed only the observed response state.",
            retryable=True,
        )
        values = response.model_dump(mode="python", warnings=False)
        values["status"] = "partial"
        values["errors"] = [*values["errors"], fault.model_dump(mode="python")]
        values["results"] = [
            {
                **item,
                "coverage": "partial",
                "errors": [*item["errors"], fault.model_dump(mode="python")],
            }
            for item in values["results"]
        ]
        return EvidenceRetrievalResponse.model_validate(values)


def _evidence_token(env_file: Path, name: str) -> str:
    process_value = os.environ.get(name)
    file_value = dotenv_values(env_file, verbose=False).get(name)
    value = process_value if process_value is not None else file_value
    if not isinstance(value, str) or len(value) < 32:
        raise ValueError(f"{name} is missing or invalid")
    return value


def _run_control(
    runtime: LocalUatDiagnosticRuntime,
    actual: LoopbackEvidenceV3DiagnosticRetriever,
    analyzer: _InstrumentedAnalyzer,
    request: ServiceReviewRunRequestV3,
    *,
    control_id: str,
    fault_mode: FaultMode,
) -> tuple[dict[str, Any], bool]:
    suffix = {
        "field-uncertainty-reread": "field-reread",
        "evidence-incomplete-retry": "evidence-retry",
        "budget-exhaustion-human-review": "budget-exhaustion",
    }[control_id]
    control_request = ServiceReviewRunRequestV3.model_validate(
        {
            **request.model_dump(mode="json", warnings=False),
            "review_run_id": f"{request.review_run_id}-{suffix}",
            "case_id": f"{request.case_id}-{suffix}",
        }
    )
    controlled = _ActualRetrieverFaultControl(actual, fault_mode=fault_mode)
    runtime._core._retriever_v3 = controlled  # type: ignore[assignment] # noqa: SLF001
    runtime._retriever = controlled  # type: ignore[assignment] # noqa: SLF001
    before_posts = actual.successful_retrieval_post_count
    before_claude_calls = analyzer.call_count
    try:
        envelope = runtime.review_experience_json(
            _canonical_bytes(control_request.model_dump(mode="json", warnings=False))
        )
    finally:
        runtime._core._retriever_v3 = actual  # noqa: SLF001
        runtime._retriever = actual  # noqa: SLF001
    result = envelope.canonical_result
    after_posts = actual.successful_retrieval_post_count
    tool_status = [
        {
            "tool_name": trace.tool_name,
            "status": trace.status,
            "cycle": trace.cycle,
            "duration_ms": trace.duration_ms,
        }
        for trace in result.action_trace
    ]
    observed_tools = {item["tool_name"] for item in tool_status}
    active_failure_codes = {
        action.action_code
        for action in result.required_actions
        if action.action_code in {"active_tool_execution_incomplete", "agent_budget_exhausted"}
    }
    field_verified = "document.reread_span/v1" in observed_tools and any(
        item["tool_name"] == "document.reread_span/v1" and item["status"] == "succeeded"
        for item in tool_status
    )
    evidence_verified = (
        "evidence.retrieve/v3" in observed_tools
        and any(
            item["tool_name"] == "evidence.retrieve/v3" and item["status"] == "succeeded"
            for item in tool_status
        )
        and result.retrieval_status == "ok"
        and result.retrieval_coverage == "complete"
    )
    safe_failure_verified = (
        result.review_disposition == "human_review_required"
        and result.execution_status == "partial"
        and "active_tool_execution_incomplete" in active_failure_codes
        and result.review_disposition != "auto_review_complete"
    )
    budget_exhaustion_verified = (
        "agent_budget_exhausted" in active_failure_codes
        and sum(
            item["tool_name"] == "evidence.retrieve/v3" and item["status"] == "failed"
            for item in tool_status
        )
        == 2
        and {
            item["cycle"]
            for item in tool_status
            if item["tool_name"] == "evidence.retrieve/v3" and item["status"] == "failed"
        }
        == {0, 1}
    )
    actual_posts = after_posts - before_posts
    claude_call_audit = analyzer.records_since(before_claude_calls)
    document_call_hashes = [
        item["document_input_sha256"]
        for item in claude_call_audit
        if item["call_kind"] == "document_analysis" and item["additional_model_call"] is False
    ]
    document_call_frequencies = sorted(
        document_call_hashes.count(item) for item in set(document_call_hashes)
    )
    field_gap_open = any(
        gap.gap_type == "field_uncertainty" and gap.status != "resolved"
        for gap in result.review_gaps
    )
    passed = {
        "field-uncertainty-reread": (
            field_verified
            and field_gap_open
            and actual_posts == 1
            and len(document_call_hashes) == 5
            and document_call_frequencies == [1, 1, 1, 2]
        ),
        "evidence-incomplete-retry": evidence_verified and actual_posts >= 2,
        "budget-exhaustion-human-review": (
            safe_failure_verified and budget_exhaustion_verified and actual_posts >= 3
        ),
    }[control_id]
    private_verification_bundle = {
        "control_id": control_id,
        "control_request": control_request.model_dump(mode="json", warnings=False),
        "retrieval_request_ids": list(controlled.request_ids),
        "canonical_result": result.model_dump(mode="json", warnings=False),
        "experience_envelope": envelope.model_dump(mode="json", warnings=False),
        "claude_call_audit": claude_call_audit,
    }
    return private_verification_bundle, passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run UAT bounded-active controls with actual Claude and the pinned "
            "loopback three-law Evidence Provider. The private canonical bundle is emitted "
            "only through stdout for its supervising Harness to validate in memory."
        )
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--env-file", required=True, type=Path)
    request_source = parser.add_mutually_exclusive_group(required=True)
    request_source.add_argument("--request", type=Path)
    request_source.add_argument(
        "--request-stdin",
        action="store_true",
        help="Read one ServiceReviewRunRequestV3 JSON object from stdin.",
    )
    parser.add_argument(
        "--evidence-token-env",
        default="EVIDENCE_SERVICE_TOKEN",
    )
    parser.add_argument(
        "--output-summary",
        required=True,
        choices=("-",),
        help="Must be '-' so the private canonical verification bundle is never persisted.",
    )
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required because this runner invokes Claude and Evidence HTTP")
    manifest = load_local_uat_manifest(args.manifest.expanduser().resolve(strict=True))
    if (
        manifest.evidence_mode != "preserved-3law-v3-scenario-eval"
        or not manifest.enhanced_experience_enabled
        or not manifest.uat_adaptive_active
    ):
        parser.error("manifest must enable the exact scenario-eval adaptive UAT profile")
    env_file = args.env_file.expanduser().resolve(strict=True)
    request_payload = (
        _load_request_from_stdin()
        if args.request_stdin
        else _load_object(args.request.expanduser().resolve(strict=True))
    )
    request = ServiceReviewRunRequestV3.model_validate(request_payload)
    token = _evidence_token(env_file, args.evidence_token_env)
    runtime = build_local_uat_runtime(
        manifest,
        env_file,
        evidence_service_token=token,
    )
    actual_retriever = runtime._retriever  # noqa: SLF001
    actual_analyzer = runtime._core._analyzer  # noqa: SLF001
    instrumented_analyzer = _InstrumentedAnalyzer(actual_analyzer)
    runtime._core._analyzer = instrumented_analyzer  # type: ignore[assignment] # noqa: SLF001
    try:
        control_runs = [
            _run_control(
                runtime,
                actual_retriever,
                instrumented_analyzer,
                request,
                control_id="field-uncertainty-reread",
                fault_mode="none",
            ),
            _run_control(
                runtime,
                actual_retriever,
                instrumented_analyzer,
                request,
                control_id="evidence-incomplete-retry",
                fault_mode="first_partial",
            ),
            _run_control(
                runtime,
                actual_retriever,
                instrumented_analyzer,
                request,
                control_id="budget-exhaustion-human-review",
                fault_mode="always_partial",
            ),
        ]
    finally:
        runtime.close()
    controls = [item for item, _passed in control_runs]
    control_passes = [passed for _item, passed in control_runs]
    summary: dict[str, Any] = {
        "schema_version": "customs-uat-adaptive-live-private-summary/v2",
        "manifest_sha256": manifest.manifest_sha256,
        "evidence_runtime_v3_sha256": (
            manifest.evidence_runtime_identity.runtime_release_v3_sha256
        ),
        "request_manifest_sha256": canonical_json_sha256(request),
        "controls": controls,
        "private_bundle_contains_document_or_model_prose": True,
        "known_gate_limitations": [],
    }
    summary["summary_sha256"] = canonical_json_sha256(summary)
    _write_private_summary_to_stdout(summary)
    return 0 if all(control_passes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
