"""Run actual Claude -> AgentRuntime V3 -> loopback Evidence V3 conformance.

This is deliberately separate from the three-law V2 Inspector. It is a
non-release, loopback-only interoperability runner: a caller supplies the
provider identity explicitly, all seventeen Evidence bindings are enforced by
the production V3 consumer, and the public entry is ``AgentRuntime.review_v3_json``.
Only a redacted machine summary and JUnit result are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence
from xml.etree import ElementTree

# Direct file execution makes Python add only this nested script directory to
# ``sys.path``. Add the repository root before importing the local ``tools``
# namespace so the documented command and ``python -m`` behave identically.
if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from customs_ai_agent.adapters.llm import (
    ClaudeDocumentAnalyzer,
    ClaudeSettings,
    LLMConfigurationError,
    LLMOutputValidationError,
    LLMProviderError,
)
from customs_ai_agent.adapters.retrieval import RetrievalError
from customs_ai_agent.contracts.evidence_v3 import EvidenceRuntimeReleaseIdentityV3
from customs_ai_agent.runtime import (
    AgentRuntime,
    RuntimeConfigurationError,
    RuntimeExecutionError,
    RuntimeInputError,
)
from customs_ai_agent.schemas.v3 import (
    ReviewRunResultV3,
    ServiceReviewRunRequestV3,
    canonical_json_sha256,
    review_result_v3_sha256,
)

from tools.agent_execution_inspector.evidence_v3_conformance import (
    LoopbackEvidenceV3ConformanceRetriever,
)


_EXPECTED_AGENT_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_EXPECTED_AGENT_REPOSITORY_ID = "customs-ai-agent"
_MAX_INPUT_BYTES = 32 * 1024 * 1024
_EXPECTED_DOCUMENT_TYPES = {
    "invoice",
    "packing_list",
    "bill_of_lading",
    "certificate_of_origin",
}


class _ConformanceFailure(RuntimeError):
    """A content-free diagnostic that is safe to persist in UAT artifacts."""

    def __init__(self, stage: str, code: str, error_type: str) -> None:
        super().__init__(code)
        self.stage = stage
        self.code = code
        self.error_type = error_type


def _safe_failure(exc: Exception, *, stage: str) -> _ConformanceFailure:
    """Reduce an exception to an allowlisted stage/code/type triple.

    Exception strings and provider response objects are intentionally ignored.
    The runner needs enough information to distinguish configuration, provider,
    Evidence, request, and assertion failures without persisting secrets or
    document content.
    """

    if isinstance(exc, _ConformanceFailure):
        return exc
    if stage == "identity_validation":
        return _ConformanceFailure(stage, "identity_validation", "contract_validation")
    if stage == "request_validation":
        return _ConformanceFailure(stage, "request_validation", "input_validation")
    if stage == "source_input_validation":
        return _ConformanceFailure(stage, "source_input_validation", "provenance_validation")
    if stage == "service_token_validation":
        return _ConformanceFailure(stage, "service_token_validation", "credential_validation")
    if isinstance(exc, LLMConfigurationError):
        return _ConformanceFailure("claude_configuration", "claude_configuration", "configuration")
    if isinstance(exc, LLMProviderError):
        return _ConformanceFailure("claude_provider", "claude_provider", "provider")
    if isinstance(exc, LLMOutputValidationError):
        return _ConformanceFailure(
            "claude_output_validation", "claude_output_validation", "provider_output"
        )
    if stage == "evidence_preflight" or isinstance(exc, RetrievalError):
        code = "evidence_preflight" if stage == "evidence_preflight" else "evidence_retrieval"
        return _ConformanceFailure(code, code, "evidence")
    if isinstance(exc, RuntimeInputError):
        return _ConformanceFailure("request_validation", "request_validation", "input_validation")
    if isinstance(exc, RuntimeConfigurationError):
        return _ConformanceFailure(
            "agent_runtime_configuration",
            "agent_runtime_configuration",
            "configuration",
        )
    if isinstance(exc, RuntimeExecutionError):
        return _ConformanceFailure("agent_execution", "agent_execution", "runtime")
    if stage == "runtime_initialization":
        return _ConformanceFailure(
            "runtime_initialization", "runtime_initialization", "configuration"
        )
    if stage == "result_assertion":
        return _ConformanceFailure("result_assertion", "result_assertion", "assertion")
    return _ConformanceFailure("internal", "unexpected_internal", "internal")


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    if not raw or len(raw) > _MAX_INPUT_BYTES:
        raise ValueError(f"{label}_size_invalid")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label}_duplicate_key")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda _item: (_ for _ in ()).throw(ValueError(f"{label}_nonfinite")),
        )
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError(f"{label}_invalid") from None
    if not isinstance(value, dict):
        raise ValueError(f"{label}_invalid")
    return value


def load_conformance_identity(path: Path) -> EvidenceRuntimeReleaseIdentityV3:
    payload = _strict_json_object(path.read_bytes(), label="identity")
    if payload.get("status") == "ready":
        payload = payload.get("evidence_runtime_release")
    if not isinstance(payload, dict):
        raise ValueError("identity_invalid")
    return EvidenceRuntimeReleaseIdentityV3.from_mapping(payload)


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    command = ["git", "-C", str(repository), *arguments]
    completed = subprocess.run(
        command,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("tracked_source_git_command_failed")
    return completed.stdout


def _git_ascii(repository: Path, *arguments: str) -> str:
    try:
        return _git_bytes(repository, *arguments).decode("ascii").strip()
    except UnicodeError:
        raise ValueError("tracked_source_git_ascii_invalid") from None


def _tracked_source_input_identity(
    request_path: Path,
    request_bytes: bytes,
    request: ServiceReviewRunRequestV3,
) -> dict[str, str]:
    """Bind one conformance run to an unchanged Git-tracked request source.

    No raw document text is returned. The relative path identifies the reviewed
    fixture; Git commit/blob object IDs prove its repository contents; raw-byte
    SHA-256 distinguishes checkout bytes; and the canonical request SHA-256 binds
    the parsed V3 meaning. Dirty, untracked, symlinked-out, or filtered bytes are
    rejected before credentials, Claude, or Evidence are used.
    """

    resolved = request_path.resolve(strict=True)
    root = _EXPECTED_AGENT_REPOSITORY_ROOT.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        raise ValueError("tracked_source_outside_expected_agent_repository") from None
    if _git_ascii(root, "rev-parse", "--is-inside-work-tree") != "true":
        raise ValueError("expected_agent_repository_invalid")
    if _git_bytes(root, "rev-parse", "--show-prefix").strip(b"\r\n"):
        raise ValueError("expected_agent_repository_root_mismatch")
    relative_posix = relative.as_posix()
    tracked = _git_bytes(
        root,
        "ls-files",
        "-z",
        "--error-unmatch",
        "--",
        relative_posix,
    )
    expected_tracked = relative_posix.encode("utf-8") + b"\x00"
    if tracked != expected_tracked:
        raise ValueError("tracked_source_path_mismatch")
    object_format = _git_ascii(root, "rev-parse", "--show-object-format")
    oid_length = {"sha1": 40, "sha256": 64}.get(object_format)
    if oid_length is None:
        raise ValueError("tracked_source_git_object_format_unsupported")
    commit_oid = _git_ascii(root, "rev-parse", "--verify", "HEAD^{commit}")
    blob_oid = _git_ascii(root, "rev-parse", "--verify", f"HEAD:{relative_posix}")
    oid_pattern = rf"[a-f0-9]{{{oid_length}}}"
    if not all(re.fullmatch(oid_pattern, value) for value in (commit_oid, blob_oid)):
        raise ValueError("tracked_source_object_id_invalid")
    history_roots = sorted(
        value
        for value in _git_ascii(root, "rev-list", "--max-parents=0", "HEAD").splitlines()
        if value
    )
    if not history_roots or any(
        re.fullmatch(oid_pattern, value) is None for value in history_roots
    ):
        raise ValueError("tracked_source_repository_history_invalid")
    history_identity = hashlib.sha256(
        (object_format + "\x00" + "\x00".join(history_roots)).encode("ascii")
    ).hexdigest()
    blob_bytes = _git_bytes(root, "cat-file", "blob", blob_oid)
    if request_bytes != blob_bytes:
        raise ValueError("tracked_source_bytes_do_not_match_head_blob")
    return {
        "repository_id": _EXPECTED_AGENT_REPOSITORY_ID,
        "repository_history_roots_sha256": history_identity,
        "repository_path_base": "agent-checkout-root",
        "tracked_relative_path": relative_posix,
        "git_object_format": object_format,
        "repository_head_oid": commit_oid,
        "git_blob_oid": blob_oid,
        "raw_bytes_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "canonical_request_sha256": canonical_json_sha256(request),
        "canonicalization": "customs-ai-agent/canonical-json-v1",
    }


def _read_token(path: Path) -> str:
    raw = path.read_bytes()
    if not raw or len(raw) > 4096:
        raise ValueError("service_token_invalid")
    try:
        token = raw.decode("utf-8").strip()
    except UnicodeError:
        raise ValueError("service_token_invalid") from None
    if not token or any(character.isspace() for character in token):
        raise ValueError("service_token_invalid")
    return token


def _non_release_runtime(
    *,
    base_url: str,
    identity: EvidenceRuntimeReleaseIdentityV3,
    service_token: str,
    env_file: Path,
) -> AgentRuntime:
    settings = ClaudeSettings.from_environment(env_file)
    analyzer = ClaudeDocumentAnalyzer.from_settings(settings)
    retriever = LoopbackEvidenceV3ConformanceRetriever(
        base_url,
        conformance_identity=identity,
        service_token=service_token,
        acknowledge_non_release=True,
        timeout_seconds=30,
        max_retries=0,
    )
    # Tool-only composition: the production constructor must continue to reject
    # an unreleased packaged lock. All dependencies are explicit and loopback is
    # enforced by LoopbackEvidenceV3ConformanceRetriever.
    runtime = object.__new__(AgentRuntime)
    runtime._analyzer = analyzer
    runtime._closed = False
    runtime._config = None
    runtime._retriever = None
    runtime._retriever_v3 = retriever
    # Test-only assertions remain deliberately untrusted.  The conformance
    # runner validates Agent/Data/Claude interoperability, never a Production
    # supporting-document trust path.
    runtime._assertion_verifier = None
    runtime._secret_markers = (
        settings.api_key.get_secret_value(),
        service_token,
    )
    return runtime


def _write_junit(path: Path, *, passed: bool, reason_code: str | None) -> None:
    suite = ElementTree.Element(
        "testsuite",
        name="agent-v3-http-conformance",
        tests="1",
        failures="0" if passed else "1",
    )
    case = ElementTree.SubElement(
        suite,
        "testcase",
        classname="customs_ai_agent.conformance",
        name="actual_claude_agent_v3_evidence_v3",
    )
    if not passed:
        failure = ElementTree.SubElement(
            case,
            "failure",
            message=reason_code or "conformance_failed",
        )
        failure.text = reason_code or "conformance_failed"
    ElementTree.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def _request_is_test_only_positive_control(request: ServiceReviewRunRequestV3) -> bool:
    declared = {document.declared_type for document in request.documents}
    assertions = [
        assertion
        for document in request.supporting_documents
        for assertion in document.verification_assertions
    ]
    return (
        len(request.documents) == 4
        and declared == _EXPECTED_DOCUMENT_TYPES
        and all(document.content_text is not None for document in request.documents)
        and bool(request.supporting_documents)
        and bool(assertions)
        and all(assertion.method == "test_issuer" for assertion in assertions)
    )


def _citation_metadata_complete(result: ReviewRunResultV3) -> bool:
    if not result.rag_evidence:
        return False
    for evidence in result.rag_evidence:
        required_values = (
            evidence.evidence_id,
            evidence.source_record_id,
            evidence.source_type,
            evidence.item_id,
            evidence.title,
            evidence.text,
            evidence.source_uri,
            evidence.version_hash,
            evidence.trade_type,
            evidence.eligibility,
        )
        if any(not str(value).strip() for value in required_values):
            return False
        if not evidence.retrieval_methods:
            return False
        if evidence.eligibility == "finding" and (
            evidence.valid_from is None
            or not evidence.law_id
            or not evidence.law_name
            or not evidence.matched_hs_code
            or not evidence.required_document_code
            or not evidence.required_document_name
            or not evidence.agencies
        ):
            return False
    return True


def _high_critical_findings_are_evidence_bound(result: ReviewRunResultV3) -> bool:
    evidence_by_id = {item.evidence_id: item for item in result.rag_evidence}
    for finding in result.regulatory_findings:
        if finding.severity not in {"high", "critical"}:
            continue
        if finding.source is None or not finding.evidence_ids:
            return False
        for evidence_id in finding.evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if (
                evidence is None
                or evidence.item_id != finding.item_id
                or evidence.required_document_code != finding.required_document_code
                or evidence.required_document_name != finding.required_document_name
                or evidence.eligibility != "finding"
            ):
                return False
    return True


def _narrative_citations_complete(result: ReviewRunResultV3) -> bool:
    references = {(item.kind, item.reference_id) for item in result.fact_ledger.entries}
    claims = [
        *result.narrative_report.summary,
        *result.narrative_report.document_consistency,
        *result.narrative_report.regulatory_review,
        *result.narrative_report.required_actions,
    ]
    return all(
        (not claim.material or bool(claim.citations))
        and all(
            (citation.kind, citation.reference_id) in references for citation in claim.citations
        )
        for claim in claims
    )


def _machine_code(value: object) -> str:
    """Return only a bounded machine code, never user-controlled context."""

    candidate = str(value).rsplit(":", 1)[-1]
    if re.fullmatch(r"[a-z][a-z0-9_]{0,127}", candidate):
        return candidate
    return "unclassified_code"


def _redacted_result_diagnostics(result: ReviewRunResultV3) -> dict[str, Any]:
    """Expose structural diagnostics without field values, text, or provider payloads."""

    documents = []
    extracted_hsk_field_count = 0
    hsk_field_names = {"hs_code", "hs_codes", "hsk", "hsk10"}
    for document_index, document in enumerate(result.document_results, start=1):
        field_names = sorted({field.name for field in document.fields})
        line_item_field_names = sorted(
            {field.name for item in document.line_items for field in item.fields}
        )
        extracted_hsk_field_count += sum(name in hsk_field_names for name in field_names)
        extracted_hsk_field_count += sum(
            field.name in hsk_field_names for item in document.line_items for field in item.fields
        )
        documents.append(
            {
                "document_index": document_index,
                "document_type": document.document_type,
                "extraction_status": document.extraction_status,
                "field_names": field_names,
                "line_item_count": len(document.line_items),
                "line_item_field_names": line_item_field_names,
            }
        )
    retrieval_request = result.retrieval_request
    resolved_hsk_count = sum(item.resolved_hsk10 is not None for item in result.retrieval_results)
    retrieval_query_hsk_count = (
        sum(item.hs_code is not None for item in retrieval_request.items)
        if retrieval_request is not None
        else 0
    )
    return {
        "model_error_codes": sorted({_machine_code(item.code) for item in result.model_errors}),
        "model_conflict_codes": sorted(
            {_machine_code(item.status) for item in result.model_conflicts}
        ),
        "retrieval_error_codes": sorted(
            {_machine_code(item.code) for item in result.retrieval_errors}
        ),
        "validation_issue_codes": sorted(
            {_machine_code(item.issue_code) for item in result.validation_issues}
        ),
        "review_gap_reason_codes": sorted(
            {_machine_code(item.reason_code) for item in result.review_gaps}
        ),
        "supporting_document_rejection_codes": sorted(
            {
                _machine_code(item.reason_code)
                for item in result.review_gaps
                if item.gap_type == "human_approval_required"
            }
        ),
        "completion_unmet_reason_codes": sorted(
            {_machine_code(item) for item in result.completion_eligibility.unmet_reason_codes}
        ),
        "action_reason_codes": sorted(
            {_machine_code(item.reason_code) for item in result.action_trace}
        ),
        "observation_reason_codes": sorted(
            {_machine_code(item.reason_code) for item in result.observations}
        ),
        "audit_event_codes": sorted(
            {
                f"{_machine_code(item.stage)}:{_machine_code(item.event)}"
                for item in result.audit_trail
            }
        ),
        "documents": documents,
        "retrieval_request_present": retrieval_request is not None,
        "retrieval_query_item_count": (
            len(retrieval_request.items) if retrieval_request is not None else 0
        ),
        "retrieval_query_hsk_present": retrieval_query_hsk_count > 0,
        "retrieval_query_hsk_count": retrieval_query_hsk_count,
        "resolved_hsk_present": resolved_hsk_count > 0,
        "resolved_hsk_count": resolved_hsk_count,
        "extracted_hsk_field_present": extracted_hsk_field_count > 0,
        "extracted_hsk_field_count": extracted_hsk_field_count,
    }


def _evaluate_machine_assertions(
    *,
    request: ServiceReviewRunRequestV3,
    result: ReviewRunResultV3,
    retriever: object,
    identity: EvidenceRuntimeReleaseIdentityV3,
) -> tuple[dict[str, bool], dict[str, Any]]:
    expected_types = [document.declared_type for document in request.documents]
    observed_by_id = {
        document.document_id: document.document_type for document in result.document_results
    }
    observed_types = [observed_by_id.get(document.document_id) for document in request.documents]
    readiness_get_count = int(getattr(retriever, "successful_readiness_get_count", 0))
    retrieval_post_count = int(getattr(retriever, "successful_retrieval_post_count", 0))
    model_calls = [call for call in result.model_calls if call.provider == "anthropic"]
    retrieval_results_complete = bool(result.retrieval_results) and all(
        item.coverage == "complete" and not item.errors for item in result.retrieval_results
    )
    diagnostics = _redacted_result_diagnostics(result)
    supporting_rejection_codes = set(diagnostics["supporting_document_rejection_codes"])
    checks = {
        "request_is_test_only_four_document_positive_control": (
            _request_is_test_only_positive_control(request)
        ),
        "review_result_v3_hash_valid": (result.result_sha256 == review_result_v3_sha256(result)),
        "evidence_runtime_identity_exact": (
            result.runtime_release.evidence_runtime_v3_sha256 == identity.runtime_release_v3_sha256
        ),
        "evidence_binding_header_count_exact": len(identity.binding_headers()) == 17,
        "successful_readiness_http_get_observed": readiness_get_count >= 1,
        "successful_retrieval_http_post_observed": retrieval_post_count >= 1,
        "retrieval_ok_complete": (
            result.retrieval_status == "ok"
            and result.retrieval_coverage == "complete"
            and not result.retrieval_errors
            and retrieval_results_complete
        ),
        "actual_anthropic_document_model_calls_observed": (
            len(model_calls) >= len(request.documents) > 0
        ),
        "model_errors_zero": not result.model_errors,
        "ci_pl_bl_co_classification_exact": (
            set(expected_types) == _EXPECTED_DOCUMENT_TYPES and observed_types == expected_types
        ),
        "citation_metadata_completeness": _citation_metadata_complete(result),
        "narrative_material_citation_completeness": _narrative_citations_complete(result),
        "high_critical_findings_evidence_bound": (
            _high_critical_findings_are_evidence_bound(result)
        ),
        "execution_succeeded": result.execution_status == "succeeded",
        "test_issuer_verification_rejected": (
            "trusted_verification_assertion_missing" in supporting_rejection_codes
        ),
        "test_issuer_never_auto_completes": (result.review_disposition == "human_review_required"),
    }
    observations = {
        "successful_readiness_get_count": readiness_get_count,
        "successful_retrieval_post_count": retrieval_post_count,
        "anthropic_model_call_count": len(model_calls),
        "model_error_count": len(result.model_errors),
        "expected_document_types": expected_types,
        "observed_document_types": observed_types,
        "rag_evidence_count": len(result.rag_evidence),
        "high_critical_finding_count": sum(
            finding.severity in {"high", "critical"} for finding in result.regulatory_findings
        ),
        **diagnostics,
    }
    return checks, observations


def run_conformance(
    *,
    base_url: str,
    identity_path: Path,
    request_path: Path,
    token_path: Path,
    env_file: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    junit_path = output_dir / "junit.xml"
    runtime: AgentRuntime | None = None
    report: dict[str, Any]
    passed = False
    reason_code: str | None = None
    source_input_identity: dict[str, str] | None = None
    stage = "identity_validation"
    try:
        identity = load_conformance_identity(identity_path)
        stage = "request_validation"
        request_bytes = request_path.read_bytes()
        request_payload = _strict_json_object(request_bytes, label="request")
        request = ServiceReviewRunRequestV3.model_validate(request_payload)
        stage = "source_input_validation"
        source_input_identity = _tracked_source_input_identity(
            request_path,
            request_bytes,
            request,
        )
        if request.review_as_of != identity.base_runtime_release.supported_review_as_of:
            raise _ConformanceFailure(
                "evidence_preflight",
                "evidence_review_date_mismatch",
                "contract_validation",
            )
        stage = "service_token_validation"
        service_token = _read_token(token_path)
        stage = "runtime_initialization"
        runtime = _non_release_runtime(
            base_url=base_url,
            identity=identity,
            service_token=service_token,
            env_file=env_file,
        )
        # Run the date-bound readiness handshake explicitly. AgentRuntime will
        # repeat it at the service boundary; this first call makes a failure
        # diagnosable as Evidence preflight before any Claude request is sent.
        stage = "evidence_preflight"
        runtime._retriever_v3.preflight(request.review_as_of)
        stage = "agent_execution"
        result = runtime.review_v3_json(request_bytes)
        stage = "result_assertion"
        checks, observations = _evaluate_machine_assertions(
            request=request,
            result=result,
            retriever=runtime._retriever_v3,
            identity=identity,
        )
        checks["tracked_source_canonical_hash_matches_result"] = (
            source_input_identity["canonical_request_sha256"] == result.input_manifest_sha256
        )
        passed = all(checks.values())
        reason_code = None if passed else "result_assertion_failed"
        report = {
            "schema_version": "agent-v3-http-conformance-report/v1",
            "status": "PASS" if passed else "FAIL",
            "non_release": True,
            "provider_scope": "TEST_ONLY_LOOPBACK_CONFORMANCE_PROVIDER",
            "production_release_evidence": False,
            "expected_disposition": "human_review_required",
            "entrypoint": "AgentRuntime.review_v3_json",
            "checks": checks,
            "observations": observations,
            "execution_status": result.execution_status,
            "assessment": result.assessment,
            "review_disposition": result.review_disposition,
            "assurance_level": result.assurance_level,
            "retrieval_status": result.retrieval_status,
            "retrieval_coverage": result.retrieval_coverage,
            "finding_count": len(result.regulatory_findings),
            "validation_issue_count": len(result.validation_issues),
            "action_count": len(result.action_trace),
            "result_sha256": result.result_sha256,
            "input_manifest_sha256": result.input_manifest_sha256,
            "source_input": source_input_identity,
            "runtime_release": result.runtime_release.model_dump(mode="json", warnings=False),
        }
        if not passed:
            report.update(
                {
                    "failure_stage": "result_assertion",
                    "error_code": reason_code,
                    "error_type": "assertion",
                }
            )
    except Exception as exc:
        failure = _safe_failure(exc, stage=stage)
        reason_code = failure.code
        report = {
            "schema_version": "agent-v3-http-conformance-report/v1",
            "status": "FAIL",
            "non_release": True,
            "provider_scope": "TEST_ONLY_LOOPBACK_CONFORMANCE_PROVIDER",
            "production_release_evidence": False,
            "entrypoint": "AgentRuntime.review_v3_json",
            "failure_stage": failure.stage,
            "error_code": failure.code,
            "error_type": failure.error_type,
        }
        if source_input_identity is not None:
            report["source_input"] = source_input_identity
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                passed = False
                reason_code = "runtime_close_failed"
                report = {
                    "schema_version": "agent-v3-http-conformance-report/v1",
                    "status": "FAIL",
                    "non_release": True,
                    "provider_scope": "TEST_ONLY_LOOPBACK_CONFORMANCE_PROVIDER",
                    "production_release_evidence": False,
                    "entrypoint": "AgentRuntime.review_v3_json",
                    "failure_stage": "runtime_close",
                    "error_code": reason_code,
                    "error_type": "runtime",
                }
                if source_input_identity is not None:
                    report["source_input"] = source_input_identity
    encoded = json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    report_path.write_text(encoded + "\n", encoding="utf-8")
    _write_junit(junit_path, passed=passed, reason_code=reason_code)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--acknowledge-non-release", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.acknowledge_non_release:
        raise SystemExit("--acknowledge-non-release is required")
    report = run_conformance(
        base_url=args.base_url,
        identity_path=args.identity,
        request_path=args.request,
        token_path=args.token_file,
        env_file=args.env_file,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
