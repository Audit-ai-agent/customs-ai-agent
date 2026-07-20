"""Opt-in public AgentRuntime -> Claude -> Evidence HTTP evaluation.

The evaluator executes one fixed, synthetic, text-only scenario through the
service-facing ``AgentRuntime`` facade. Reports are intentionally value-free:
they contain only bounded model/prompt/schema identities, counts, statuses,
error codes, and cryptographic hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
from collections import Counter
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator

from dotenv import dotenv_values
from pydantic import SecretStr

import customs_ai_agent.adapters.retrieval as retrieval_module
import customs_ai_agent.runtime as runtime_module
from customs_ai_agent.adapters.llm import (
    CLAUDE_CANDIDATE_SCHEMA_VERSION,
    DEFAULT_CLAUDE_MODEL,
    LLMError,
)
from customs_ai_agent.adapters.retrieval import (
    RetrievalError,
    fetch_evidence_runtime_readiness,
    validate_evidence_destination,
)
from customs_ai_agent.contracts import (
    ContractSchemaError,
    schema_sha256,
    verify_packaged_schemas,
)
from customs_ai_agent.contracts.evidence_runtime_release import (
    EvidenceRuntimeReleaseIdentity,
)
from customs_ai_agent.fixtures.scenarios import get_scenario
from customs_ai_agent.prompts import (
    DOCUMENT_ANALYSIS_PROMPT_ID,
    DOCUMENT_ANALYSIS_PROMPT_VERSION,
    load_document_analysis_system_prompt,
)
from customs_ai_agent.runtime import AgentRuntime, AgentRuntimeError, RuntimeConfig
from customs_ai_agent.schemas.llm import ClaudeDocumentCandidate
from customs_ai_agent.schemas.models import (
    ReviewRunResultV2,
    ServiceReviewRunRequestV2,
)
from tools.validate_v2_conformance_identity import (
    ConformanceIdentityValidationError,
    validate_conformance_identity,
)


DEFAULT_BASE_URL = "http://127.0.0.1:8010"
SCENARIO_ID = "evidence_live_exact_hsk10"
EXPECTED_DOCUMENT_COUNT = 4
REPORT_SCHEMA_VERSION = "runtime-live-evaluation-report/v2"
LIVE_PROVIDER_SCOPE = "AGENT_TO_EVIDENCE_LIVE_RELEASE_EVALUATION"
PROVIDER_CONFORMANCE_SCOPE = "PROVIDER_CONFORMANCE_NOT_RAG_QUALITY"
RESULT_SCHEMA_VERSION = "2.0"
TEXT_OUTPUT_SCHEMA_ID = "customs_claude_document_candidate"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SAFE_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_CONFORMANCE_OVERLAY_LOCK = threading.RLock()
_EVIDENCE_ENV_KEYS = frozenset(
    {
        "EVIDENCE_SERVICE_ALLOWED_ORIGINS",
        "EVIDENCE_SERVICE_TOKEN",
        "EVIDENCE_SERVICE_TOKEN_HEADER",
    }
)


class RuntimeLiveEvaluationError(RuntimeError):
    """Content-free evaluator error that is safe to reduce to a code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@contextmanager
def _provider_conformance_runtime_identity(
    identity: EvidenceRuntimeReleaseIdentity,
) -> Iterator[None]:
    """Temporarily bind evaluator-local runtime loaders to one validated identity."""

    with _CONFORMANCE_OVERLAY_LOCK:
        original_runtime_loader = runtime_module.load_released_evidence_runtime_identity
        original_retrieval_loader = retrieval_module.load_released_evidence_runtime_identity

        def load_validated_identity() -> EvidenceRuntimeReleaseIdentity:
            return identity

        runtime_module.load_released_evidence_runtime_identity = load_validated_identity
        retrieval_module.load_released_evidence_runtime_identity = load_validated_identity
        try:
            yield
        finally:
            retrieval_module.load_released_evidence_runtime_identity = original_retrieval_loader
            runtime_module.load_released_evidence_runtime_identity = original_runtime_loader


def _safe_code(value: object) -> str:
    candidate = str(value or "")
    return candidate if _SAFE_CODE.fullmatch(candidate) else "UNSAFE_CODE_REDACTED"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text_output_schema_sha256() -> str:
    return _canonical_sha256(ClaudeDocumentCandidate.model_json_schema())


def _text_prompt_sha256() -> str:
    return hashlib.sha256(load_document_analysis_system_prompt().encode("utf-8")).hexdigest()


def _contract_preflight() -> tuple[dict[str, str | None], list[str]]:
    failures: list[str] = []
    request_hash: str | None = None
    result_hash: str | None = None
    try:
        verify_packaged_schemas()
        request_hash = schema_sha256("backend", "v2", "review-request")
        result_hash = schema_sha256("backend", "v2", "review-result")
    except ContractSchemaError:
        failures.append("packaged_contract_drift")
    return (
        {
            "request_schema_version": "v2",
            "request_schema_sha256": request_hash,
            "result_schema_version": "v2",
            "result_schema_sha256": result_hash,
        },
        failures,
    )


def _explicit_env_values(env_file: Path) -> dict[str, str]:
    try:
        path = env_file.expanduser()
        if not path.is_file():
            raise OSError
        raw = dotenv_values(dotenv_path=path, verbose=False)
    except (OSError, UnicodeError, ValueError):
        raise RuntimeLiveEvaluationError("env_file_unreadable") from None
    return {name: value for name in _EVIDENCE_ENV_KEYS if isinstance((value := raw.get(name)), str)}


def _configured_value(
    name: str,
    file_values: dict[str, str],
    default: str | None = None,
) -> str | None:
    if name in os.environ:
        return os.environ[name]
    return file_values.get(name, default)


def _evidence_options(
    *,
    env_file: Path,
    base_url: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    file_values = _explicit_env_values(env_file)
    configured_origins = _configured_value(
        "EVIDENCE_SERVICE_ALLOWED_ORIGINS",
        file_values,
    )
    allowed_origins = frozenset(
        origin.strip() for origin in (configured_origins or "").split(",") if origin.strip()
    )
    try:
        normalized_base_url = validate_evidence_destination(
            base_url,
            allowed_origins,
        )
    except (TypeError, ValueError):
        raise RuntimeLiveEvaluationError("unsafe_evidence_destination") from None

    token = _configured_value("EVIDENCE_SERVICE_TOKEN", file_values)
    token_header = _configured_value(
        "EVIDENCE_SERVICE_TOKEN_HEADER",
        file_values,
        "Authorization",
    )
    if not token_header or token_header.strip().casefold() != "authorization":
        raise RuntimeLiveEvaluationError("evidence_token_header_invalid")

    readiness_options: dict[str, Any] = {
        "service_token": token if token else None,
        "allowed_origins": allowed_origins,
    }
    runtime_options: dict[str, Any] = {
        "evidence_service_token": SecretStr(token) if token else None,
        "evidence_allowed_origins": allowed_origins,
    }
    return normalized_base_url, readiness_options, runtime_options


def _base_report(
    *,
    contracts: dict[str, str | None],
    status: str,
    readiness_status: str,
    network_status: str,
    failures: Iterable[str],
    provider_conformance: bool,
    release_ready: bool = False,
    agent_execution_passed: bool = False,
) -> dict[str, Any]:
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "scope": (PROVIDER_CONFORMANCE_SCOPE if provider_conformance else LIVE_PROVIDER_SCOPE),
        "status": status,
        "readiness_status": readiness_status,
        "network_status": network_status,
        "release_ready": release_ready,
        "implementation_candidate_ready": False,
        "agent_execution_passed": agent_execution_passed,
        "scenario_sha256": hashlib.sha256(SCENARIO_ID.encode("utf-8")).hexdigest(),
        "contracts": contracts,
        "models": [],
        "prompts": [],
        "output_schemas": [],
        "counts": {
            "document_count": 0,
            "document_completed_count": 0,
            "model_call_count": 0,
            "model_error_count": 0,
            "retrieval_result_count": 0,
            "retrieval_error_count": 0,
            "evidence_count": 0,
            "validation_issue_count": 0,
            "regulatory_finding_count": 0,
        },
        "result_status": None,
        "retrieval_status": None,
        "retrieval_coverage_status": None,
        "requires_human_review": None,
        "hashes": {
            "result_sha256": None,
            "knowledge_snapshot_sha256": None,
        },
        "errors": [{"code": _safe_code(code)} for code in sorted(set(failures))],
        "redaction_status": {
            "document_text": "OMITTED",
            "extracted_values": "OMITTED",
            "source_uri": "OMITTED",
            "api_key": "OMITTED",
            "service_token": "OMITTED",
            "env_file_path": "OMITTED",
            "raw_response": "OMITTED",
        },
    }


def _group_model_metadata(result: ReviewRunResultV2) -> list[dict[str, Any]]:
    counts = Counter((call.provider, call.model_id) for call in result.model_calls)
    return [
        {
            "provider": provider,
            "model_id": model_id,
            "call_count": count,
        }
        for (provider, model_id), count in sorted(counts.items())
    ]


def _group_prompt_metadata(result: ReviewRunResultV2) -> list[dict[str, Any]]:
    counts = Counter(
        (call.prompt_id, call.prompt_version, call.prompt_sha256) for call in result.model_calls
    )
    return [
        {
            "prompt_id": prompt_id,
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_sha256,
            "call_count": count,
        }
        for (prompt_id, prompt_version, prompt_sha256), count in sorted(
            counts.items(),
            key=lambda item: tuple("" if value is None else str(value) for value in item[0]),
        )
    ]


def _group_output_schema_metadata(
    result: ReviewRunResultV2,
) -> list[dict[str, Any]]:
    counts = Counter(
        (
            call.output_schema_id,
            call.output_schema_version,
            call.output_schema_sha256,
        )
        for call in result.model_calls
    )
    return [
        {
            "schema_id": schema_id,
            "schema_version": schema_version,
            "schema_sha256": schema_hash,
            "call_count": count,
        }
        for (schema_id, schema_version, schema_hash), count in sorted(
            counts.items(),
            key=lambda item: tuple("" if value is None else str(value) for value in item[0]),
        )
    ]


def _result_summary(result: ReviewRunResultV2) -> dict[str, Any]:
    payload = result.model_dump(mode="json")
    snapshot_id = result.knowledge_snapshot_id
    return {
        "models": _group_model_metadata(result),
        "prompts": _group_prompt_metadata(result),
        "output_schemas": _group_output_schema_metadata(result),
        "counts": {
            "document_count": len(result.document_results),
            "document_completed_count": sum(
                document.extraction_status == "completed" for document in result.document_results
            ),
            "model_call_count": len(result.model_calls),
            "model_error_count": len(result.model_errors),
            "retrieval_result_count": len(result.retrieval_results),
            "retrieval_error_count": len(result.retrieval_errors),
            "evidence_count": len(result.rag_evidence),
            "validation_issue_count": len(result.validation_issues),
            "regulatory_finding_count": len(result.regulatory_findings),
        },
        "result_status": result.status,
        "retrieval_status": result.retrieval_status,
        "retrieval_coverage_status": result.retrieval_coverage,
        "requires_human_review": result.requires_human_review,
        "hashes": {
            "result_sha256": _canonical_sha256(payload),
            "knowledge_snapshot_sha256": (
                hashlib.sha256(snapshot_id.encode("utf-8")).hexdigest() if snapshot_id else None
            ),
        },
        "model_error_codes": [_safe_code(error.code) for error in result.model_errors],
        "retrieval_error_codes": [_safe_code(error.code) for error in result.retrieval_errors],
    }


def _agent_execution_failures(result: ReviewRunResultV2) -> list[str]:
    failures: list[str] = []
    expected_prompt_hash = _text_prompt_sha256()
    expected_output_schema_hash = _text_output_schema_sha256()
    if result.schema_version != RESULT_SCHEMA_VERSION:
        failures.append("result_schema_version_invalid")
    if len(result.document_results) != EXPECTED_DOCUMENT_COUNT:
        failures.append("document_count_invalid")
    if any(document.extraction_status != "completed" for document in result.document_results):
        failures.append("document_extraction_incomplete")
    if len(result.model_calls) != EXPECTED_DOCUMENT_COUNT:
        failures.append("model_call_count_invalid")
    if result.model_errors:
        failures.append("model_errors_present")
    for call in result.model_calls:
        if call.provider != "anthropic" or call.model_id != DEFAULT_CLAUDE_MODEL:
            failures.append("model_identity_invalid")
        if (
            call.prompt_id != DOCUMENT_ANALYSIS_PROMPT_ID
            or call.prompt_version != DOCUMENT_ANALYSIS_PROMPT_VERSION
            or call.prompt_sha256 != expected_prompt_hash
        ):
            failures.append("prompt_identity_invalid")
        if (
            call.output_schema_id != TEXT_OUTPUT_SCHEMA_ID
            or call.output_schema_version != CLAUDE_CANDIDATE_SCHEMA_VERSION
            or call.output_schema_sha256 != expected_output_schema_hash
        ):
            failures.append("model_output_schema_invalid")
        if call.input_modality != "text":
            failures.append("unexpected_model_input_modality")

    incomplete_retrieval = bool(
        result.retrieval_status != "ok"
        or result.retrieval_coverage != "complete"
        or not result.knowledge_snapshot_id
        or result.retrieval_errors
        or not result.retrieval_results
        or any(
            item.coverage != "complete"
            or item.errors
            or item.hs_resolution not in {"exact", "unique_descendant"}
            or not item.resolved_hsk10
            for item in result.retrieval_results
        )
    )
    if incomplete_retrieval and (
        result.status not in {"needs_review", "blocked"} or not result.requires_human_review
    ):
        failures.append("incomplete_retrieval_not_fail_closed")
    return sorted(set(failures))


def _ready_contract_failures(result: ReviewRunResultV2) -> list[str]:
    failures: list[str] = []
    if result.retrieval_status != "ok":
        failures.append("ready_retrieval_status_not_ok")
    if result.retrieval_coverage != "complete":
        failures.append("ready_retrieval_coverage_incomplete")
    if not result.knowledge_snapshot_id:
        failures.append("ready_snapshot_missing")
    if result.retrieval_errors:
        failures.append("ready_retrieval_errors_present")
    if not result.retrieval_results:
        failures.append("ready_retrieval_results_missing")
    if any(
        item.coverage != "complete"
        or item.errors
        or item.hs_resolution not in {"exact", "unique_descendant"}
        or not item.resolved_hsk10
        for item in result.retrieval_results
    ):
        failures.append("ready_retrieval_contract_incomplete")
    return sorted(set(failures))


def _readiness_failure_code(exc: Exception) -> str:
    if isinstance(exc, RetrievalError) and "HTTP status 503" in str(exc):
        return "evidence_readiness_unready_503"
    return "evidence_readiness_failed"


def _runtime_failure_code(exc: Exception) -> str:
    if isinstance(exc, (AgentRuntimeError, LLMError)):
        return _safe_code(exc.code)
    return "agent_runtime_execution_failed"


def _explicit_service_request(*, review_as_of: date) -> ServiceReviewRunRequestV2:
    """Expand the fixed evaluation fixture before crossing the service boundary.

    ``AgentRuntime`` deliberately rejects fixture identifiers because a service
    caller must bind every review to an explicit case and document set.  This
    evaluator may use a package-owned synthetic fixture, but it must convert the
    fixture to the same explicit request shape a Backend caller would send.
    """

    try:
        scenario = get_scenario(SCENARIO_ID)
        return ServiceReviewRunRequestV2(
            case_id=str(scenario["case_id"]),
            documents=scenario["documents"],
            llm_mode="llm",
            review_run_id="runtime-live-eval-v1",
            review_as_of=review_as_of,
        )
    except (KeyError, TypeError, ValueError):
        raise RuntimeLiveEvaluationError("service_fixture_invalid") from None


def _evaluate_live_network(
    *,
    contracts: dict[str, str | None],
    normalized_url: str,
    readiness_options: dict[str, Any],
    runtime_options: dict[str, Any],
    env_file: Path,
    provider_conformance: bool,
    allow_unready_diagnostic: bool,
    diagnostic_review_as_of: date | None,
) -> tuple[int, dict[str, Any]]:
    readiness_status = "READY"
    try:
        runtime_identity = fetch_evidence_runtime_readiness(
            normalized_url,
            **readiness_options,
        )
        review_as_of = runtime_identity.supported_review_as_of
    except Exception as exc:
        readiness_failure = _readiness_failure_code(exc)
        if (
            readiness_failure != "evidence_readiness_unready_503"
            or not allow_unready_diagnostic
            or diagnostic_review_as_of is None
        ):
            return 1, _base_report(
                contracts=contracts,
                status="FAIL",
                readiness_status=(
                    "UNREADY_503"
                    if readiness_failure == "evidence_readiness_unready_503"
                    else "FAILED"
                ),
                network_status="LIVE_READINESS_ONLY",
                failures=[readiness_failure],
                provider_conformance=provider_conformance,
            )
        review_as_of = diagnostic_review_as_of
        readiness_status = "UNREADY_503_DIAGNOSTIC"

    try:
        config = RuntimeConfig.from_environment(
            evidence_base_url=normalized_url,
            env_file=env_file,
            **runtime_options,
        )
        with AgentRuntime(config) as runtime:
            result = runtime.review(_explicit_service_request(review_as_of=review_as_of))
        if not isinstance(result, ReviewRunResultV2):
            raise RuntimeLiveEvaluationError("runtime_result_contract_invalid")
    except Exception as exc:
        failure_code = (
            exc.code if isinstance(exc, RuntimeLiveEvaluationError) else _runtime_failure_code(exc)
        )
        return 1, _base_report(
            contracts=contracts,
            status="FAIL",
            readiness_status=readiness_status,
            network_status="LIVE_AGENT_ATTEMPTED",
            failures=[failure_code],
            provider_conformance=provider_conformance,
        )

    summary = _result_summary(result)
    execution_failures = _agent_execution_failures(result)
    agent_execution_passed = not execution_failures
    report = _base_report(
        contracts=contracts,
        status="FAIL",
        readiness_status=readiness_status,
        network_status="LIVE_AGENT_EXECUTED",
        failures=[],
        provider_conformance=provider_conformance,
        agent_execution_passed=agent_execution_passed,
    )
    report.update(
        {
            key: value
            for key, value in summary.items()
            if key not in {"model_error_codes", "retrieval_error_codes"}
        }
    )
    observed_error_codes = [
        *summary["model_error_codes"],
        *summary["retrieval_error_codes"],
    ]

    if readiness_status == "UNREADY_503_DIAGNOSTIC":
        failures = [
            *execution_failures,
            *observed_error_codes,
            "evidence_readiness_incomplete",
        ]
        report["status"] = "BLOCKED_DATA_READINESS" if agent_execution_passed else "FAIL"
        report["release_ready"] = False
        report["errors"] = [{"code": _safe_code(code)} for code in sorted(set(failures))]
        return 1, report

    contract_failures = _ready_contract_failures(result)
    failures = [
        *execution_failures,
        *contract_failures,
        *observed_error_codes,
    ]
    contract_ready = not failures
    if contract_ready and not provider_conformance:
        # This evaluator can prove Agent execution and the pinned Evidence HTTP
        # contract, but it does not consume a signed/hashed Data RAG-quality
        # attestation. A caller-controlled omission of --provider-conformance
        # must therefore never promote the same test provider to release-ready.
        failures.append("rag_quality_attestation_missing")
        report["status"] = "BLOCKED_RAG_RELEASE_ATTESTATION"
    else:
        report["status"] = "PASS" if contract_ready else "FAIL"
    report["release_ready"] = False
    report["implementation_candidate_ready"] = contract_ready and provider_conformance
    report["errors"] = [{"code": _safe_code(code)} for code in sorted(set(failures))]
    return (0 if contract_ready and provider_conformance else 1), report


def evaluate(
    *,
    env_file: Path,
    base_url: str = DEFAULT_BASE_URL,
    live: bool = False,
    provider_conformance: bool = False,
    provider_runtime_lock: Path | None = None,
    runtime_identity: Path | None = None,
    conformance_manifest_output: Path | None = None,
    conformance_base_commit: str | None = None,
    allow_unready_diagnostic: bool = False,
    diagnostic_review_as_of: date | None = None,
) -> tuple[int, dict[str, Any]]:
    contracts, failures = _contract_preflight()
    if failures:
        return 1, _base_report(
            contracts=contracts,
            status="FAIL",
            readiness_status="NOT_ATTEMPTED",
            network_status="DISABLED_CONTRACT_PREFLIGHT_FAILED",
            failures=failures,
            provider_conformance=provider_conformance,
        )
    if not live:
        return 1, _base_report(
            contracts=contracts,
            status="FAIL",
            readiness_status="NOT_ATTEMPTED",
            network_status="DISABLED_LIVE_FLAG_REQUIRED",
            failures=["live_flag_required"],
            provider_conformance=provider_conformance,
        )
    if diagnostic_review_as_of is not None and not allow_unready_diagnostic:
        return 1, _base_report(
            contracts=contracts,
            status="FAIL",
            readiness_status="NOT_ATTEMPTED",
            network_status="DISABLED_DIAGNOSTIC_PREFLIGHT_FAILED",
            failures=["diagnostic_date_without_override"],
            provider_conformance=provider_conformance,
        )

    conformance_inputs = (
        provider_runtime_lock,
        runtime_identity,
        conformance_manifest_output,
        conformance_base_commit,
    )
    if provider_conformance and any(value is None for value in conformance_inputs):
        return 1, _base_report(
            contracts=contracts,
            status="FAIL",
            readiness_status="NOT_ATTEMPTED",
            network_status="DISABLED_CONFORMANCE_PREFLIGHT_FAILED",
            failures=["provider_conformance_artifacts_required"],
            provider_conformance=True,
        )
    if not provider_conformance and any(value is not None for value in conformance_inputs):
        return 1, _base_report(
            contracts=contracts,
            status="FAIL",
            readiness_status="NOT_ATTEMPTED",
            network_status="DISABLED_CONFORMANCE_PREFLIGHT_FAILED",
            failures=["provider_conformance_flag_required"],
            provider_conformance=False,
        )

    validated_identity: EvidenceRuntimeReleaseIdentity | None = None
    if provider_conformance:
        if (
            provider_runtime_lock is None
            or runtime_identity is None
            or conformance_manifest_output is None
            or conformance_base_commit is None
        ):
            return 1, _base_report(
                contracts=contracts,
                status="FAIL",
                readiness_status="NOT_ATTEMPTED",
                network_status="DISABLED_CONFORMANCE_PREFLIGHT_FAILED",
                failures=["provider_conformance_artifacts_required"],
                provider_conformance=True,
            )
        try:
            validation = validate_conformance_identity(
                repository=REPOSITORY_ROOT,
                provider_runtime_lock_path=provider_runtime_lock,
                runtime_identity_path=runtime_identity,
                manifest_output=conformance_manifest_output,
                expected_base_commit=conformance_base_commit,
                test_only_conformance=True,
            )
        except ConformanceIdentityValidationError as exc:
            return 1, _base_report(
                contracts=contracts,
                status="FAIL",
                readiness_status="NOT_ATTEMPTED",
                network_status="DISABLED_CONFORMANCE_PREFLIGHT_FAILED",
                failures=[exc.code],
                provider_conformance=True,
            )
        validated_identity = validation.identity

    try:
        normalized_url, readiness_options, runtime_options = _evidence_options(
            env_file=env_file,
            base_url=base_url,
        )
    except RuntimeLiveEvaluationError as exc:
        return 1, _base_report(
            contracts=contracts,
            status="FAIL",
            readiness_status="NOT_ATTEMPTED",
            network_status="DISABLED_CONFIGURATION_PREFLIGHT_FAILED",
            failures=[exc.code],
            provider_conformance=provider_conformance,
        )

    def run_network() -> tuple[int, dict[str, Any]]:
        return _evaluate_live_network(
            contracts=contracts,
            normalized_url=normalized_url,
            readiness_options=readiness_options,
            runtime_options=runtime_options,
            env_file=env_file,
            provider_conformance=provider_conformance,
            allow_unready_diagnostic=allow_unready_diagnostic,
            diagnostic_review_as_of=diagnostic_review_as_of,
        )

    if validated_identity is None:
        return run_network()
    with _provider_conformance_runtime_identity(validated_identity):
        return run_network()


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed synthetic text scenario through public AgentRuntime, "
            "Claude, and the pinned Evidence HTTP service."
        )
    )
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Required to permit Evidence and Anthropic API calls.",
    )
    parser.add_argument(
        "--provider-conformance",
        action="store_true",
        help=(
            "Classify a successful run as provider transport/contract conformance only, "
            "never as RAG quality or release readiness. Without this flag, a ready "
            "provider remains blocked until a separate RAG-quality attestation exists. "
            "Requires all --conformance-* inputs."
        ),
    )
    parser.add_argument(
        "--provider-runtime-lock",
        type=Path,
        help="External Data provider lock used only with --provider-conformance.",
    )
    parser.add_argument(
        "--runtime-identity",
        type=Path,
        help="External sealed runtime identity used only with --provider-conformance.",
    )
    parser.add_argument(
        "--conformance-manifest-output",
        type=Path,
        help="New external path for the non-deployable conformance validation manifest.",
    )
    parser.add_argument(
        "--conformance-base-commit",
        help="Exact clean Agent commit validated for evaluator-only conformance.",
    )
    parser.add_argument(
        "--allow-unready-diagnostic",
        action="store_true",
        help=(
            "Permit one non-release Agent execution only when readiness returns "
            "HTTP 503. Requires --diagnostic-review-as-of."
        ),
    )
    parser.add_argument(
        "--diagnostic-review-as-of",
        type=_parse_date,
        help="Explicit YYYY-MM-DD used only for the HTTP 503 diagnostic path.",
    )
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exit_code, report = evaluate(
        env_file=args.env_file,
        base_url=args.base_url,
        live=args.live,
        provider_conformance=args.provider_conformance,
        provider_runtime_lock=args.provider_runtime_lock,
        runtime_identity=args.runtime_identity,
        conformance_manifest_output=args.conformance_manifest_output,
        conformance_base_commit=args.conformance_base_commit,
        allow_unready_diagnostic=args.allow_unready_diagnostic,
        diagnostic_review_as_of=args.diagnostic_review_as_of,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
