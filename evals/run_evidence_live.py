"""Opt-in Agent -> pinned Evidence HTTP contract evaluation.

The evaluator runs exactly one synthetic ``happy_path`` workflow. It does not
import the Data Platform package or connect to its databases. The emitted
report contains only bounded status/count/hash metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

from dotenv import dotenv_values

from customs_ai_agent.adapters import retrieval as retrieval_adapter
from customs_ai_agent.adapters.retrieval import (
    HttpRegulationRetriever,
    fetch_supported_review_as_of,
)
from customs_ai_agent.runner import run_review
from customs_ai_agent.schemas.models import ReviewRunRequest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPOSITORY_ROOT / "contracts" / "evidence" / "v1.lock.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8010"
EXPECTED_EVIDENCE_SCHEMA_SHA256 = "531b59ba4e86f78cef20a4fca052996efd8b88c48853ac12113ca76e69cdd69a"
EXPECTED_PROVIDER_RELEASE = "evidence-api-v1.2.0"
EXPECTED_PINNED_PROVIDER_REVISION = "0f6fbe031ddf531b44bc8f338bd2ec1d39e47039"
EXPECTED_PROVIDER_BUILD_SHA256 = "a2e17f85f189a964f48a3650e0fd76a5203e718e05427da539eba3beeacf2d1c"
REPORT_SCHEMA_VERSION = "evidence-live-evaluation-report/v2"
REPORT_SCOPE = "SYNTHETIC_AGENT_TO_EVIDENCE_HTTP_NOT_RELEASE"
ALLOWED_SCENARIOS = frozenset({"happy_path", "evidence_live_exact_hsk10"})
_HTTP_MAX_RETRIES = 1
_READY_READINESS_GET_RANGE = range(2, (2 * (_HTTP_MAX_RETRIES + 1)) + 1)
_READY_RETRIEVAL_POST_RANGE = range(1, (_HTTP_MAX_RETRIES + 1) + 1)
_SAFE_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SAFE_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_SAFE_HEADER_NAME = re.compile(r"^[A-Za-z0-9-]{1,64}$")
_EVIDENCE_ENV_KEYS = frozenset(
    {
        "EVIDENCE_SERVICE_TOKEN",
        "EVIDENCE_SERVICE_TOKEN_HEADER",
    }
)


class EvidenceLiveEvaluationError(RuntimeError):
    """Content-free evaluator error safe to reduce to a bounded code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass
class _EvidenceHttpCounts:
    """Bounded HTTP-attempt counts observed at the Evidence adapter opener."""

    readiness_get_count: int = 0
    retrieval_post_count: int = 0
    other_http_count: int = 0
    observer_error_code: str | None = None


_HTTP_OBSERVER_STATE_LOCK = threading.Lock()
_HTTP_OBSERVER_ACTIVE = False


@contextmanager
def _observe_evidence_http_calls() -> Iterator[_EvidenceHttpCounts]:
    """Count actual adapter opener attempts for this one-shot evaluator.

    The evaluator is a single-workflow CLI. The wrapper is restored in
    ``finally`` and records attempts before delegating, so failed and retried
    requests remain observable without weakening the retriever's readiness
    checks or changing its network behavior.
    """

    global _HTTP_OBSERVER_ACTIVE

    counts = _EvidenceHttpCounts()
    with _HTTP_OBSERVER_STATE_LOCK:
        if _HTTP_OBSERVER_ACTIVE:
            counts.observer_error_code = "evidence_http_observer_busy"
        else:
            _HTTP_OBSERVER_ACTIVE = True
    if counts.observer_error_code is not None:
        # Never run an unobserved nested/concurrent live evaluation. The
        # caller inspects this code before constructing a retriever or making
        # any HTTP request.
        yield counts
        return

    original_urlopen = retrieval_adapter.urlopen

    def observed_urlopen(request: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            method = request.get_method()
            path = urlsplit(str(request.full_url)).path
        except (AttributeError, TypeError, ValueError):
            counts.other_http_count += 1
        else:
            if method == "GET" and path == "/internal/rag/v1/health/ready":
                counts.readiness_get_count += 1
            elif (
                method == "POST" and path == retrieval_adapter.HttpRegulationRetriever.endpoint_path
            ):
                counts.retrieval_post_count += 1
            else:
                counts.other_http_count += 1
        return original_urlopen(request, *args, **kwargs)

    retrieval_adapter.urlopen = observed_urlopen
    try:
        yield counts
    finally:
        retrieval_adapter.urlopen = original_urlopen
        with _HTTP_OBSERVER_STATE_LOCK:
            _HTTP_OBSERVER_ACTIVE = False


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_code(value: object) -> str:
    candidate = str(value or "")
    return candidate if _SAFE_CODE.fullmatch(candidate) else "UNSAFE_CODE_REDACTED"


def _safe_sha256(value: object) -> str | None:
    candidate = str(value or "")
    return candidate if _SAFE_SHA256.fullmatch(candidate) else None


def _contract_preflight(
    *,
    lock_path: Path = LOCK_PATH,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    lock: dict[str, Any] = {}
    consumer_sha256: str | None = None
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        relative_snapshot = Path(str(lock["consumer_snapshot_path"]))
        if relative_snapshot.is_absolute() or ".." in relative_snapshot.parts:
            raise ValueError
        snapshot_path = (REPOSITORY_ROOT / relative_snapshot).resolve(strict=True)
        snapshot_path.relative_to(REPOSITORY_ROOT)
        snapshot_bytes = snapshot_path.read_bytes()
        json.loads(snapshot_bytes)
        consumer_sha256 = _sha256_bytes(snapshot_bytes)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        failures.append("consumer_contract_unreadable")

    locked_sha256 = lock.get("sha256")
    if locked_sha256 != EXPECTED_EVIDENCE_SCHEMA_SHA256:
        failures.append("lock_schema_sha256_unexpected")
    if consumer_sha256 != locked_sha256:
        failures.append("consumer_contract_drift")
    if lock.get("provider_release") != EXPECTED_PROVIDER_RELEASE:
        failures.append("provider_release_pin_unexpected")
    if lock.get("provider_revision") != EXPECTED_PINNED_PROVIDER_REVISION:
        failures.append("provider_release_revision_pin_unexpected")
    if lock.get("provider_build_sha256") != EXPECTED_PROVIDER_BUILD_SHA256:
        failures.append("provider_build_sha256_pin_unexpected")
    if lock.get("canonical_owner") != "customs-data-platform":
        failures.append("canonical_owner_unexpected")
    if lock.get("consumer") != "customs-ai-agent":
        failures.append("consumer_owner_unexpected")

    report = {
        "schema_version": _safe_code(lock.get("schema_version")),
        "expected_schema_sha256": EXPECTED_EVIDENCE_SCHEMA_SHA256,
        "locked_schema_sha256": _safe_sha256(locked_sha256),
        "consumer_schema_sha256": consumer_sha256,
        "consumer_matches_pinned_schema": bool(
            consumer_sha256 and consumer_sha256 == locked_sha256 == EXPECTED_EVIDENCE_SCHEMA_SHA256
        ),
        "provider_release": _safe_code(lock.get("provider_release")),
        "pinned_provider_release_revision": _safe_code(lock.get("provider_revision")),
        "provider_build_sha256": _safe_sha256(lock.get("provider_build_sha256")),
        "provider_repository_head_status": "not_observed_http_only",
    }
    return report, sorted(set(failures))


def _validate_loopback_base_url(base_url: str) -> None:
    try:
        parsed = urlsplit(base_url)
        hostname = parsed.hostname
        _ = parsed.port
        is_loopback = False
        if hostname and hostname.casefold() == "localhost":
            is_loopback = True
        elif hostname:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        if (
            parsed.scheme not in {"http", "https"}
            or not is_loopback
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError
    except (ValueError, TypeError):
        raise EvidenceLiveEvaluationError("unsafe_base_url") from None


def _explicit_env_values(env_file: Path | None) -> dict[str, str]:
    if env_file is None:
        return {}
    try:
        if not env_file.expanduser().resolve(strict=True).is_file():
            raise OSError
        raw = dotenv_values(dotenv_path=env_file, verbose=False)
    except (OSError, UnicodeError, ValueError):
        raise EvidenceLiveEvaluationError("env_file_unreadable") from None
    return {name: value for name in _EVIDENCE_ENV_KEYS if isinstance((value := raw.get(name)), str)}


def _configured_value(
    name: str,
    file_values: dict[str, str],
    default: str | None = None,
) -> str | None:
    if name in os.environ:
        return os.environ[name]
    return file_values.get(name, default)


def _retriever_options(env_file: Path | None) -> dict[str, Any]:
    file_values = _explicit_env_values(env_file)
    token = _configured_value("EVIDENCE_SERVICE_TOKEN", file_values)
    token_header = _configured_value(
        "EVIDENCE_SERVICE_TOKEN_HEADER",
        file_values,
        "X-Demo-Token",
    )
    if not token_header or not _SAFE_HEADER_NAME.fullmatch(token_header.strip()):
        raise EvidenceLiveEvaluationError("token_header_invalid")
    return {
        "service_token": token if token else None,
        "token_header": token_header.strip(),
        "contract_identity_mode": "required",
        "max_retries": _HTTP_MAX_RETRIES,
    }


def _retrieval_is_incomplete(state: dict[str, Any]) -> bool:
    results = state.get("retrieval_results") or []
    return bool(
        state.get("retrieval_status") != "ok"
        or state.get("retrieval_coverage") != "complete"
        or not state.get("knowledge_snapshot_id")
        or state.get("retrieval_errors")
        or not results
        or any(
            result.get("coverage") != "complete"
            or result.get("hs_resolution") not in {"exact", "unique_descendant"}
            or bool(result.get("errors"))
            for result in results
        )
    )


def _safe_workflow_report(
    state: dict[str, Any],
    *,
    supported_review_as_of: str,
) -> dict[str, Any]:
    snapshot_id = state.get("knowledge_snapshot_id")
    retrieval_errors = state.get("retrieval_errors") or []
    results = state.get("retrieval_results") or []
    resolution_counts: dict[str, int] = {}
    result_error_codes: list[str] = []
    for result in results:
        resolution = _safe_code(result.get("hs_resolution"))
        resolution_counts[resolution] = resolution_counts.get(resolution, 0) + 1
        result_error_codes.extend(
            _safe_code(error.get("code"))
            for error in result.get("errors") or []
            if isinstance(error, dict)
        )
    retrieval_error_codes = [
        _safe_code(error.get("code")) for error in retrieval_errors if isinstance(error, dict)
    ]
    return {
        "supported_review_as_of": supported_review_as_of,
        "retrieval_status": _safe_code(state.get("retrieval_status")),
        "retrieval_coverage": _safe_code(state.get("retrieval_coverage")),
        "knowledge_snapshot_present": bool(snapshot_id),
        "knowledge_snapshot_id_sha256": (
            _sha256_bytes(str(snapshot_id).encode("utf-8")) if snapshot_id else None
        ),
        "retrieval_result_count": len(results),
        "hs_resolution_counts": dict(sorted(resolution_counts.items())),
        "retrieval_error_codes": sorted(set([*retrieval_error_codes, *result_error_codes])),
        "validation_issue_count": len(state.get("validation_issues") or []),
        "regulatory_finding_count": len(state.get("regulatory_findings") or []),
        "final_status": _safe_code(state.get("final_status")),
        "requires_human_review": bool(state.get("requires_human_review")),
    }


def _base_report(
    *,
    contract: dict[str, Any],
    status: str,
    failures: list[str],
    network_mode: str,
    workflow: dict[str, Any] | None = None,
    agent_workflow_count: int = 0,
    http_counts: _EvidenceHttpCounts | None = None,
    readiness_status: str = "not_attempted",
    scenario_id: str = "happy_path",
) -> dict[str, Any]:
    observed = http_counts or _EvidenceHttpCounts()
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scope": REPORT_SCOPE,
        "status": status,
        "network_mode": network_mode,
        "readiness_status": readiness_status,
        "scenario_id": scenario_id,
        "case_count": agent_workflow_count,
        "agent_workflow_count": agent_workflow_count,
        "readiness_get_count": observed.readiness_get_count,
        "retrieval_post_count": observed.retrieval_post_count,
        "other_http_count": observed.other_http_count,
        "http_count_unit": "adapter_urlopen_attempt_including_retries",
        "contract": contract,
        "workflow": workflow,
        "failures": [{"code": _safe_code(code)} for code in failures],
        "redaction": {
            "request_field_values": "OMITTED",
            "evidence_text": "OMITTED",
            "source_uri": "OMITTED",
            "token": "OMITTED",
            "env_file": "OMITTED",
            "absolute_paths": "OMITTED",
        },
    }


def evaluate(
    *,
    base_url: str = DEFAULT_BASE_URL,
    env_file: Path | None = None,
    live: bool = False,
    lock_path: Path = LOCK_PATH,
    allow_unready_diagnostic: bool = False,
    diagnostic_review_as_of: date | None = None,
    scenario_id: str = "happy_path",
) -> tuple[int, dict[str, Any]]:
    if scenario_id not in ALLOWED_SCENARIOS:
        contract, _ = _contract_preflight(lock_path=lock_path)
        return 1, _base_report(
            contract=contract,
            status="FAIL",
            failures=["scenario_not_allowed"],
            network_mode="disabled_scenario_preflight_failed",
            scenario_id="UNSAFE_SCENARIO_REDACTED",
        )
    contract, failures = _contract_preflight(lock_path=lock_path)
    if failures:
        return 1, _base_report(
            contract=contract,
            status="FAIL",
            failures=failures,
            network_mode="disabled_contract_preflight_failed",
            scenario_id=scenario_id,
        )
    try:
        _validate_loopback_base_url(base_url)
    except EvidenceLiveEvaluationError as exc:
        return 1, _base_report(
            contract=contract,
            status="FAIL",
            failures=[exc.code],
            network_mode="disabled_unsafe_url",
            scenario_id=scenario_id,
        )
    if not live:
        return 1, _base_report(
            contract=contract,
            status="FAIL",
            failures=["live_flag_required"],
            network_mode="preflight_only_network_disabled",
            scenario_id=scenario_id,
        )

    with _observe_evidence_http_calls() as http_counts:
        if http_counts.observer_error_code is not None:
            return 1, _base_report(
                contract=contract,
                status="FAIL",
                failures=[http_counts.observer_error_code],
                network_mode="disabled_http_observer_busy",
                http_counts=http_counts,
                scenario_id=scenario_id,
            )
        try:
            options = _retriever_options(env_file)
            retriever = HttpRegulationRetriever(base_url, **options)
        except EvidenceLiveEvaluationError as exc:
            return 1, _base_report(
                contract=contract,
                status="FAIL",
                failures=[exc.code],
                network_mode="disabled_retriever_preflight_failed",
                http_counts=http_counts,
                scenario_id=scenario_id,
            )
        except Exception:
            return 1, _base_report(
                contract=contract,
                status="FAIL",
                failures=["retriever_preflight_failed"],
                network_mode="disabled_retriever_preflight_failed",
                http_counts=http_counts,
                scenario_id=scenario_id,
            )
        readiness_status = "ready"
        try:
            review_as_of = fetch_supported_review_as_of(base_url, **options)
        except Exception:
            if not allow_unready_diagnostic or diagnostic_review_as_of is None:
                return 1, _base_report(
                    contract=contract,
                    status="FAIL",
                    failures=["evidence_readiness_failed"],
                    network_mode="live",
                    http_counts=http_counts,
                    readiness_status="failed",
                    scenario_id=scenario_id,
                )
            review_as_of = diagnostic_review_as_of
            readiness_status = "failed_diagnostic_override"
        try:
            state = run_review(
                ReviewRunRequest(
                    scenario_id=scenario_id,
                    review_run_id="evidence-live-eval-v1",
                    review_as_of=review_as_of,
                    llm_mode="mock",
                ),
                retriever=retriever,
            )
        except Exception:
            return 1, _base_report(
                contract=contract,
                status="FAIL",
                failures=["evidence_workflow_failed"],
                network_mode=(
                    "live_unready_diagnostic"
                    if readiness_status == "failed_diagnostic_override"
                    else "live"
                ),
                agent_workflow_count=1,
                http_counts=http_counts,
                readiness_status=readiness_status,
                scenario_id=scenario_id,
            )

        workflow = _safe_workflow_report(
            state,
            supported_review_as_of=review_as_of.isoformat(),
        )
        incomplete = _retrieval_is_incomplete(state)
        final_status = state.get("final_status")
        requires_human_review = bool(state.get("requires_human_review"))
        decision_failures: list[str] = []
        if incomplete:
            if final_status == "completed":
                decision_failures.append("incomplete_retrieval_completed")
            if final_status != "needs_review" or not requires_human_review:
                decision_failures.append("incomplete_retrieval_not_routed_to_review")
            workflow["retrieval_decision_class"] = "incomplete_fail_closed"
        else:
            if final_status not in {"completed", "needs_review", "blocked"}:
                decision_failures.append("complete_retrieval_outcome_invalid")
            elif (final_status == "completed") == requires_human_review:
                decision_failures.append("complete_retrieval_human_flag_inconsistent")
            workflow["retrieval_decision_class"] = "complete_contract"

        policy_passed = not decision_failures
        http_measurement_failures: list[str] = []
        if http_counts.other_http_count:
            http_measurement_failures.append("unexpected_evidence_http_attempt")
        if readiness_status == "ready":
            if http_counts.readiness_get_count not in _READY_READINESS_GET_RANGE:
                http_measurement_failures.append("evidence_readiness_http_count_invalid")
            if http_counts.retrieval_post_count not in _READY_RETRIEVAL_POST_RANGE:
                http_measurement_failures.append("evidence_retrieval_http_count_invalid")
        decision_failures.extend(http_measurement_failures)
        if readiness_status == "failed_diagnostic_override":
            decision_failures.append("evidence_readiness_incomplete")
        passed = not decision_failures
        workflow["fail_closed_policy_passed"] = policy_passed
        workflow["release_readiness_passed"] = (
            readiness_status == "ready" and not http_measurement_failures
        )
        return (0 if passed else 1), _base_report(
            contract=contract,
            status=(
                "PASS"
                if passed
                else (
                    "BLOCKED"
                    if readiness_status == "failed_diagnostic_override" and policy_passed
                    else "FAIL"
                )
            ),
            failures=decision_failures,
            network_mode=(
                "live_unready_diagnostic"
                if readiness_status == "failed_diagnostic_override"
                else "live"
            ),
            workflow=workflow,
            agent_workflow_count=1,
            http_counts=http_counts,
            readiness_status=readiness_status,
            scenario_id=scenario_id,
        )


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one opt-in synthetic Agent workflow against the pinned Evidence HTTP API."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--scenario",
        choices=sorted(ALLOWED_SCENARIOS),
        default="happy_path",
        help=(
            "Synthetic workflow fixture. Use evidence_live_exact_hsk10 for the "
            "current exact-HSK10 non-release Data diagnostic."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Required to permit readiness and one Evidence retrieval workflow.",
    )
    parser.add_argument(
        "--allow-unready-diagnostic",
        action="store_true",
        help=(
            "If readiness fails, permit one non-release fail-closed diagnostic. "
            "Requires --diagnostic-review-as-of and always exits nonzero."
        ),
    )
    parser.add_argument(
        "--diagnostic-review-as-of",
        type=_parse_date,
        help=(
            "Explicit snapshot-effective date for the unready diagnostic path. "
            "It never converts failed readiness into a release PASS."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exit_code, report = evaluate(
        base_url=args.base_url,
        env_file=args.env_file,
        live=args.live,
        allow_unready_diagnostic=args.allow_unready_diagnostic,
        diagnostic_review_as_of=args.diagnostic_review_as_of,
        scenario_id=args.scenario,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
