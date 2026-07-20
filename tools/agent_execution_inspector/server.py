"""Serve the localhost-only text Agent execution inspector.

This engineering tool deliberately does not alter the packaged production
Evidence release lock.  It accepts one externally generated, non-release
three-law diagnostic manifest, overlays that identity only while a single
diagnostic run is executing, and reuses the real Claude + Evidence-v2
transport.  It can never make a release decision.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import ipaddress
import json
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import ValidationError

import customs_ai_agent.adapters.retrieval as retrieval_module
from customs_ai_agent.adapters.llm import build_document_analyzer, ensure_llm_mode_available
from customs_ai_agent.adapters.retrieval import (
    ProductionHttpRegulationRetriever,
    validate_evidence_base_url,
)
from customs_ai_agent.contracts.evidence_runtime_release import (
    EvidenceRuntimeReleaseIdentity,
    canonical_json_bytes,
)
from customs_ai_agent.runner import run_review_v2, to_review_run_result_v2
from customs_ai_agent.schemas.models import (
    EvidenceRetrievalRequest,
    EvidenceRetrievalResponse,
    ReviewRunRequestV2,
    ServiceReviewRunRequestV2,
)


SCOPE = "DIAGNOSTIC_ONLY_3_OF_58"
ENVELOPE_SCHEMA_VERSION = "agent-execution-inspector/v1"
MANIFEST_SCHEMA_VERSION = "three-law-evidence-diagnostic/v1"
MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_DOCUMENTS = 6
MAX_TOTAL_TEXT_CHARS = 500_000
MAX_MANIFEST_BYTES = 256 * 1024
MAX_TOKEN_BYTES = 1024
MAX_PUBLIC_RESULT_BYTES = 8 * 1024 * 1024
MAX_RUN_RECORDS = 8
RUN_RECORD_TTL_SECONDS = 15 * 60.0
RUN_SCHEMA_VERSION = "agent-execution-inspector-run/v1"
SCENARIO_CATALOG_SCHEMA_VERSION = "agent-execution-inspector-scenarios/v1"
MAX_SCENARIO_CATALOG_BYTES = 512 * 1024
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
INSPECTOR_HEADER = "X-Agent-Inspector-Request"
INSPECTOR_HEADER_VALUE = "1"
STAGES = (
    "load_case",
    "classify_documents",
    "extract_documents",
    "normalize_documents",
    "pair_documents",
    "validate_cross_documents",
    "self_correct_extraction",
    "retrieve_regulations",
    "compliance_review",
    "generate_xai_report",
    "route_to_human_or_complete",
    "redact_sensitive_inputs",
)
_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9\-._~+/=]{32,512}$")
_SAFE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SAFE_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "api_key",
        "anthropic_api_key",
        "authorization",
        "content_text",
        "data_base64",
        "documents",
        "dsn",
        "env_file",
        "evidence_service_token",
        "file_path",
        "filename",
        "image_pages",
        "manifest_path",
        "ocr_ledger_path",
        "pdf_pages",
        "raw_prompt",
        "raw_response",
        "source_path",
        "service_token",
        "token",
    }
)
_NO_STORE_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; img-src 'none'; "
        "connect-src 'self'; object-src 'none'; base-uri 'none'; "
        "frame-ancestors 'none'; form-action 'self'"
    ),
    "Cross-Origin-Resource-Policy": "same-origin",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}
_SCENARIO_SECRET_MARKERS = (
    "anthropic_api_key=",
    "authorization: bearer ",
    "neo4j://",
    "postgresql://",
    "qdrant_api_key=",
    "sk-ant-",
)
_SCENARIO_URI_PATTERN = re.compile(r"(?i)(?<![A-Za-z0-9+.-])[A-Za-z][A-Za-z0-9+.-]*://")
_SCENARIO_WINDOWS_DRIVE_PATH_PATTERN = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
_SCENARIO_UNC_OR_DEVICE_PATH_PATTERN = re.compile(r"(?:\\\\|//)")
_SCENARIO_POSIX_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9._~+\-])/(?!/)"
    r"[A-Za-z0-9._~+\-]+(?:/[A-Za-z0-9._~+\-]+)*"
)


class InspectorConfigurationError(RuntimeError):
    """A configuration failure safe to reduce to a content-free code."""


class InspectorExecutionError(RuntimeError):
    """A diagnostic execution failure safe to reduce to a content-free code."""


@dataclass(frozen=True, slots=True)
class DiagnosticManifest:
    identity: EvidenceRuntimeReleaseIdentity
    data_revision: str
    expected_production_law_count: int
    observed_target_law_count: int
    observed_article_count: int
    qdrant_point_count: int
    neo4j_article_count: int
    law_article_manifest_sha256: str
    postgres_qdrant_ids_sha256: str
    qdrant_point_ids_sha256: str
    postgres_graph_ids_sha256: str
    neo4j_article_ids_sha256: str

    def public_metadata(self) -> dict[str, Any]:
        return {
            "scope": SCOPE,
            "release_ready": False,
            "rag_quality_evaluated": False,
            "expected_production_law_count": self.expected_production_law_count,
            "observed_target_law_count": self.observed_target_law_count,
            "observed_article_count": self.observed_article_count,
            "qdrant_point_count": self.qdrant_point_count,
            "neo4j_article_count": self.neo4j_article_count,
            "law_article_manifest_sha256": self.law_article_manifest_sha256,
            "postgres_qdrant_ids_sha256": self.postgres_qdrant_ids_sha256,
            "qdrant_point_ids_sha256": self.qdrant_point_ids_sha256,
            "postgres_graph_ids_sha256": self.postgres_graph_ids_sha256,
            "neo4j_article_ids_sha256": self.neo4j_article_ids_sha256,
            "data_revision": self.data_revision,
            "evidence_runtime_release": self.identity.to_wire(),
        }


@dataclass(frozen=True, slots=True)
class InspectorSettings:
    host: str
    port: int
    max_body_bytes: int = MAX_BODY_BYTES
    max_documents: int = MAX_DOCUMENTS
    max_total_text_chars: int = MAX_TOTAL_TEXT_CHARS
    run_timeout_seconds: float = 300.0
    max_run_records: int = MAX_RUN_RECORDS
    run_record_ttl_seconds: float = RUN_RECORD_TTL_SECONDS

    def __post_init__(self) -> None:
        if self.host not in LOOPBACK_HOSTS:
            raise InspectorConfigurationError("non_loopback_bind_forbidden")
        if not 1 <= self.port <= 65_535:
            raise InspectorConfigurationError("invalid_port")
        if not 1 <= self.max_documents <= MAX_DOCUMENTS:
            raise InspectorConfigurationError("invalid_document_limit")
        if not 1 <= self.max_body_bytes <= MAX_BODY_BYTES:
            raise InspectorConfigurationError("invalid_body_limit")
        if not 1 <= self.max_total_text_chars <= MAX_TOTAL_TEXT_CHARS:
            raise InspectorConfigurationError("invalid_text_limit")
        if not 1 <= self.run_timeout_seconds <= 600:
            raise InspectorConfigurationError("invalid_run_timeout")
        if not 1 <= self.max_run_records <= MAX_RUN_RECORDS:
            raise InspectorConfigurationError("invalid_run_record_limit")
        if not 1 <= self.run_record_ttl_seconds <= 3600:
            raise InspectorConfigurationError("invalid_run_record_ttl")


@dataclass(slots=True)
class InspectorRunRecord:
    """Bounded in-memory record containing public output, never request input."""

    run_id: str
    created_monotonic: float
    completion: asyncio.Event = field(default_factory=asyncio.Event)
    status: str = "running"
    completed_monotonic: float | None = None
    result: dict[str, Any] | None = None
    error_code: str | None = None

    def public_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": self.status,
            "terminal": self.status in {"succeeded", "failed"},
            "poll_after_ms": 500,
        }
        if self.status == "succeeded" and self.result is not None:
            payload["result"] = self.result
        elif self.status == "failed":
            payload["error"] = {
                "code": self.error_code or "review_execution_failed",
                "message": "The local diagnostic run could not be completed.",
                "retryable": True,
            }
        _assert_safe_public_value(payload)
        return payload


class InspectorExecutor(Protocol):
    def health(self) -> dict[str, Any]: ...

    def execute(self, request: ServiceReviewRunRequestV2) -> dict[str, Any]: ...


def _is_loopback(value: str | None) -> bool:
    if value is None:
        return False
    if value.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _single_header_values(request: Request, name: bytes) -> list[str]:
    values: list[str] = []
    for raw_name, raw_value in request.scope.get("headers", []):
        if raw_name.lower() != name:
            continue
        try:
            values.append(raw_value.decode("ascii"))
        except UnicodeDecodeError:
            return []
    return values


def _valid_local_authority(value: str, settings: InspectorSettings) -> bool:
    if not value or any(character.isspace() for character in value):
        return False
    try:
        parsed = urlsplit(f"//{value}")
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.hostname is not None
        and _is_loopback(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
        and (port == settings.port or (port is None and settings.port in {80, 443}))
    )


def _valid_local_origin(value: str, settings: InspectorSettings) -> bool:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.casefold() == "http"
        and parsed.hostname is not None
        and _is_loopback(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
        and (port == settings.port or (port is None and settings.port == 80))
    )


def _safe_key_token(value: object) -> str:
    return "".join(
        character
        for character in str(value).casefold()
        if character.isascii() and character.isalnum()
    )


_FORBIDDEN_OUTPUT_KEY_TOKENS = frozenset(_safe_key_token(key) for key in _FORBIDDEN_OUTPUT_KEYS)


def _assert_safe_public_value(value: Any, *, depth: int = 0) -> None:
    if depth > 32:
        raise InspectorExecutionError("public_output_depth_exceeded")
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise InspectorExecutionError("public_output_nonfinite")
        return
    if isinstance(value, list):
        if len(value) > 20_000:
            raise InspectorExecutionError("public_output_list_too_large")
        for item in value:
            _assert_safe_public_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 20_000:
            raise InspectorExecutionError("public_output_object_too_large")
        tokens = {_safe_key_token(key) for key in value}
        if tokens.intersection(_FORBIDDEN_OUTPUT_KEY_TOKENS):
            raise InspectorExecutionError("public_output_forbidden_key")
        for item in value.values():
            _assert_safe_public_value(item, depth=depth + 1)
        return
    raise InspectorExecutionError("public_output_type_invalid")


def _assert_bounded_public_result(value: dict[str, Any]) -> None:
    _assert_safe_public_value(value)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise InspectorExecutionError("public_output_serialization_failed") from None
    if len(encoded) > MAX_PUBLIC_RESULT_BYTES:
        raise InspectorExecutionError("public_output_size_exceeded")


def _prune_run_records(
    records: dict[str, InspectorRunRecord],
    *,
    settings: InspectorSettings,
    now: float | None = None,
    reserve_slot: bool = False,
) -> None:
    observed_now = time.monotonic() if now is None else now
    expired = [
        run_id
        for run_id, record in records.items()
        if record.completed_monotonic is not None
        and observed_now - record.completed_monotonic >= settings.run_record_ttl_seconds
    ]
    for run_id in expired:
        records.pop(run_id, None)

    while reserve_slot and len(records) >= settings.max_run_records:
        terminal = [record for record in records.values() if record.completed_monotonic is not None]
        if not terminal:
            break
        oldest = min(
            terminal,
            key=lambda record: record.completed_monotonic or record.created_monotonic,
        )
        records.pop(oldest.run_id, None)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        raise InspectorConfigurationError("invalid_json") from None
    if not isinstance(value, dict):
        raise InspectorConfigurationError("json_object_required")
    return value


def _read_regular_file(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    if not path.is_absolute():
        raise InspectorConfigurationError(f"{label}_absolute_path_required")
    if path.is_symlink() or not path.is_file():
        raise InspectorConfigurationError(f"{label}_regular_file_required")
    try:
        size = path.stat().st_size
    except OSError:
        raise InspectorConfigurationError(f"{label}_unavailable") from None
    if size <= 0 or size > maximum_bytes:
        raise InspectorConfigurationError(f"{label}_size_invalid")
    try:
        raw = path.read_bytes()
    except OSError:
        raise InspectorConfigurationError(f"{label}_unavailable") from None
    if len(raw) != size:
        raise InspectorConfigurationError(f"{label}_changed_during_read")
    return raw


def load_service_token(path: Path) -> str:
    raw = _read_regular_file(path, maximum_bytes=MAX_TOKEN_BYTES, label="service_token")
    try:
        value = raw.decode("ascii").strip()
    except UnicodeDecodeError:
        raise InspectorConfigurationError("service_token_invalid") from None
    if (
        _TOKEN_PATTERN.fullmatch(value) is None
        or "\r" in value
        or "\n" in value
        or len(set(value)) < 12
    ):
        raise InspectorConfigurationError("service_token_invalid")
    return value


def _required_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InspectorConfigurationError("diagnostic_manifest_invalid")
    return value


def load_diagnostic_manifest(path: Path) -> DiagnosticManifest:
    payload = _strict_json_object(
        _read_regular_file(
            path,
            maximum_bytes=MAX_MANIFEST_BYTES,
            label="diagnostic_manifest",
        )
    )
    if (
        payload.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or payload.get("scope") != SCOPE
        or payload.get("release_ready") is not False
        or payload.get("rag_quality_evaluated") is not False
    ):
        raise InspectorConfigurationError("diagnostic_manifest_invalid")
    data_revision = payload.get("data_revision")
    if not isinstance(data_revision, str) or _SAFE_REVISION.fullmatch(data_revision) is None:
        raise InspectorConfigurationError("diagnostic_manifest_invalid")
    expected = _required_int(payload, "expected_production_law_count")
    expected_diagnostic = _required_int(payload, "expected_diagnostic_law_count")
    observed = _required_int(payload, "observed_target_law_count")
    observed_total = _required_int(payload, "observed_total_law_count")
    articles = _required_int(payload, "observed_article_count")
    qdrant = _required_int(payload, "qdrant_point_count")
    neo4j = _required_int(payload, "neo4j_article_count")
    if (
        expected != 58
        or expected_diagnostic != 3
        or observed != 3
        or observed_total != 3
        or not 0 < observed < expected
        or articles <= 0
        or qdrant != articles
        or neo4j != articles
    ):
        raise InspectorConfigurationError("diagnostic_manifest_invalid")
    digests = {
        field: payload.get(field)
        for field in (
            "law_article_manifest_sha256",
            "postgres_qdrant_ids_sha256",
            "qdrant_point_ids_sha256",
            "postgres_graph_ids_sha256",
            "neo4j_article_ids_sha256",
        )
    }
    if any(
        not isinstance(value, str) or _SAFE_SHA256.fullmatch(value) is None
        for value in digests.values()
    ):
        raise InspectorConfigurationError("diagnostic_manifest_invalid")
    if (
        digests["postgres_qdrant_ids_sha256"] != digests["qdrant_point_ids_sha256"]
        or digests["postgres_graph_ids_sha256"] != digests["neo4j_article_ids_sha256"]
    ):
        raise InspectorConfigurationError("diagnostic_manifest_invalid")
    if payload.get("runtime_stability") != {
        "omp_num_threads": "1",
        "mkl_num_threads": "1",
        "tokenizers_parallelism": "false",
        "hf_hub_offline": "1",
    }:
        raise InspectorConfigurationError("diagnostic_manifest_invalid")
    try:
        identity = EvidenceRuntimeReleaseIdentity.from_mapping(
            payload.get("evidence_runtime_release")
        )
    except ValueError:
        raise InspectorConfigurationError("diagnostic_manifest_invalid") from None
    if identity.provider_revision != data_revision:
        raise InspectorConfigurationError("diagnostic_manifest_invalid")
    _assert_safe_public_value(payload)
    return DiagnosticManifest(
        identity=identity,
        data_revision=data_revision,
        expected_production_law_count=expected,
        observed_target_law_count=observed,
        observed_article_count=articles,
        qdrant_point_count=qdrant,
        neo4j_article_count=neo4j,
        law_article_manifest_sha256=str(digests["law_article_manifest_sha256"]),
        postgres_qdrant_ids_sha256=str(digests["postgres_qdrant_ids_sha256"]),
        qdrant_point_ids_sha256=str(digests["qdrant_point_ids_sha256"]),
        postgres_graph_ids_sha256=str(digests["postgres_graph_ids_sha256"]),
        neo4j_article_ids_sha256=str(digests["neo4j_article_ids_sha256"]),
    )


def load_scenario_catalog(catalog_path: Path, schema_path: Path) -> dict[str, Any]:
    """Load the curated browser demo catalog and reject unsafe or drifting inputs."""

    catalog = _strict_json_object(
        _read_regular_file(
            catalog_path,
            maximum_bytes=MAX_SCENARIO_CATALOG_BYTES,
            label="scenario_catalog",
        )
    )
    schema = _strict_json_object(
        _read_regular_file(
            schema_path,
            maximum_bytes=MAX_SCENARIO_CATALOG_BYTES,
            label="scenario_catalog_schema",
        )
    )
    try:
        Draft202012Validator.check_schema(schema)
        validation_errors = list(Draft202012Validator(schema).iter_errors(catalog))
    except SchemaError:
        raise InspectorConfigurationError("scenario_catalog_schema_invalid") from None
    if validation_errors:
        raise InspectorConfigurationError("scenario_catalog_invalid")
    if catalog.get("schema_version") != SCENARIO_CATALOG_SCHEMA_VERSION:
        raise InspectorConfigurationError("scenario_catalog_version_invalid")

    scenarios = catalog.get("scenarios")
    if not isinstance(scenarios, list):
        raise InspectorConfigurationError("scenario_catalog_invalid")
    identifiers = [scenario.get("id") for scenario in scenarios]
    orders = [scenario.get("order") for scenario in scenarios]
    if (
        len(identifiers) != len(set(identifiers))
        or len(orders) != len(set(orders))
        or orders != sorted(orders)
    ):
        raise InspectorConfigurationError("scenario_catalog_order_invalid")

    for scenario in scenarios:
        try:
            ServiceReviewRunRequestV2.model_validate(
                {
                    "case_id": scenario["case_id"],
                    "documents": scenario["documents"],
                    "llm_mode": "llm",
                    "review_run_id": "scenario_catalog_validation",
                    "review_as_of": "2000-01-01",
                    "trade_type": scenario["trade_type"],
                    "jurisdiction": "KR",
                    "uploaded_requirement_document_codes": scenario[
                        "uploaded_requirement_document_codes"
                    ],
                }
            )
        except (KeyError, ValidationError):
            raise InspectorConfigurationError("scenario_catalog_contract_invalid") from None

    pending: list[Any] = [catalog]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str):
            normalized = value.casefold()
            if any(marker in normalized for marker in _SCENARIO_SECRET_MARKERS):
                raise InspectorConfigurationError("scenario_catalog_secret_marker_detected")
            if (
                _SCENARIO_URI_PATTERN.search(value)
                or _SCENARIO_WINDOWS_DRIVE_PATH_PATTERN.search(value)
                or _SCENARIO_UNC_OR_DEVICE_PATH_PATTERN.search(value)
                or _SCENARIO_POSIX_ABSOLUTE_PATH_PATTERN.search(value)
            ):
                raise InspectorConfigurationError("scenario_catalog_non_synthetic_marker_detected")
    return catalog


_IDENTITY_OVERLAY_LOCK = threading.RLock()


@contextmanager
def diagnostic_runtime_identity(
    identity: EvidenceRuntimeReleaseIdentity,
) -> Iterator[None]:
    """Temporarily supply one external diagnostic identity to the V2 consumer.

    This process is single-flight.  The original packaged loader is restored
    before the lock is released, including on every failure path.
    """

    with _IDENTITY_OVERLAY_LOCK:
        original = retrieval_module.load_released_evidence_runtime_identity
        retrieval_module.load_released_evidence_runtime_identity = lambda: identity
        try:
            yield
        finally:
            retrieval_module.load_released_evidence_runtime_identity = original


class DiagnosticProductionHttpRegulationRetriever(ProductionHttpRegulationRetriever):
    """Overlay a non-release identity only while Evidence code is executing."""

    def __init__(
        self,
        base_url: str,
        *,
        diagnostic_identity: EvidenceRuntimeReleaseIdentity,
        service_token: str,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        self._diagnostic_identity = diagnostic_identity
        with diagnostic_runtime_identity(diagnostic_identity):
            super().__init__(
                base_url,
                service_token=service_token,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )

    def preflight(
        self,
        review_as_of: date | None = None,
        *,
        _deadline: float | None = None,
    ) -> EvidenceRuntimeReleaseIdentity:
        with diagnostic_runtime_identity(self._diagnostic_identity):
            return super().preflight(review_as_of, _deadline=_deadline)

    def retrieve(
        self,
        request: EvidenceRetrievalRequest,
    ) -> EvidenceRetrievalResponse:
        with diagnostic_runtime_identity(self._diagnostic_identity):
            return super().retrieve(request)


def _pairing_projection(state: Mapping[str, Any]) -> dict[str, Any]:
    raw = state.get("pairing_result")
    pairing = raw if isinstance(raw, dict) else {}
    scalar_fields = (
        "invoice_document_id",
        "packing_list_document_id",
        "minimum_line_item_pair_score",
        "document_set_status",
        "unmatched_invoice_item_status",
        "unmatched_packing_item_status",
        "line_item_pairing_status",
    )
    list_fields = (
        "line_item_links",
        "line_item_pairing_ambiguities",
        "document_type_conflicts",
        "missing_required_document_types",
        "unmatched_invoice_item_ids",
        "unmatched_packing_item_ids",
    )
    projected = {field: pairing.get(field) for field in scalar_fields}
    projected.update({field: list(pairing.get(field) or []) for field in list_fields})
    case_profile = pairing.get("case_profile")
    projected["case_profile"] = dict(case_profile) if isinstance(case_profile, dict) else {}
    _assert_safe_public_value(projected)
    return projected


def _input_manifest(request: ServiceReviewRunRequestV2) -> list[dict[str, Any]]:
    return [
        {
            "document_id": document.document_id,
            "declared_type": document.declared_type,
            "character_count": len(document.content_text or ""),
            "utf8_byte_count": len((document.content_text or "").encode("utf-8")),
            "sha256": hashlib.sha256((document.content_text or "").encode("utf-8")).hexdigest(),
        }
        for document in request.documents
    ]


def _execution_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    evidence = result.get("rag_evidence")
    findings = result.get("regulatory_findings")
    issues = result.get("validation_issues")
    model_calls = result.get("model_calls")
    model_errors = result.get("model_errors")
    retrieval_results = result.get("retrieval_results")
    return {
        "status": result.get("status"),
        "requires_human_review": bool(result.get("requires_human_review")),
        "model_call_count": len(model_calls) if isinstance(model_calls, list) else 0,
        "model_error_count": len(model_errors) if isinstance(model_errors, list) else 0,
        "retrieval_result_count": (
            len(retrieval_results) if isinstance(retrieval_results, list) else 0
        ),
        "evidence_count": len(evidence) if isinstance(evidence, list) else 0,
        "finding_count": len(findings) if isinstance(findings, list) else 0,
        "validation_issue_count": len(issues) if isinstance(issues, list) else 0,
        "mock_fallback_count": 0,
    }


def _assert_no_mock_fallback(result: Mapping[str, Any]) -> None:
    def has_mock_marker(value: object) -> bool:
        if isinstance(value, str):
            return value.casefold() == "mock"
        if isinstance(value, list):
            return any(has_mock_marker(item) for item in value)
        if isinstance(value, dict):
            return any(has_mock_marker(key) or has_mock_marker(item) for key, item in value.items())
        return False

    for document in result.get("document_results") or []:
        if (
            isinstance(document, dict)
            and str(document.get("extraction_provider")).casefold() == "mock"
        ):
            raise InspectorExecutionError("mock_fallback_detected")
    for evidence in result.get("rag_evidence") or []:
        if not isinstance(evidence, dict):
            continue
        methods = evidence.get("retrieval_methods") or []
        raw_scores = evidence.get("raw_scores") or {}
        if has_mock_marker(methods) or has_mock_marker(raw_scores):
            raise InspectorExecutionError("mock_fallback_detected")


class ActualInspectorExecutor:
    def __init__(
        self,
        *,
        env_file: Path,
        evidence_base_url: str,
        service_token: str,
        diagnostic_manifest: DiagnosticManifest,
        timeout_seconds: float = 90.0,
    ) -> None:
        if not env_file.is_absolute() or env_file.is_symlink() or not env_file.is_file():
            raise InspectorConfigurationError("env_file_regular_absolute_required")
        normalized_url = validate_evidence_base_url(evidence_base_url)
        if not _is_loopback(urlsplit(normalized_url).hostname):
            raise InspectorConfigurationError("diagnostic_evidence_must_be_loopback")
        if not 1 <= timeout_seconds <= 300:
            raise InspectorConfigurationError("evidence_timeout_invalid")
        # Validate credential/model configuration without starting a network call.
        claude_settings = ensure_llm_mode_available("llm", env_file)
        if claude_settings is None:
            raise InspectorConfigurationError("claude_configuration_missing")
        self._env_file = env_file
        self._evidence_base_url = normalized_url
        self._service_token = service_token
        self._manifest = diagnostic_manifest
        self._timeout_seconds = timeout_seconds
        self._model_id = claude_settings.model

    def _retriever(self) -> ProductionHttpRegulationRetriever:
        return DiagnosticProductionHttpRegulationRetriever(
            self._evidence_base_url,
            diagnostic_identity=self._manifest.identity,
            service_token=self._service_token,
            timeout_seconds=self._timeout_seconds,
            max_retries=1,
        )

    def health(self) -> dict[str, Any]:
        try:
            retriever = self._retriever()
            observed = retriever.preflight()
            data_status = "ready" if observed == self._manifest.identity else "identity_mismatch"
        except Exception:
            data_status = "unavailable"
        payload = {
            "schema_version": ENVELOPE_SCHEMA_VERSION,
            "status": "ready" if data_status == "ready" else "degraded",
            "execution_modes": {
                "claude": "ACTUAL_CLAUDE",
                "evidence_http": "ACTUAL_HTTP_DATA_V2",
                "data_scope": SCOPE,
                "input": "TEXT_ONLY_OCR_NOT_USED",
                "release": "NOT_RELEASE_READY",
                "mock_fallback": "FORBIDDEN",
            },
            "claude_configured": True,
            "model_id": self._model_id,
            "data_status": data_status,
            "data": self._manifest.public_metadata(),
        }
        _assert_safe_public_value(payload)
        return payload

    def execute(self, request: ServiceReviewRunRequestV2) -> dict[str, Any]:
        if any(
            document.content_text is None or document.image_pages or document.pdf_pages is not None
            for document in request.documents
        ):
            raise InspectorExecutionError("text_only_request_required")
        workflow_request = ReviewRunRequestV2.model_validate(
            request.model_dump(mode="python", warnings=False)
        )
        retriever = self._retriever()
        # Dependency admission precedes Claude so an unavailable diagnostic
        # provider cannot consume model tokens.
        observed = retriever.preflight(request.review_as_of)
        if observed != self._manifest.identity:
            raise InspectorExecutionError("diagnostic_identity_mismatch")
        analyzer = build_document_analyzer(self._env_file)
        try:
            state = run_review_v2(
                workflow_request,
                retriever=retriever,
                analyzer=analyzer,
            )
            result_model = to_review_run_result_v2(state)
        finally:
            analyzer.close()

        result = result_model.model_dump(mode="json", warnings=False)
        _assert_no_mock_fallback(result)
        result_sha256 = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
        envelope = {
            "schema_version": ENVELOPE_SCHEMA_VERSION,
            "scope": SCOPE,
            "release_ready": False,
            "rag_quality_evaluated": False,
            "execution_modes": {
                "claude": "ACTUAL_CLAUDE",
                "evidence_http": "ACTUAL_HTTP_DATA_V2",
                "data_scope": SCOPE,
                "input": "TEXT_ONLY_OCR_NOT_USED",
                "release": "NOT_RELEASE_READY",
                "mock_fallback": "FORBIDDEN",
            },
            "input_manifest": _input_manifest(request),
            "stages": list(STAGES),
            "pairing": _pairing_projection(state),
            "summary": _execution_summary(result),
            "data": self._manifest.public_metadata(),
            "result_sha256": result_sha256,
            "result": result,
        }
        _assert_safe_public_value(envelope)
        return envelope


async def _read_bounded_json(request: Request, maximum_bytes: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            declared_bytes = int(declared)
        except ValueError:
            raise InspectorExecutionError("invalid_content_length") from None
        if declared_bytes < 0 or declared_bytes > maximum_bytes:
            raise InspectorExecutionError("request_too_large")
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > maximum_bytes:
            raise InspectorExecutionError("request_too_large")
        chunks.append(bytes(chunk))
    return b"".join(chunks)


def _safe_error(
    code: str,
    status_code: int,
    *,
    retryable: bool = False,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        headers=_NO_STORE_HEADERS,
        content={
            "error": {
                "request_id": uuid.uuid4().hex,
                "code": code,
                "message": "The local diagnostic request could not be completed.",
                "retryable": retryable,
            }
        },
    )


def create_app(
    *,
    settings: InspectorSettings,
    executor: InspectorExecutor,
    html_path: Path | None = None,
    scenario_catalog_path: Path | None = None,
    scenario_catalog_schema_path: Path | None = None,
) -> FastAPI:
    selected_html = html_path or Path(__file__).with_name("inspector.html")
    if selected_html.is_symlink() or not selected_html.is_file():
        raise InspectorConfigurationError("inspector_html_unavailable")
    html_bytes = _read_regular_file(
        selected_html.resolve(),
        maximum_bytes=2 * 1024 * 1024,
        label="inspector_html",
    )
    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise InspectorConfigurationError("inspector_html_invalid") from None
    catalog = load_scenario_catalog(
        (scenario_catalog_path or Path(__file__).with_name("scenario_catalog.json")).resolve(),
        (
            scenario_catalog_schema_path or Path(__file__).with_name("scenario_catalog.schema.json")
        ).resolve(),
    )

    app = FastAPI(
        title="Local Agent Execution Inspector",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.inspector_settings = settings
    app.state.executor = executor
    app.state.scenario_catalog = catalog
    app.state.execution_admission = asyncio.Semaphore(1)
    app.state.background_tasks = set()
    app.state.run_records = {}

    def track_background(task: asyncio.Task[Any]) -> None:
        app.state.background_tasks.add(task)
        task.add_done_callback(app.state.background_tasks.discard)

    async def expire_run(run_id: str, completed_monotonic: float) -> None:
        await asyncio.sleep(settings.run_record_ttl_seconds)
        records: dict[str, InspectorRunRecord] = app.state.run_records
        record = records.get(run_id)
        if record is not None and record.completed_monotonic == completed_monotonic:
            records.pop(run_id, None)

    async def execute_run(
        record: InspectorRunRecord,
        parsed: ServiceReviewRunRequestV2,
        admission: asyncio.Semaphore,
    ) -> None:
        try:
            envelope = await asyncio.to_thread(executor.execute, parsed)
            _assert_bounded_public_result(envelope)
        except asyncio.CancelledError:
            record.status = "failed"
            record.error_code = "review_execution_cancelled"
            raise
        except Exception:
            record.status = "failed"
            record.error_code = "review_execution_failed"
        else:
            record.result = envelope
            record.status = "succeeded"
        finally:
            completed_monotonic = time.monotonic()
            record.completed_monotonic = completed_monotonic
            record.completion.set()
            admission.release()
            expiry = asyncio.create_task(expire_run(record.run_id, completed_monotonic))
            track_background(expiry)

    @app.middleware("http")
    async def local_boundary(request: Request, call_next: Callable[..., Any]) -> Any:
        client_host = request.client.host if request.client is not None else None
        if not _is_loopback(client_host):
            return _safe_error("loopback_client_required", 403)
        host_values = _single_header_values(request, b"host")
        if len(host_values) != 1 or not _valid_local_authority(host_values[0], settings):
            return _safe_error("local_host_required", 403)
        origin_values = _single_header_values(request, b"origin")
        if len(origin_values) > 1 or (
            origin_values and not _valid_local_origin(origin_values[0], settings)
        ):
            return _safe_error("same_origin_required", 403)
        response = await call_next(request)
        for name, value in _NO_STORE_HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(html, headers=_NO_STORE_HEADERS)

    @app.get("/api/health")
    async def health() -> JSONResponse:
        try:
            payload = await asyncio.to_thread(executor.health)
            _assert_safe_public_value(payload)
        except Exception:
            payload = {
                "schema_version": ENVELOPE_SCHEMA_VERSION,
                "status": "degraded",
                "data_status": "unavailable",
                "release_ready": False,
            }
        return JSONResponse(payload, headers=_NO_STORE_HEADERS)

    @app.get("/api/scenarios")
    async def scenarios() -> JSONResponse:
        return JSONResponse(app.state.scenario_catalog, headers=_NO_STORE_HEADERS)

    @app.get("/api/runs/{run_id}")
    async def run_status(request: Request, run_id: str) -> JSONResponse:
        if request.headers.get(INSPECTOR_HEADER) != INSPECTOR_HEADER_VALUE:
            return _safe_error("inspector_header_required", 403)
        if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
            return _safe_error("run_not_found", 404)
        records: dict[str, InspectorRunRecord] = request.app.state.run_records
        _prune_run_records(records, settings=settings)
        record = records.get(run_id)
        if record is None:
            return _safe_error("run_not_found", 404)
        return JSONResponse(record.public_payload(), headers=_NO_STORE_HEADERS)

    @app.post("/api/reviews")
    async def review(request: Request) -> JSONResponse:
        if request.headers.get(INSPECTOR_HEADER) != INSPECTOR_HEADER_VALUE:
            return _safe_error("inspector_header_required", 403)
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().casefold()
        if content_type != "application/json":
            return _safe_error("application_json_required", 415)
        try:
            raw = await _read_bounded_json(request, settings.max_body_bytes)
            payload = _strict_json_object(raw)
            parsed = ServiceReviewRunRequestV2.model_validate(payload)
            if len(parsed.documents) > settings.max_documents:
                raise InspectorExecutionError("document_limit_exceeded")
            if (
                sum(len(document.content_text or "") for document in parsed.documents)
                > settings.max_total_text_chars
            ):
                raise InspectorExecutionError("text_limit_exceeded")
            if any(
                document.content_text is None
                or document.image_pages
                or document.pdf_pages is not None
                for document in parsed.documents
            ):
                raise InspectorExecutionError("text_only_request_required")
        except InspectorExecutionError as exc:
            status = 413 if exc.args and "limit" in str(exc.args[0]) else 422
            return _safe_error(str(exc.args[0]) if exc.args else "request_invalid", status)
        except (InspectorConfigurationError, ValidationError):
            return _safe_error("request_invalid", 422)

        admission: asyncio.Semaphore = request.app.state.execution_admission
        if admission.locked():
            return _safe_error("inspector_busy", 503, retryable=True)
        await admission.acquire()
        records: dict[str, InspectorRunRecord] = request.app.state.run_records
        _prune_run_records(records, settings=settings, reserve_slot=True)
        if len(records) >= settings.max_run_records:
            admission.release()
            return _safe_error("inspector_run_capacity", 503, retryable=True)

        record = InspectorRunRecord(
            run_id=uuid.uuid4().hex,
            created_monotonic=time.monotonic(),
        )
        records[record.run_id] = record
        worker = asyncio.create_task(execute_run(record, parsed, admission))
        track_background(worker)
        try:
            await asyncio.wait_for(
                asyncio.shield(record.completion.wait()),
                timeout=settings.run_timeout_seconds,
            )
        except TimeoutError:
            status_code = 200 if record.completion.is_set() else 202
        else:
            status_code = 200
        return JSONResponse(
            record.public_payload(),
            status_code=status_code,
            headers=_NO_STORE_HEADERS,
        )

    return app


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("port must be an integer from 1 to 65535") from None
    if not 1 <= port <= 65_535:
        raise argparse.ArgumentTypeError("port must be an integer from 1 to 65535")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--three-law-diagnostic", action="store_true", required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--evidence-base-url", required=True)
    parser.add_argument("--evidence-token-file", type=Path, required=True)
    parser.add_argument("--diagnostic-manifest", type=Path, required=True)
    parser.add_argument("--host", choices=sorted(LOOPBACK_HOSTS), default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8021)
    parser.add_argument("--run-timeout-seconds", type=float, default=300.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.live is not True or args.three_law_diagnostic is not True:
            raise InspectorConfigurationError("explicit_diagnostic_ack_required")
        settings = InspectorSettings(
            host=args.host,
            port=args.port,
            run_timeout_seconds=args.run_timeout_seconds,
        )
        manifest = load_diagnostic_manifest(args.diagnostic_manifest.resolve())
        token = load_service_token(args.evidence_token_file.resolve())
        executor = ActualInspectorExecutor(
            env_file=args.env_file.resolve(),
            evidence_base_url=args.evidence_base_url,
            service_token=token,
            diagnostic_manifest=manifest,
        )
        app = create_app(settings=settings, executor=executor)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "event": "agent_execution_inspector_configuration_failed",
                    "exception_type": type(exc).__name__,
                },
                sort_keys=True,
            )
        )
        return 78
    print(
        json.dumps(
            {
                "event": "agent_execution_inspector_starting",
                "host": settings.host,
                "port": settings.port,
                "scope": SCOPE,
                "release_ready": False,
            },
            sort_keys=True,
        )
    )
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="warning",
        access_log=False,
        proxy_headers=False,
        server_header=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
