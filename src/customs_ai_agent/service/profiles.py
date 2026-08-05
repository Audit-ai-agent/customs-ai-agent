"""Loopback-only development and functional-UAT Agent execution."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from collections.abc import Mapping
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import ValidationError

from customs_ai_agent.adapters.llm import (
    ClaudeDocumentAnalyzer,
    ClaudeSettings,
    DocumentAnalysis,
)
from customs_ai_agent.adapters.retrieval import RetrievalError, validate_evidence_destination
from customs_ai_agent.schemas.llm import (
    ClaudeDocumentCandidate,
    ClaudeFieldCandidate,
    ClaudeLineItemCandidate,
)
from customs_ai_agent.scenario_transcript import ScenarioTranscriptRoot, ScenarioTranscriptRun
from customs_ai_agent.schemas.v3 import (
    RequiredActionV1,
    ReviewRunResultV3,
    ServiceReviewRunRequestV3,
    canonical_json_sha256,
    review_result_v3_sha256,
)
from customs_ai_agent.service.runtime_errors import (
    RuntimeConfigurationError,
    RuntimeExecutionError,
    RuntimeInputError,
)
from customs_ai_agent.service.settings import ServiceSettings


ServiceMode = Literal["local-development", "functional-uat"]
SERVICE_MODE_HEADER = "X-Customs-Service-Mode"
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_HASH = re.compile(r"^[a-f0-9]{64}$")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


_HTTP = build_opener(_NoRedirect())


def service_mode(settings: ServiceSettings) -> ServiceMode:
    if settings.service_mode in {"local-development", "functional-uat"}:
        return settings.service_mode
    raise RuntimeConfigurationError("service_mode_invalid", "The Agent profile mode is invalid.")


def _service_secret_values(settings: ServiceSettings) -> tuple[str, str]:
    """Values held by this process that must never survive in a transcript."""

    return (
        settings.bearer_token.get_secret_value(),
        settings.evidence_service_token.get_secret_value(),
    )


def _transcript_root(
    settings: ServiceSettings, supplied: ScenarioTranscriptRoot | None = None
) -> ScenarioTranscriptRoot:
    """Functional UAT always records locally; development remains explicit opt-in."""

    mode = service_mode(settings)
    root = supplied or ScenarioTranscriptRoot.for_service_mode(
        mode, sensitive_values=_service_secret_values(settings)
    )
    return root.with_sensitive_values(_service_secret_values(settings))


def _one_header(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    values = getattr(headers, "get_all", lambda _name: None)(name)
    if isinstance(values, (list, tuple)):
        return values[0] if len(values) == 1 and isinstance(values[0], str) else None
    value = getattr(headers, "get", lambda _name: None)(name)
    return value if isinstance(value, str) else None


class ProfileEvidenceRetriever:
    """Minimal Data HTTP client: readiness, one retrieve call, and mode matching."""

    readiness_path = "/internal/rag/v3/health/ready"
    retrieve_path = "/internal/rag/v3/retrieve"

    def __init__(
        self, settings: ServiceSettings, *, transcript_root: ScenarioTranscriptRoot | None = None
    ) -> None:
        self.mode = service_mode(settings)
        self._timeout = settings.readiness_timeout_seconds
        self._transcript_root = _transcript_root(settings, transcript_root)
        token = settings.evidence_service_token
        if token is None:
            raise RuntimeConfigurationError(
                "evidence_service_token_missing", "The Data service token is missing."
            )
        try:
            self._base_url = validate_evidence_destination(settings.evidence_base_url)
            host = urlsplit(self._base_url).hostname
            if host is None or not ipaddress.ip_address(host).is_loopback:
                raise ValueError
        except (TypeError, ValueError):
            raise RuntimeConfigurationError(
                "evidence_base_url_invalid", "The Data service address is invalid."
            ) from None
        self._token = token.get_secret_value()

    def _headers(self, *, json_body: bool, request_id: str | None = None) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._token}",
            SERVICE_MODE_HEADER: self.mode,
        }
        if json_body:
            headers["Content-Type"] = "application/json; charset=utf-8"
        if request_id:
            headers["X-Request-ID"] = request_id
        return headers

    def _request(
        self,
        path: str,
        *,
        method: Literal["GET", "POST"],
        payload: dict[str, Any] | None = None,
        review_run_id: str | None = None,
    ) -> dict[str, Any]:
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            if payload is not None
            else None
        )
        request = Request(  # noqa: S310 - the base URL is verified loopback
            f"{self._base_url}{path}",
            data=encoded,
            headers=self._headers(
                json_body=encoded is not None,
                request_id=(payload or {}).get("request_id"),
            ),
            method=method,
        )
        try:
            with _HTTP.open(request, timeout=self._timeout) as response:
                if _one_header(response, SERVICE_MODE_HEADER) != self.mode:
                    self._record_failure(
                        review_run_id, method=method, path=path, error_code="data_mode_mismatch"
                    )
                    raise RetrievalError("Data service mode mismatch")
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except RetrievalError:
            raise
        except (HTTPError, URLError, TimeoutError, OSError):
            self._record_failure(
                review_run_id, method=method, path=path, error_code="data_request_failed"
            )
            raise RetrievalError("Data service request failed") from None
        if not isinstance(raw, bytes) or len(raw) > _MAX_RESPONSE_BYTES:
            self._record_failure(
                review_run_id, method=method, path=path, error_code="data_response_invalid"
            )
            raise RetrievalError("Data service response is invalid")
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            self._record_failure(
                review_run_id, method=method, path=path, error_code="data_response_invalid"
            )
            raise RetrievalError("Data service response is invalid") from None
        if not isinstance(decoded, dict):
            self._record_failure(
                review_run_id, method=method, path=path, error_code="data_response_invalid"
            )
            raise RetrievalError("Data service response is invalid")
        self._record_exchange(
            review_run_id,
            method=method,
            path=path,
            request_body=payload,
            response_body=decoded,
        )
        return decoded

    def _record_exchange(
        self,
        review_run_id: str | None,
        *,
        method: Literal["GET", "POST"],
        path: str,
        request_body: dict[str, Any] | None,
        response_body: dict[str, Any],
    ) -> None:
        if review_run_id is None:
            return
        transcript = self._transcript_root.for_run(review_run_id)
        if transcript is not None:
            transcript.write(
                "agent-data",
                "http_exchange",
                {
                    "method": method,
                    "path": path,
                    "request_body": request_body,
                    "response_body": response_body,
                },
            )

    def _record_failure(
        self,
        review_run_id: str | None,
        *,
        method: Literal["GET", "POST"],
        path: str,
        error_code: str,
    ) -> None:
        if review_run_id is None:
            return
        transcript = self._transcript_root.for_run(review_run_id)
        if transcript is not None:
            # Do not retain a provider error body or the original request on a
            # failed exchange.  The safe code is enough to diagnose the edge.
            transcript.write(
                "agent-data",
                "http_failed",
                {"method": method, "path": path, "error_code": error_code},
            )

    def readiness(self, *, review_run_id: str | None = None) -> dict[str, str]:
        response = self._request(self.readiness_path, method="GET", review_run_id=review_run_id)
        if response != {"status": "ready"}:
            self._record_failure(
                review_run_id,
                method="GET",
                path=self.readiness_path,
                error_code="data_readiness_invalid",
            )
            raise RetrievalError("Data service is not ready")
        return {"status": "ready"}

    def retrieve(self, request: dict[str, Any]) -> dict[str, Any]:
        review_run_id = request.get("review_run_id")
        if not isinstance(review_run_id, str):
            raise RetrievalError("Data service request is invalid")
        self.readiness(review_run_id=review_run_id)
        response = self._request(
            self.retrieve_path,
            method="POST",
            payload=request,
            review_run_id=review_run_id,
        )
        if (
            response.get("schema_version") != "3.0"
            or response.get("request_id") != request["request_id"]
            or response.get("status") not in {"ok", "partial", "failed"}
            or not isinstance(response.get("results"), list)
        ):
            self._record_failure(
                review_run_id,
                method="POST",
                path=self.retrieve_path,
                error_code="data_contract_invalid",
            )
            raise RetrievalError("Data service response does not satisfy the shared contract")
        return response


class _DeterministicDocumentAnalyzer:
    """Stable no-key analyzer for the default profile."""

    model_id = "development-fixture-deterministic"

    def analyze(
        self, document: Mapping[str, Any], *, transcript: ScenarioTranscriptRun | None = None
    ) -> DocumentAnalysis:
        del transcript
        text = str(document.get("content_text") or "")
        upper = text.upper()
        document_type: Literal["invoice", "unknown"] = (
            "invoice" if "INVOICE" in upper else "unknown"
        )
        line_items: list[ClaudeLineItemCandidate] = []
        match = re.search(r"\bHS(?:\s*CODE)?\s*[:#-]?\s*([0-9]{6,10})\b", text, re.I)
        if match is not None:
            quote = text[max(0, match.start() - 160) : min(len(text), match.end() + 160)]
            line_items.append(
                ClaudeLineItemCandidate(
                    source_quote=quote,
                    fields=[
                        ClaudeFieldCandidate(
                            name="hs_code", value=match.group(1), source_quote=quote
                        )
                    ],
                )
            )
        return DocumentAnalysis(
            candidate=ClaudeDocumentCandidate(
                document_type=document_type,
                classification_source_quote="INVOICE" if document_type == "invoice" else "",
                line_items=line_items,
            ),
            model_id=self.model_id,
        )


def _origin_country(candidate: ClaudeDocumentCandidate) -> str | None:
    for field in candidate.fields:
        if field.name in {"origin_country", "country_of_origin"}:
            value = field.value.strip().upper()
            if re.fullmatch(r"[A-Z]{2}", value):
                return value
    return None


def _retrieval_items(
    request: ServiceReviewRunRequestV3,
    analyses: list[DocumentAnalysis],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for document, analysis in zip(request.documents, analyses, strict=True):
        candidate = analysis.candidate
        hs_candidates: list[tuple[str | None, str]] = [
            (field.value, line.source_quote)
            for line in candidate.line_items
            for field in line.fields
            if field.name == "hs_code" and re.fullmatch(r"[0-9]{6,10}", field.value)
        ]
        hs_candidates.extend(
            (field.value, field.source_quote)
            for field in candidate.fields
            if field.name == "hs_code" and re.fullmatch(r"[0-9]{6,10}", field.value)
        )
        if not hs_candidates:
            hs_candidates = [(None, document.content_text[:2000])]
        for index, (hs_code, description) in enumerate(hs_candidates, start=1):
            item_id = f"{document.document_id}-{index}"
            items.append(
                {
                    "item_id": item_id[:128],
                    "description": str(description).strip()[:2000] or "customs document",
                    "hs_code": hs_code,
                    "origin_country": _origin_country(candidate),
                    "destination_country": "KR",
                }
            )
    return items[:100]


def _retrieve_request(
    request: ServiceReviewRunRequestV3, analyses: list[DocumentAnalysis]
) -> dict[str, Any]:
    request_id = "rag-" + hashlib.sha256(request.review_run_id.encode("utf-8")).hexdigest()[:32]
    return {
        "schema_version": "3.0",
        "request_id": request_id,
        "case_id": request.case_id,
        "review_run_id": request.review_run_id,
        "review_as_of": request.review_as_of.isoformat(),
        "jurisdiction": request.jurisdiction,
        "trade_type": request.trade_type,
        "top_k": 10,
        "items": _retrieval_items(request, analyses),
    }


def _evidence_from_data(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for item_result in response.get("results", []):
        if not isinstance(item_result, dict):
            raise RetrievalError("Data service result is invalid")
        for item in item_result.get("evidence", []):
            if not isinstance(item, dict):
                raise RetrievalError("Data service evidence is invalid")
            required = (
                "evidence_id",
                "item_id",
                "title",
                "text",
                "source_uri",
                "source_type",
                "version_hash",
            )
            if any(not isinstance(item.get(name), str) or not item[name] for name in required):
                raise RetrievalError("Data service evidence is invalid")
            if _HASH.fullmatch(item["version_hash"]) is None:
                raise RetrievalError("Data service evidence is invalid")
            evidence.append(
                {
                    "evidence_id": item["evidence_id"],
                    "item_id": item["item_id"],
                    "title": item["title"],
                    "text": item["text"],
                    "source_uri": item["source_uri"],
                    "source_type": item["source_type"],
                    "source_hash": item["version_hash"],
                    "law_id": item.get("law_id") if isinstance(item.get("law_id"), str) else None,
                    "law_name": item.get("law_name")
                    if isinstance(item.get("law_name"), str)
                    else None,
                    "eligibility": item.get("eligibility"),
                    "required_document_code": item.get("required_document_code"),
                    "required_document_name": item.get("required_document_name"),
                }
            )
    return evidence


def _findings(
    request: ServiceReviewRunRequestV3,
    analyses: list[DocumentAnalysis],
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for document, analysis in zip(request.documents, analyses, strict=True):
        if analysis.candidate.document_type == "unknown":
            findings.append(
                {
                    "kind": "validation_issue",
                    "code": "document_type_unknown",
                    "severity": "medium",
                    "message": "문서 유형을 확인할 수 없어 담당자 검토가 필요합니다.",
                    "item_id": document.document_id,
                    "evidence_ids": [],
                }
            )
        has_hs = any(
            field.name == "hs_code"
            for line in analysis.candidate.line_items
            for field in line.fields
        ) or any(field.name == "hs_code" for field in analysis.candidate.fields)
        if not has_hs:
            findings.append(
                {
                    "kind": "validation_issue",
                    "code": "hs_code_not_found",
                    "severity": "medium",
                    "message": "HS 코드를 확인할 수 없어 담당자 검토가 필요합니다.",
                    "item_id": document.document_id,
                    "evidence_ids": [],
                }
            )
    for item in evidence:
        if item.pop("eligibility") == "finding":
            code = item.pop("required_document_code") or "data_requirement"
            name = item.pop("required_document_name") or item["title"]
            findings.append(
                {
                    "kind": "regulatory_finding",
                    "code": str(code)[:128],
                    "severity": "medium",
                    "message": f"필수 확인 항목: {name}",
                    "item_id": item["item_id"],
                    "evidence_ids": [item["evidence_id"]],
                }
            )
        else:
            item.pop("required_document_code")
            item.pop("required_document_name")
    return findings


def _actions(findings: list[dict[str, Any]]) -> list[RequiredActionV1]:
    actions = [
        RequiredActionV1(
            action_code="inspect_" + re.sub(r"[^a-z0-9_]+", "_", finding["code"].casefold())[:96],
            message=finding["message"],
        )
        for finding in findings
    ]
    actions.append(
        RequiredActionV1(
            action_code="human_review_required",
            message="자동완료하지 않습니다. 담당자가 문서와 법령 Evidence를 검토해야 합니다.",
        )
    )
    return actions


class ProfileRuntime:
    def __init__(
        self, settings: ServiceSettings, *, transcript_root: ScenarioTranscriptRoot | None = None
    ) -> None:
        self.mode = service_mode(settings)
        self._closed = False
        self._transcript_root = _transcript_root(settings, transcript_root)
        if self.mode == "local-development":
            self._analyzer: Any = _DeterministicDocumentAnalyzer()
        else:
            try:
                claude_settings = ClaudeSettings.from_environment()
                self._transcript_root = self._transcript_root.with_sensitive_values(
                    (claude_settings.api_key.get_secret_value(),)
                )
                self._analyzer = ClaudeDocumentAnalyzer.from_settings(claude_settings)
            except Exception:
                raise RuntimeConfigurationError(
                    "functional_uat_claude_configuration_invalid",
                    "Functional UAT requires a valid local Claude configuration.",
                ) from None
        self._retriever = ProfileEvidenceRetriever(settings, transcript_root=self._transcript_root)

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeExecutionError("runtime_closed", "The Agent runtime is closed.")

    @staticmethod
    def _decode(payload: bytes) -> ServiceReviewRunRequestV3:
        try:
            parsed = ServiceReviewRunRequestV3.model_validate(json.loads(payload.decode("utf-8")))
        except (UnicodeDecodeError, ValidationError, ValueError):
            raise RuntimeInputError(
                "review_request_invalid", "The Agent request does not satisfy the service contract."
            ) from None
        if _SAFE_RUN_ID.fullmatch(parsed.review_run_id) is None:
            raise RuntimeInputError("review_request_invalid", "The Agent request is invalid.")
        return parsed

    def review_v3_json(self, payload: bytes) -> ReviewRunResultV3:
        self._assert_open()
        request = self._decode(payload)
        transcript = self._transcript_root.for_run(request.review_run_id)
        if transcript is not None and self.mode == "local-development":
            transcript.write(
                "claude",
                "not_called",
                {
                    "provider": "claude",
                    "called": False,
                    "reason": "local-development uses the deterministic document analyzer",
                },
            )
        try:
            analyses = [
                self._analyzer.analyze(
                    document.model_dump(mode="json", warnings=False), transcript=transcript
                )
                for document in request.documents
            ]
            data_response = self._retriever.retrieve(_retrieve_request(request, analyses))
            evidence = _evidence_from_data(data_response)
            findings = _findings(request, analyses, evidence)
        except RetrievalError:
            raise RuntimeExecutionError(
                "evidence_dependency_unready",
                "The Data Evidence service is unavailable or profile-mismatched.",
            ) from None
        except RuntimeExecutionError:
            raise
        except Exception:
            raise RuntimeExecutionError(
                "profile_review_execution_failed", "The local Agent review could not be completed."
            ) from None
        data_status = str(data_response["status"])
        status = "blocked" if data_status == "failed" else "needs_review"
        execution_status = (
            "failed"
            if status == "blocked"
            else ("partial" if data_status == "partial" else "succeeded")
        )
        assessment = (
            "issues_detected"
            if findings
            else ("no_issue_detected" if data_status == "ok" else "indeterminate")
        )
        payload_result: dict[str, Any] = {
            "schema_version": "3.0",
            "case_id": request.case_id,
            "review_run_id": request.review_run_id,
            "review_as_of": request.review_as_of,
            "trade_type": request.trade_type,
            "jurisdiction": request.jurisdiction,
            "status": status,
            "requires_human_review": True,
            "execution_status": execution_status,
            "assessment": assessment,
            "input_manifest_sha256": canonical_json_sha256(request),
            "report": {
                "overall_status": status,
                "summary": f"문서 {len(request.documents)}건, Evidence {len(evidence)}건, 확인 항목 {len(findings)}건입니다.",
                "disclaimer": "이 결과는 기능 확인용이며 담당자의 최종 검토가 필요합니다.",
            },
            "evidence": evidence,
            "findings": findings,
            "required_actions": _actions(findings),
            "engine": {
                "kind": "deterministic" if self.mode == "local-development" else "claude",
                "model_id": str(self._analyzer.model_id),
            },
            "result_sha256": "0" * 64,
        }
        payload_result["result_sha256"] = review_result_v3_sha256(payload_result)
        try:
            result = ReviewRunResultV3.model_validate(payload_result)
        except ValidationError:
            raise RuntimeExecutionError(
                "profile_result_invalid", "The Agent result could not be validated."
            ) from None
        return result

    def close(self) -> None:
        self._closed = True
        close = getattr(self._analyzer, "close", None)
        if callable(close):
            close()


def build_profile_runtime(settings: ServiceSettings) -> ProfileRuntime:
    return ProfileRuntime(settings)


def profile_readiness_probe(settings: ServiceSettings) -> Any:
    return ProfileEvidenceRetriever(settings).readiness


__all__ = [
    "ProfileEvidenceRetriever",
    "ProfileRuntime",
    "SERVICE_MODE_HEADER",
    "build_profile_runtime",
    "profile_readiness_probe",
    "service_mode",
]
