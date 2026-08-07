"""Loopback-only development and functional-UAT Agent execution."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from collections.abc import Mapping
from typing import Any, Literal, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import ValidationError

from customs_ai_agent.adapters.llm import (
    ClaudeDocumentAnalyzer,
    ClaudeGroundedResponder,
    ClaudeSettings,
    DocumentAnalysis,
)
from customs_ai_agent.adapters.retrieval import RetrievalError, validate_evidence_destination
from customs_ai_agent.schemas.llm import (
    ClaudeDocumentCandidate,
    ClaudeFieldCandidate,
    ClaudeLineItemCandidate,
)
from customs_ai_agent.schemas.assistant import (
    AssistantAnswerRequestV1,
    AssistantAnswerResponseV1,
)
from customs_ai_agent.scenario_transcript import ScenarioTranscriptRoot, ScenarioTranscriptRun
from customs_ai_agent.schemas.v3 import (
    EngineV3,
    EvidenceV3,
    FindingV3,
    ReportSectionV3,
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
from customs_ai_agent.service.assistant import (
    deterministic_assistant_answer,
    provider_answer_allowed,
    provider_snapshot,
    response_from_provider_candidate,
)
from customs_ai_agent.service.report import (
    build_grounded_report_sections,
    validate_grounded_report_sections,
)
from customs_ai_agent.service.user_text import review_result_markers, reviewer_text_is_safe
from customs_ai_agent.service.workflow import (
    finding_for_evidence_requirement,
    run_bounded_workflow,
)


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

    def record_retrieve_skipped(self, *, review_run_id: str) -> None:
        """Record an input-side skip without making a Data request.

        Functional-UAT transcripts require an Agent-to-Data stream for every
        review.  An empty commodity plan is deliberately not sent to Data,
        because the shared retrieve contract rejects it.  This event keeps
        that distinction visible without treating the Data service as
        unavailable or retaining document values.
        """

        if self.mode != "functional-uat":
            return
        transcript = self._transcript_root.for_run(review_run_id)
        if transcript is not None:
            transcript.write(
                "agent-data",
                "retrieve_skipped",
                {
                    "reason_code": "no_hs_qualified_commodity",
                    "retrieve_called": False,
                },
            )


class _DeterministicDocumentAnalyzer:
    """Stable no-key text extractor used by the default profile.

    It intentionally performs only deterministic extraction.  Normalisation,
    matching, arithmetic, and findings are owned by ``workflow.py`` so the
    development fixture exercises the same bounded workflow as Claude UAT.
    """

    model_id = "development-fixture-deterministic"

    def analyze(
        self,
        document: Mapping[str, Any],
        *,
        transcript: ScenarioTranscriptRun | None = None,
        focus_fields: tuple[str, ...] = (),
    ) -> DocumentAnalysis:
        del transcript, focus_fields
        text = str(document.get("content_text") or "")
        document_type, classification_quote = _deterministic_document_type(text)
        fields = _deterministic_document_fields(text, document_type)
        line_items: list[ClaudeLineItemCandidate] = []
        for line in text.splitlines():
            candidate = _deterministic_line_item(line)
            if candidate is not None:
                line_items.append(candidate)
        if not line_items:
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
                classification_source_quote=classification_quote,
                fields=fields,
                line_items=line_items,
            ),
            model_id=self.model_id,
        )


def _deterministic_document_type(
    text: str,
) -> tuple[
    Literal["invoice", "packing_list", "bill_of_lading", "certificate_of_origin", "unknown"], str
]:
    rules: tuple[tuple[str, str], ...] = (
        ("bill_of_lading", r"\b(?:bill\s+of\s+lading|b\s*/\s*l)\b"),
        ("packing_list", r"\b(?:packing\s+list|packing\s+note)\b"),
        ("certificate_of_origin", r"\b(?:certificate\s+of\s+origin|c\s*/\s*o)\b"),
        ("invoice", r"\b(?:commercial\s+invoice|invoice)\b"),
    )
    matches: list[tuple[int, int, str, str]] = []
    for priority, (document_type, pattern) in enumerate(rules):
        match = re.search(pattern, text, re.IGNORECASE)
        if match is not None:
            matches.append((match.start(), priority, document_type, match.group(0)))
    if matches:
        _position, _priority, document_type, quote = min(matches)
        return cast(
            Literal[
                "invoice",
                "packing_list",
                "bill_of_lading",
                "certificate_of_origin",
                "unknown",
            ],
            document_type,
        ), quote
    return "unknown", ""


def _deterministic_document_fields(text: str, document_type: str) -> list[ClaudeFieldCandidate]:
    rules: dict[str, tuple[tuple[str, str], ...]] = {
        "invoice": (
            ("invoice_no", r"(?:invoice\s*(?:no|number)|inv\s*no)\s*[:#-]\s*([^\n|;]+)"),
            (
                "total_amount",
                r"(?:invoice\s+total|total\s+amount|grand\s+total)\s*[:#-]\s*([^\n|;]+)",
            ),
            (
                "country_of_origin",
                r"(?:country\s+of\s+origin|origin\s+country)\s*[:#-]\s*([^\n|;]+)",
            ),
            ("hs_code", r"\bhs\s*(?:code)?\s*[:#-]\s*([0-9]{6,10})\b"),
        ),
        "packing_list": (
            ("packing_no", r"(?:packing\s*(?:no|number)|pl\s*no)\s*[:#-]\s*([^\n|;]+)"),
            ("invoice_no", r"(?:invoice\s*(?:no|number)|inv\s*no)\s*[:#-]\s*([^\n|;]+)"),
            (
                "bl_no",
                r"(?:b\s*/\s*l\s*(?:no|number)?|bill\s+of\s+lading\s*(?:no|number)?)\s*[:#-]\s*([^\n|;]+)",
            ),
            ("container_numbers", r"(?:container\s*(?:no|number|numbers)?)\s*[:#-]\s*([^\n|;]+)"),
            (
                "total_gross_weight_kg",
                r"(?:total\s+gross\s+weight|gross\s+weight)\s*[:#-]\s*([^\n|;]+)",
            ),
            (
                "country_of_origin",
                r"(?:country\s+of\s+origin|origin\s+country)\s*[:#-]\s*([^\n|;]+)",
            ),
        ),
        "certificate_of_origin": (
            (
                "certificate_no",
                r"(?:certificate\s*(?:no|number)|c\s*/\s*o\s*(?:no|number)?)\s*[:#-]\s*([^\n|;]+)",
            ),
            (
                "origin_country",
                r"(?:country\s+of\s+origin|origin\s+country|origin)\s*[:#-]\s*([^\n|;]+)",
            ),
            ("hs_code", r"\bhs\s*(?:code)?\s*[:#-]\s*([0-9]{6,10})\b"),
        ),
        "bill_of_lading": (
            (
                "bl_no",
                r"(?:b\s*/\s*l\s*(?:no|number)?|bill\s+of\s+lading\s*(?:no|number)?)\s*[:#-]\s*([^\n|;]+)",
            ),
            ("container_numbers", r"(?:container\s*(?:no|number|numbers)?)\s*[:#-]\s*([^\n|;]+)"),
            ("gross_weight_kg", r"(?:gross\s+weight|total\s+gross\s+weight)\s*[:#-]\s*([^\n|;]+)"),
        ),
    }
    fields: list[ClaudeFieldCandidate] = []
    for name, pattern in rules.get(document_type, ()):
        match = re.search(pattern, text, re.IGNORECASE)
        if match is None:
            continue
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        quote = text[line_start : len(text) if line_end < 0 else line_end].strip()
        value = match.group(1).strip()
        if value and value in quote:
            fields.append(ClaudeFieldCandidate(name=name, value=value, source_quote=quote))
    return fields


def _deterministic_line_item(line: str) -> ClaudeLineItemCandidate | None:
    quote = line.strip()
    if not quote or not re.search(
        r"\b(?:description|goods|item\s*\d+|qty|quantity|unit\s+price|hs\s*code)\b",
        quote,
        re.IGNORECASE,
    ):
        return None
    field_rules: tuple[tuple[str, str], ...] = (
        ("description", r"(?:description|goods|item\s*\d+)\s*[:#-]\s*([^|;]+)"),
        ("hs_code", r"\bhs\s*(?:code)?\s*[:#-]\s*([0-9]{6,10})\b"),
        ("quantity", r"(?:quantity|qty)\s*[:#-]\s*([^|;]+)"),
        ("unit", r"\bunit\b(?!\s*price)\s*[:#-]\s*([^|;]+)"),
        ("unit_price", r"(?:unit\s+price|price)\s*[:#-]\s*([^|;]+)"),
        ("amount", r"(?:line\s+amount|amount)\s*[:#-]\s*([^|;]+)"),
        ("origin_country", r"(?:origin\s+country|country\s+of\s+origin|origin)\s*[:#-]\s*([^|;]+)"),
        ("gross_weight_kg", r"(?:gross\s+weight)\s*[:#-]\s*([^|;]+)"),
    )
    fields: list[ClaudeFieldCandidate] = []
    for name, pattern in field_rules:
        match = re.search(pattern, quote, re.IGNORECASE)
        if match is not None and match.group(1).strip() in quote:
            fields.append(
                ClaudeFieldCandidate(
                    name=name,
                    value=match.group(1).strip(),
                    source_quote=quote,
                )
            )
    return ClaudeLineItemCandidate(source_quote=quote, fields=fields) if fields else None


def _retrieve_request(
    request: ServiceReviewRunRequestV3, items: list[dict[str, Any]]
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
        "items": items,
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
    return _dedupe_evidence_records(evidence)


def _requirement_identity(item: Mapping[str, Any]) -> tuple[str, str, str]:
    """Identify one regulatory requirement without using document-local IDs."""

    law_id = str(item.get("law_id") or "").casefold()
    code = str(item.get("required_document_code") or "").casefold()
    name = str(item.get("required_document_name") or item.get("title") or "").casefold()
    return law_id, code or name, "requirement" if code or name else "evidence"


def _dedupe_evidence_records(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one public Evidence record for a repeated regulatory requirement."""

    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in evidence:
        identity = (
            _requirement_identity(item)
            if item.get("eligibility") == "finding"
            else ("", str(item["evidence_id"]), "evidence")
        )
        current = unique.get(identity)
        if current is None:
            item["_linked_item_ids"] = [str(item["item_id"])]
            unique[identity] = item
            continue
        linked_item_ids = current.setdefault("_linked_item_ids", [str(current["item_id"])])
        if isinstance(linked_item_ids, list) and str(item["item_id"]) not in linked_item_ids:
            linked_item_ids.append(str(item["item_id"]))
    return list(unique.values())


def _findings_from_evidence(
    evidence: list[dict[str, Any]],
    *,
    document_ids_by_item: Mapping[str, list[str]],
) -> list[FindingV3]:
    """Create one finding per requirement and retain every linked document."""

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in evidence:
        if item.get("eligibility") != "finding":
            continue
        identity = _requirement_identity(item)
        group = grouped.setdefault(
            identity,
            {
                "code": str(item.get("required_document_code") or "data_requirement")[:128],
                "name": str(item.get("required_document_name") or item["title"]),
                "item_ids": set(),
                "evidence_ids": set(),
                "document_ids": set(),
            },
        )
        item_id = str(item["item_id"])
        linked_item_ids = item.get("_linked_item_ids", [item_id])
        if not isinstance(linked_item_ids, list) or not all(
            isinstance(value, str) and value for value in linked_item_ids
        ):
            linked_item_ids = [item_id]
        group["item_ids"].update(linked_item_ids)
        group["evidence_ids"].add(str(item["evidence_id"]))
        for linked_item_id in linked_item_ids:
            group["document_ids"].update(document_ids_by_item.get(linked_item_id, []))

    findings: list[FindingV3] = []
    for group in grouped.values():
        item_ids = sorted(group["item_ids"])
        findings.append(
            finding_for_evidence_requirement(
                code=group["code"],
                message=f"필수 확인 항목: {group['name']}",
                item_id=item_ids[0],
                evidence_ids=sorted(group["evidence_ids"]),
                document_ids=sorted(group["document_ids"]),
            )
        )
    return findings


def _public_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {
        "evidence_id",
        "item_id",
        "title",
        "text",
        "source_uri",
        "source_type",
        "source_hash",
        "law_id",
        "law_name",
    }
    return [{key: value for key, value in item.items() if key in allowed} for item in evidence]


def _actions(findings: list[FindingV3]) -> list[RequiredActionV1]:
    actions = [
        RequiredActionV1(
            action_code=(
                "inspect_"
                + re.sub(r"[^a-z0-9_]+", "_", finding.code.casefold())[:84]
                + "_"
                + finding.finding_id[-12:]
            )[:128],
            message=finding.message,
            finding_ids=[finding.finding_id],
        )
        for finding in findings
    ]
    actions.append(
        RequiredActionV1(
            action_code="human_review_required",
            message="자동완료하지 않습니다. 담당자가 문서와 법령 근거를 검토해야 합니다.",
            finding_ids=[],
        )
    )
    return actions


def _dedupe_findings(findings: list[FindingV3]) -> list[FindingV3]:
    """Keep the public result stable when sources converge on one finding."""

    unique: dict[str, FindingV3] = {}
    for finding in findings:
        unique.setdefault(finding.finding_id, finding)
    return list(unique.values())


def _provider_report_claim_is_grounded(
    claim: Mapping[str, Any],
    *,
    fact_ids: set[str],
    comparison_ids: set[str],
    evidence_ids: set[str],
) -> bool:
    """Accept only a claim whose every cited ID belongs to this review."""

    references = (
        (claim.get("fact_ids", []), fact_ids),
        (claim.get("comparison_ids", []), comparison_ids),
        (claim.get("evidence_ids", []), evidence_ids),
    )
    has_reference = False
    for values, known_ids in references:
        if not isinstance(values, list):
            return False
        if any(not isinstance(value, str) or value not in known_ids for value in values):
            return False
        has_reference = has_reference or bool(values)
    return has_reference


class ProfileRuntime:
    def __init__(
        self, settings: ServiceSettings, *, transcript_root: ScenarioTranscriptRoot | None = None
    ) -> None:
        self.mode = service_mode(settings)
        self._closed = False
        self._transcript_root = _transcript_root(settings, transcript_root)
        if self.mode == "local-development":
            self._analyzer: Any = _DeterministicDocumentAnalyzer()
            self._grounded_responder: ClaudeGroundedResponder | None = None
        else:
            try:
                claude_settings = ClaudeSettings.from_environment()
                self._transcript_root = self._transcript_root.with_sensitive_values(
                    (claude_settings.api_key.get_secret_value(),)
                )
                self._analyzer = ClaudeDocumentAnalyzer.from_settings(claude_settings)
                try:
                    self._grounded_responder = ClaudeGroundedResponder.from_settings(
                        claude_settings
                    )
                except Exception:
                    # A review remains usable with its deterministic report
                    # fallback if the optional bounded composition adapter is
                    # unavailable after the required extraction adapter loads.
                    self._grounded_responder = None
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

    @staticmethod
    def _decode_assistant(payload: bytes) -> AssistantAnswerRequestV1:
        try:
            parsed = AssistantAnswerRequestV1.model_validate(json.loads(payload.decode("utf-8")))
        except (UnicodeDecodeError, ValidationError, ValueError):
            raise RuntimeInputError(
                "assistant_request_invalid",
                "The assistant request does not satisfy the service contract.",
            ) from None
        if _SAFE_RUN_ID.fullmatch(parsed.review_run_id) is None:
            raise RuntimeInputError(
                "assistant_request_invalid", "The assistant request is invalid."
            )
        return parsed

    def _engine(self) -> EngineV3:
        return EngineV3(
            kind="deterministic" if self.mode == "local-development" else "claude",
            model_id=str(self._analyzer.model_id),
        )

    def _compose_functional_report(
        self,
        result: ReviewRunResultV3,
        *,
        transcript: ScenarioTranscriptRun | None,
    ) -> list[ReportSectionV3]:
        """Try one bounded Claude composition, then retain the deterministic report."""

        fallback = result.report.sections
        responder = self._grounded_responder
        if self.mode != "functional-uat" or responder is None:
            return fallback
        try:
            candidate = responder.respond(
                {"review_snapshot": provider_snapshot(result)},
                transcript=transcript,
                purpose="review_report",
            )
            raw_sections = candidate.get("sections")
            if not isinstance(raw_sections, list):
                raise ValueError("report_sections_missing")
            # A provider can add narrative claims despite the prompt.  Never
            # let an ungrounded or unknown-reference claim cross the public
            # boundary. Preserve grounded provider sections, but fill a
            # missing safe section from the deterministic report instead of
            # discarding all useful Claude composition.
            known_facts = {
                fact.fact_id
                for analysis in result.document_analyses
                for fact in [
                    *analysis.facts,
                    *(fact for item in analysis.line_items for fact in item.facts),
                ]
            }
            known_comparisons = {item.comparison_id for item in result.comparisons}
            known_evidence = {item.evidence_id for item in result.evidence}
            fallback_by_section = {section.section: section for section in fallback}
            has_human_correction = any(
                getattr(fact.source, "source_kind", None) == "human_provided"
                for analysis in result.document_analyses
                for fact in [
                    *analysis.facts,
                    *(fact for item in analysis.line_items for fact in item.facts),
                ]
            )
            provider_sections: dict[str, dict[str, Any]] = {}
            for raw_section in raw_sections:
                if not isinstance(raw_section, Mapping):
                    continue
                section_name = raw_section.get("section")
                if (
                    not isinstance(section_name, str)
                    or section_name not in fallback_by_section
                    or section_name in provider_sections
                    or (section_name == "execution_process" and has_human_correction)
                ):
                    continue
                raw_claims = raw_section.get("claims")
                if not isinstance(raw_claims, list):
                    continue
                grounded_claims = [
                    claim
                    for claim in raw_claims
                    if isinstance(claim, Mapping)
                    and _provider_report_claim_is_grounded(
                        claim,
                        fact_ids=known_facts,
                        comparison_ids=known_comparisons,
                        evidence_ids=known_evidence,
                    )
                ]
                if grounded_claims:
                    provider_sections[section_name] = {
                        **dict(raw_section),
                        "claims": grounded_claims,
                    }
            normalized_sections = [
                provider_sections.get(
                    fallback_section.section,
                    fallback_section.model_dump(mode="json", warnings=False),
                )
                for fallback_section in fallback
            ]
            sections = [ReportSectionV3.model_validate(item) for item in normalized_sections]
            reviewer_text = (
                value
                for section in sections
                for value in (section.title, *(claim.text for claim in section.claims))
            )
            if not reviewer_text_is_safe(
                reviewer_text,
                markers=review_result_markers(result),
                filenames=(analysis.filename for analysis in result.document_analyses),
            ):
                raise ValueError("provider_report_user_text_unsafe")
            validate_grounded_report_sections(
                sections,
                analyses=result.document_analyses,
                comparisons=result.comparisons,
                evidence=result.evidence,
            )
            return sections
        except Exception:
            if transcript is not None:
                transcript.write(
                    "claude",
                    "review_report_fallback",
                    {"reason_code": "provider_report_unavailable_or_ungrounded"},
                )
            return fallback

    def assistant_answer_json(self, payload: bytes) -> AssistantAnswerResponseV1:
        """Answer once from an immutable review snapshot; never mutate Case state."""

        self._assert_open()
        request = self._decode_assistant(payload)
        transcript = self._transcript_root.for_run(request.review_run_id)
        fallback = deterministic_assistant_answer(request, engine=self._engine())
        responder = self._grounded_responder
        provider_grounded_scope = fallback.answer_scope in {
            "case_grounded",
            "evidence_grounded",
        } and provider_answer_allowed(request)
        if self.mode == "functional-uat" and responder is not None and provider_grounded_scope:
            try:
                candidate = responder.respond(
                    {
                        "review_snapshot": provider_snapshot(request.review_result),
                        "question": request.question,
                        "recent_turns": [
                            turn.model_dump(mode="json", warnings=False)
                            for turn in request.recent_turns
                        ],
                    },
                    transcript=transcript,
                    purpose="assistant_answer",
                )
                answer = response_from_provider_candidate(candidate, request, engine=self._engine())
            except Exception:
                if transcript is not None:
                    transcript.write(
                        "claude",
                        "assistant_answer_fallback",
                        {"reason_code": "provider_answer_unavailable_or_ungrounded"},
                    )
                answer = fallback
        else:
            answer = fallback
            if transcript is not None:
                transcript.write(
                    "claude",
                    "assistant_answer_deterministic",
                    {
                        "answer_scope": answer.answer_scope,
                        "citation_count": len(answer.citations),
                        "reason_code": (
                            "scope_policy_deterministic"
                            if self.mode == "functional-uat"
                            else "local_development_deterministic"
                        ),
                    },
                )
        return answer

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
            workflow_output = run_bounded_workflow(
                request,
                self._analyzer,
                transcript=transcript,
            )
            if workflow_output.retrieval_items:
                data_response = self._retriever.retrieve(
                    _retrieve_request(request, workflow_output.retrieval_items)
                )
                evidence_records = _evidence_from_data(data_response)
                workflow_output = workflow_output.with_evidence_status(str(data_response["status"]))
            else:
                # Data rejects an empty retrieval request by contract.  This is
                # an input limitation, not an unhealthy or partial Data store.
                data_response = {"status": "ok", "results": []}
                evidence_records = []
                self._retriever.record_retrieve_skipped(review_run_id=request.review_run_id)
                workflow_output = workflow_output.with_evidence_skipped()
            findings = _dedupe_findings(
                [
                    *workflow_output.findings,
                    *_findings_from_evidence(
                        evidence_records,
                        document_ids_by_item=workflow_output.retrieval_document_ids,
                    ),
                ]
            )
            evidence = [
                EvidenceV3.model_validate(item) for item in _public_evidence(evidence_records)
            ]
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
            else (
                "partial"
                if data_status == "partial" or workflow_output.workflow.status == "partial"
                else "succeeded"
            )
        )
        assessment = (
            "issues_detected"
            if findings
            else (
                "no_issue_detected"
                if data_status == "ok" and workflow_output.workflow.status == "completed"
                else "indeterminate"
            )
        )
        actions = _actions(findings)
        report_sections = build_grounded_report_sections(
            analyses=workflow_output.document_analyses,
            comparisons=workflow_output.comparisons,
            evidence=evidence,
            findings=findings,
            actions=actions,
            evidence_retrieval_skipped=not workflow_output.retrieval_items,
        )
        workflow = workflow_output.with_report_generated().workflow
        payload_result: dict[str, Any] = {
            "schema_version": "3.2",
            "case_id": request.case_id,
            "review_run_id": request.review_run_id,
            "review_as_of": request.review_as_of,
            "trade_type": request.trade_type,
            "jurisdiction": request.jurisdiction,
            "status": status,
            "requires_human_review": True,
            "execution_status": execution_status,
            "assessment": assessment,
            # Bind the result to the exact compact HTTP request, not to
            # Pydantic's materialised defaults for absent child-run fields.
            "input_manifest_sha256": canonical_json_sha256(request.input_manifest_wire()),
            "report": {
                "overall_status": status,
                "summary": (
                    f"문서 {len(request.documents)}건을 제한된 Agentic 워크플로우로 분석했고, "
                    f"법령 근거 {len(evidence)}건과 확인 항목 {len(findings)}건을 기록했습니다."
                ),
                "disclaimer": "이 결과는 기능 확인용이며 담당자의 최종 검토가 필요합니다.",
                "sections": report_sections,
            },
            "workflow": workflow,
            "document_analyses": workflow_output.document_analyses,
            "comparisons": workflow_output.comparisons,
            "evidence": evidence,
            "findings": findings,
            "required_actions": actions,
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
        functional_sections = self._compose_functional_report(result, transcript=transcript)
        if functional_sections != result.report.sections:
            recomposed = result.model_dump(mode="json", warnings=False)
            recomposed["report"]["sections"] = [
                section.model_dump(mode="json", warnings=False) for section in functional_sections
            ]
            recomposed["result_sha256"] = review_result_v3_sha256(recomposed)
            try:
                result = ReviewRunResultV3.model_validate(recomposed)
            except ValidationError:
                # A provider result which passes its local validation but not
                # the complete public result is never exposed; use fallback.
                result = ReviewRunResultV3.model_validate(payload_result)
        if transcript is not None:
            transcript.write(
                "claude",
                "review_report_generated",
                {
                    "section_count": len(result.report.sections),
                    "source": "claude"
                    if functional_sections != report_sections
                    else "deterministic",
                },
            )
        return result

    def close(self) -> None:
        self._closed = True
        close = getattr(self._analyzer, "close", None)
        if callable(close):
            close()
        responder_close = getattr(self._grounded_responder, "close", None)
        if callable(responder_close):
            responder_close()


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
