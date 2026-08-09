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
    KnowledgeAnswerRequestV1,
    KnowledgeAnswerResponseV1,
)
from customs_ai_agent.scenario_transcript import (
    KnowledgeTranscriptRoot,
    KnowledgeTranscriptTurn,
    ScenarioTranscriptError,
    ScenarioTranscriptRoot,
    ScenarioTranscriptRun,
)
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
from customs_ai_agent.service.knowledge_input import (
    KnowledgeInputBoundaryError,
    assert_knowledge_input_is_safe,
    compact_knowledge_turn_text,
)
from customs_ai_agent.service.settings import ServiceSettings
from customs_ai_agent.service.assistant import (
    GuidanceContext,
    assistant_question_policy,
    combine_case_and_guidance,
    deterministic_assistant_answer,
    deterministic_knowledge_answer,
    guidance_answer_for_case,
    guidance_context_from_data,
    guidance_policy_for_registry,
    knowledge_question_policy,
    provider_answer_allowed,
    provider_snapshot,
    response_from_knowledge_provider_candidate,
    response_from_provider_candidate,
    unregistered_document_input_answer,
)
from customs_ai_agent.service.case_input import sanitize_case_assistant_request
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
DataPeer = Literal["evidence", "guidance"]
SERVICE_MODE_HEADER = "X-Customs-Service-Mode"
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_HASH = re.compile(r"^[a-f0-9]{64}$")
_AWB_NUMBER_LABEL_PATTERN = (
    r"(?:a\s*/\s*w\s*b\s*(?:no|number)?|(?<![A-Za-z0-9-])awb(?![A-Za-z0-9-])"
    r"\s*(?:no\.?|number)?|"
    r"air\s*waybill\s*(?:no|number)?)"
)
_GUIDANCE_TOPIC_HINTS = frozenset(
    {
        "customs_clearance",
        "customs_declaration",
        "customs_duty",
        "hs_classification",
        "origin_verification",
        "trade_documents",
        "import_requirements",
        "export_requirements",
        "transport_documents",
        "fta_procedure",
        "trade_terms",
    }
)
_GUIDANCE_SAFE_QUERY = {
    ("customs_clearance", "import_clearance_flow"): "대한민국 수입 통관 흐름 공식 안내",
    ("customs_clearance", "export_clearance_flow"): "대한민국 수출 통관 흐름 공식 안내",
    ("customs_declaration", "import_declaration"): "대한민국 수입 신고 절차 공식 안내",
    ("customs_declaration", "export_declaration"): "대한민국 수출 신고 절차 공식 안내",
    ("customs_duty", "customs_value_basics"): "대한민국 수입 과세가격 기초 공식 안내",
    ("customs_duty", "duty_payment_basics"): "대한민국 수입 관세 납세 기초 공식 안내",
    ("hs_classification", "classification_principles"): "대한민국 HS 품목분류 원칙 공식 안내",
    ("hs_classification", "advance_ruling_or_confirmation"): "대한민국 HS 사전심사 확인 공식 안내",
    ("origin_verification", "origin_evidence"): "대한민국 원산지 증빙 공식 안내",
    ("origin_verification", "origin_verification"): "대한민국 원산지 검증 공식 안내",
    (
        "trade_documents",
        "commercial_and_declaration_documents",
    ): "대한민국 상업 신고 관련 서류 공식 안내",
    (
        "import_requirements",
        "agency_confirmation_and_supporting_documents",
    ): "대한민국 수입 요건 보완서류 공식 안내",
    (
        "export_requirements",
        "agency_confirmation_and_supporting_documents",
    ): "대한민국 수출 요건 보완서류 공식 안내",
    ("transport_documents", "bill_of_lading"): "대한민국 선하증권 B L 공식 안내",
    ("transport_documents", "air_waybill"): "대한민국 항공화물운송장 AWB 공식 안내",
    ("fta_procedure", "origin_certificate"): "대한민국 FTA 원산지증명 공식 안내",
    ("fta_procedure", "preference_procedure"): "대한민국 FTA 특혜 절차 공식 안내",
    ("trade_terms", "cost_risk_delivery_terms"): "대한민국 무역 거래조건 공식 안내",
}

# V1 has no public scope contract.  Keep this explicit rather than falling
# back to the first V2 scope or sending an undeclared V2 field to the
# preserved peer.  Every V2 target has exactly one conservative V1 topic
# projection; user wording never participates in this mapping or in the
# subsequent Data request.
_GUIDANCE_LEGACY_TOPIC_BY_SCOPE = {
    ("customs_clearance", "import_clearance_flow"): "customs_clearance",
    ("customs_clearance", "export_clearance_flow"): "customs_clearance",
    ("customs_declaration", "import_declaration"): "customs_declaration",
    ("customs_declaration", "export_declaration"): "customs_declaration",
    ("customs_duty", "customs_value_basics"): "customs_duty",
    ("customs_duty", "duty_payment_basics"): "customs_duty",
    ("hs_classification", "classification_principles"): "hs_classification",
    ("hs_classification", "advance_ruling_or_confirmation"): "hs_classification",
    ("origin_verification", "origin_evidence"): "origin_verification",
    ("origin_verification", "origin_verification"): "origin_verification",
    (
        "trade_documents",
        "commercial_and_declaration_documents",
    ): "trade_documents",
    (
        "import_requirements",
        "agency_confirmation_and_supporting_documents",
    ): "import_requirements",
    (
        "export_requirements",
        "agency_confirmation_and_supporting_documents",
    ): "export_requirements",
    ("transport_documents", "bill_of_lading"): "transport_documents",
    ("transport_documents", "air_waybill"): "transport_documents",
    ("fta_procedure", "origin_certificate"): "fta_procedure",
    ("fta_procedure", "preference_procedure"): "fta_procedure",
    ("trade_terms", "cost_risk_delivery_terms"): "trade_terms",
}
_GUIDANCE_LEGACY_SAFE_QUERY = {
    "customs_clearance": "대한민국 수출입 통관 절차 기본 공식 안내",
    "customs_declaration": "대한민국 수출입 신고 절차 기본 공식 안내",
    "customs_duty": "대한민국 관세와 과세가격 기본 공식 안내",
    "hs_classification": "대한민국 HS 품목분류 기본 공식 안내",
    "origin_verification": "대한민국 원산지 확인 기본 공식 안내",
    "trade_documents": "대한민국 무역 서류 기본 공식 안내",
    "import_requirements": "대한민국 수입 요건 기본 공식 안내",
    "export_requirements": "대한민국 수출 요건 기본 공식 안내",
    "transport_documents": "대한민국 운송 서류 기본 공식 안내",
    "fta_procedure": "대한민국 FTA 절차 기본 공식 안내",
    "trade_terms": "대한민국 무역 거래조건 기본 공식 안내",
}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


_HTTP = build_opener(_NoRedirect())


def service_mode(settings: ServiceSettings) -> ServiceMode:
    if settings.service_mode in {"local-development", "functional-uat"}:
        return settings.service_mode
    raise RuntimeConfigurationError("service_mode_invalid", "The Agent profile mode is invalid.")


def _service_secret_values(settings: ServiceSettings) -> tuple[str, ...]:
    """Values held by this process that must never survive in a transcript."""

    values = [
        settings.bearer_token.get_secret_value(),
        settings.evidence_service_token.get_secret_value(),
    ]
    if settings.guidance_service_token is not None:
        values.append(settings.guidance_service_token.get_secret_value())
    return tuple(values)


def _transcript_root(
    settings: ServiceSettings, supplied: ScenarioTranscriptRoot | None = None
) -> ScenarioTranscriptRoot:
    """Functional UAT always records locally; development remains explicit opt-in."""

    mode = service_mode(settings)
    root = supplied or ScenarioTranscriptRoot.for_service_mode(
        mode, sensitive_values=_service_secret_values(settings)
    )
    return root.with_sensitive_values(_service_secret_values(settings))


def _knowledge_transcript_root(
    settings: ServiceSettings, supplied: KnowledgeTranscriptRoot | None = None
) -> KnowledgeTranscriptRoot:
    """Select the isolated standalone-guidance transcript boundary.

    It intentionally has a distinct environment switch and destination from
    review scenarios: a knowledge conversation must never become a Case or
    document transcript merely because review recording was enabled.
    """

    mode = service_mode(settings)
    root = supplied or KnowledgeTranscriptRoot.for_service_mode(
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
    """Minimal Data HTTP client with closed review and guidance operations.

    Document-review readiness and retrieval always use the Evidence peer.
    Guidance uses its own pair only when the complete optional guidance peer
    is configured; otherwise it intentionally uses the existing Evidence
    origin and token.  Both peers are independently loopback-validated and
    must return the Agent's exact service-mode header.
    """

    readiness_path = "/internal/rag/v3/health/ready"
    retrieve_path = "/internal/rag/v3/retrieve"
    guidance_query_path = "/internal/rag/v3/guidance/query"

    def __init__(
        self, settings: ServiceSettings, *, transcript_root: ScenarioTranscriptRoot | None = None
    ) -> None:
        self.mode = service_mode(settings)
        self._timeout = settings.readiness_timeout_seconds
        self._transcript_root = _transcript_root(settings, transcript_root)
        evidence_token = settings.evidence_service_token
        guidance_token = settings.guidance_service_token or evidence_token
        guidance_base_url = settings.guidance_base_url or settings.evidence_base_url
        self._evidence_base_url = self._validated_peer_url(
            settings.evidence_base_url, peer="evidence"
        )
        self._guidance_base_url = self._validated_peer_url(guidance_base_url, peer="guidance")
        self._evidence_token = evidence_token.get_secret_value()
        self._guidance_token = guidance_token.get_secret_value()
        self._guidance_registry_version = settings.guidance_registry_version
        self._guidance_v2_ready = False

    @property
    def guidance_registry_version(self) -> Literal["1.0", "2.0"] | None:
        """Configured guidance generation, if this runtime has one.

        ``None`` remains a local-development compatibility mode only.  A
        Functional-UAT caller must select a versioned composed profile before
        the Agent is allowed to query general guidance.
        """

        return self._guidance_registry_version

    @staticmethod
    def _validated_peer_url(base_url: str, *, peer: DataPeer) -> str:
        try:
            normalized = validate_evidence_destination(base_url)
            host = urlsplit(normalized).hostname
            if host is None or not ipaddress.ip_address(host).is_loopback:
                raise ValueError
        except (TypeError, ValueError):
            raise RuntimeConfigurationError(
                f"{peer}_base_url_invalid", "The Data service address is invalid."
            ) from None
        return normalized

    def _peer_credentials(self, peer: DataPeer) -> tuple[str, str]:
        if peer == "evidence":
            return self._evidence_base_url, self._evidence_token
        if peer == "guidance":
            return self._guidance_base_url, self._guidance_token
        raise RetrievalError("Data service peer is invalid")

    @staticmethod
    def _failure_code(peer: DataPeer, condition: str) -> str:
        return f"{'data' if peer == 'evidence' else 'guidance'}_{condition}"

    def _headers(
        self, *, peer: DataPeer, json_body: bool, request_id: str | None = None
    ) -> dict[str, str]:
        _base_url, token = self._peer_credentials(peer)
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
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
        transcript: ScenarioTranscriptRun | None = None,
        peer: DataPeer = "evidence",
    ) -> dict[str, Any]:
        if review_run_id is not None and transcript is not None:
            raise RetrievalError("Data transcript binding is invalid")
        base_url, _token = self._peer_credentials(peer)
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
            f"{base_url}{path}",
            data=encoded,
            headers=self._headers(
                peer=peer,
                json_body=encoded is not None,
                request_id=(payload or {}).get("request_id"),
            ),
            method=method,
        )
        try:
            with _HTTP.open(request, timeout=self._timeout) as response:
                if _one_header(response, SERVICE_MODE_HEADER) != self.mode:
                    self._record_failure(
                        review_run_id,
                        transcript=transcript,
                        method=method,
                        path=path,
                        error_code=self._failure_code(peer, "mode_mismatch"),
                    )
                    raise RetrievalError("Data service mode mismatch")
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except RetrievalError:
            raise
        except (HTTPError, URLError, TimeoutError, OSError):
            self._record_failure(
                review_run_id,
                transcript=transcript,
                method=method,
                path=path,
                error_code=self._failure_code(peer, "request_failed"),
            )
            raise RetrievalError("Data service request failed") from None
        if not isinstance(raw, bytes) or len(raw) > _MAX_RESPONSE_BYTES:
            self._record_failure(
                review_run_id,
                transcript=transcript,
                method=method,
                path=path,
                error_code=self._failure_code(peer, "response_invalid"),
            )
            raise RetrievalError("Data service response is invalid")
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            self._record_failure(
                review_run_id,
                transcript=transcript,
                method=method,
                path=path,
                error_code=self._failure_code(peer, "response_invalid"),
            )
            raise RetrievalError("Data service response is invalid") from None
        if not isinstance(decoded, dict):
            self._record_failure(
                review_run_id,
                transcript=transcript,
                method=method,
                path=path,
                error_code=self._failure_code(peer, "response_invalid"),
            )
            raise RetrievalError("Data service response is invalid")
        self._record_exchange(
            review_run_id,
            transcript=transcript,
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
        transcript: ScenarioTranscriptRun | None = None,
        method: Literal["GET", "POST"],
        path: str,
        request_body: dict[str, Any] | None,
        response_body: dict[str, Any],
    ) -> None:
        active_transcript = transcript
        if active_transcript is None and review_run_id is not None:
            active_transcript = self._transcript_root.for_run(review_run_id)
        if active_transcript is not None:
            active_transcript.write(
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
        transcript: ScenarioTranscriptRun | None = None,
        method: Literal["GET", "POST"],
        path: str,
        error_code: str,
    ) -> None:
        active_transcript = transcript
        if active_transcript is None and review_run_id is not None:
            active_transcript = self._transcript_root.for_run(review_run_id)
        if active_transcript is not None:
            # Do not retain a provider error body or the original request on a
            # failed exchange.  The safe code is enough to diagnose the edge.
            active_transcript.write(
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

    def _ensure_v2_guidance_ready(
        self,
        *,
        review_run_id: str | None = None,
        transcript: ScenarioTranscriptRun | None = None,
    ) -> None:
        """Check the active V2 pointer once before its first bounded query.

        The V2 Data service makes this readiness endpoint fail closed when the
        active snapshot, registry, or source verification is not ready.  The
        subsequent query still undergoes full response provenance validation.
        """

        if self._guidance_registry_version != "2.0" or self._guidance_v2_ready:
            return
        response = self._request(
            self.readiness_path,
            method="GET",
            review_run_id=review_run_id,
            transcript=transcript,
            peer="guidance",
        )
        if response != {"status": "ready"}:
            self._record_failure(
                review_run_id,
                transcript=transcript,
                method="GET",
                path=self.readiness_path,
                error_code="guidance_v2_readiness_invalid",
            )
            raise RetrievalError("V2 guidance snapshot is not ready")
        self._guidance_v2_ready = True

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

    def guidance_query(
        self,
        request: dict[str, Any],
        *,
        review_run_id: str | None = None,
        transcript: ScenarioTranscriptRun | None = None,
    ) -> dict[str, Any]:
        """Make the one permitted official-guidance query for an assistant turn.

        It deliberately does not call the item/HS review endpoint.  A V2
        profile performs one cached readiness check before its first bounded
        query; V1 and local dev-fixture calls retain the single-query path.
        """

        topic_hint = request.get("topic_hint")
        configured_registry = self._guidance_registry_version
        if configured_registry == "1.0":
            request_shape_is_valid = (
                "scope_hint" not in request
                and topic_hint in _GUIDANCE_TOPIC_HINTS
                and _GUIDANCE_LEGACY_SAFE_QUERY.get(str(topic_hint)) == request.get("query")
            )
        else:
            # Local development intentionally follows the V2 pair contract,
            # because its deterministic fixture implements the V2 registry.
            # A Functional-UAT V2 peer receives the same exact pair.
            scope_hint = request.get("scope_hint")
            request_shape_is_valid = isinstance(scope_hint, str) and _GUIDANCE_SAFE_QUERY.get(
                (str(topic_hint), scope_hint)
            ) == request.get("query")
        if (
            request.get("schema_version") != "3.0"
            or not isinstance(request.get("request_id"), str)
            or not isinstance(request.get("query"), str)
            or not 1 <= len(request["query"]) <= 1_200
            or request.get("trade_direction") not in {"import", "export", "both"}
            or topic_hint not in _GUIDANCE_TOPIC_HINTS
            or not request_shape_is_valid
            or not isinstance(request.get("as_of"), str)
            or request.get("top_k") not in {1, 2, 3}
        ):
            raise RetrievalError("Data guidance request is invalid")
        self._ensure_v2_guidance_ready(review_run_id=review_run_id, transcript=transcript)
        response = self._request(
            self.guidance_query_path,
            method="POST",
            payload=request,
            review_run_id=review_run_id,
            transcript=transcript,
            peer="guidance",
        )
        if (
            response.get("schema_version") != "3.0"
            or response.get("request_id") != request["request_id"]
            or response.get("status") not in {"covered", "no_coverage", "partial", "failed"}
            or response.get("coverage") not in {"covered", "no_coverage", "partial", "unavailable"}
            or not isinstance(response.get("results"), list)
        ):
            self._record_failure(
                review_run_id,
                transcript=transcript,
                method="POST",
                path=self.guidance_query_path,
                error_code="guidance_contract_invalid",
            )
            raise RetrievalError("Data guidance response does not satisfy the shared contract")
        if (
            self._guidance_registry_version is not None
            and response.get("registry_version") != self._guidance_registry_version
        ):
            self._record_failure(
                review_run_id,
                transcript=transcript,
                method="POST",
                path=self.guidance_query_path,
                error_code="guidance_registry_version_mismatch",
            )
            raise RetrievalError("Data guidance registry generation does not match the profile")
        return response

    def record_retrieve_skipped(
        self,
        *,
        review_run_id: str,
        reason_code: Literal[
            "no_hs_qualified_commodity", "export_destination_country_missing"
        ] = "no_hs_qualified_commodity",
    ) -> None:
        """Record an input-side skip without making a Data request.

        Functional-UAT transcripts, and an explicitly enabled development
        scenario transcript, require an Agent-to-Data stream for every review.
        An empty commodity plan is deliberately not sent to Data because the
        shared retrieve contract rejects it. This event keeps that distinction
        visible without treating the Data service as unavailable or retaining
        document values. Ordinary development has no transcript root, so this
        remains a no-op there.
        """

        transcript = self._transcript_root.for_run(review_run_id)
        if transcript is not None:
            transcript.write(
                "agent-data",
                "retrieve_skipped",
                {
                    "reason_code": reason_code,
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
    Literal[
        "invoice",
        "packing_list",
        "bill_of_lading",
        "air_waybill",
        "certificate_of_origin",
        "export_declaration",
        "supporting_evidence",
        "unknown",
    ],
    str,
]:
    rules: tuple[tuple[str, str], ...] = (
        (
            "air_waybill",
            r"(?:\b(?:air\s*waybill|airway\s*bill|a\s*/\s*w\s*b)\b|항공\s*화물\s*운송장)",
        ),
        (
            "bill_of_lading",
            r"(?:\b(?:bill\s+of\s+lading|b\s*/\s*l)\b|선하\s*증권)",
        ),
        (
            "packing_list",
            r"(?:\b(?:packing\s+list|packing\s+note)\b|포장\s*명세서)",
        ),
        (
            "certificate_of_origin",
            r"(?:\b(?:certificate\s+of\s+origin|c\s*/\s*o)\b|원산지\s*증명서)",
        ),
        (
            "export_declaration",
            r"(?:\b(?:export\s+declaration|export\s+clearance|export\s+declaration\s+certificate)\b|수출\s*신고(?:필증|증|서)?)",
        ),
        (
            "supporting_evidence",
            r"(?:\b(?:supporting\s+(?:evidence|document)|submission\s+(?:receipt|evidence)|filing\s+(?:receipt|evidence))\b|(?:보완|제출)\s*(?:증빙|서류|확인서))",
        ),
        ("invoice", r"(?:\b(?:commercial\s+invoice|invoice)\b|상업\s*송장|인보이스)"),
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
                "air_waybill",
                "certificate_of_origin",
                "export_declaration",
                "supporting_evidence",
                "unknown",
            ],
            document_type,
        ), quote
    return "unknown", ""


def _deterministic_document_fields(text: str, document_type: str) -> list[ClaudeFieldCandidate]:
    rules: dict[str, tuple[tuple[str, str], ...]] = {
        "invoice": (
            (
                "invoice_no",
                r"(?:invoice\s*(?:no|number)|inv\s*no|(?:상업\s*)?송장\s*번호|인보이스\s*번호)\s*[:#-]\s*([^\n|;]+)",
            ),
            (
                "total_amount",
                r"(?:invoice\s+total|total\s+amount|grand\s+total|총액|합계\s*금액)\s*[:#-]\s*([^\n|;]+)",
            ),
            (
                "country_of_origin",
                r"(?:country\s+of\s+origin|origin\s+country|원산지)\s*[:#-]\s*([^\n|;]+)",
            ),
            ("hs_code", r"\bhs\s*(?:code)?\s*[:#-]\s*([0-9]{6,10})\b"),
        ),
        "packing_list": (
            (
                "packing_no",
                r"(?:packing\s*(?:no|number)|pl\s*no|포장\s*명세서\s*번호)\s*[:#-]\s*([^\n|;]+)",
            ),
            (
                "invoice_no",
                r"(?:invoice\s*(?:no|number)|inv\s*no|(?:상업\s*)?송장\s*번호|인보이스\s*번호)\s*[:#-]\s*([^\n|;]+)",
            ),
            (
                "bl_no",
                r"(?:b\s*/\s*l\s*(?:no|number)?|bill\s+of\s+lading\s*(?:no|number)?)\s*[:#-]\s*([^\n|;]+)",
            ),
            (
                "awb_no",
                _AWB_NUMBER_LABEL_PATTERN + r"\s*[:#-]\s*([^\n|;]+)",
            ),
            ("container_numbers", r"(?:container\s*(?:no|number|numbers)?)\s*[:#-]\s*([^\n|;]+)"),
            (
                "total_gross_weight_kg",
                r"(?:total\s+gross\s+weight|gross\s+weight)\s*[:#-]\s*([^\n|;]+)",
            ),
            (
                "country_of_origin",
                r"(?:country\s+of\s+origin|origin\s+country|원산지)\s*[:#-]\s*([^\n|;]+)",
            ),
        ),
        "certificate_of_origin": (
            (
                "certificate_no",
                r"(?:certificate\s*(?:no|number)|c\s*/\s*o\s*(?:no|number)?|원산지\s*증명서\s*번호)\s*[:#-]\s*([^\n|;]+)",
            ),
            (
                "origin_country",
                r"(?:country\s+of\s+origin|origin\s+country|origin|원산지)\s*[:#-]\s*([^\n|;]+)",
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
        "air_waybill": (
            (
                "awb_no",
                _AWB_NUMBER_LABEL_PATTERN + r"\s*[:#-]\s*([^\n|;]+)",
            ),
            ("gross_weight_kg", r"(?:gross\s+weight|total\s+gross\s+weight)\s*[:#-]\s*([^\n|;]+)"),
        ),
        "export_declaration": (
            (
                "export_declaration_no",
                r"(?:export\s+declaration\s*(?:no|number)?|declaration\s*(?:no|number)?|수출\s*신고(?:필증)?\s*번호)\s*[:#-]\s*([^\n|;]+)",
            ),
            (
                "filing_date",
                r"(?:filing\s+date|declaration\s+date|submitted\s+on|신고\s*일(?:자)?|제출\s*일(?:자)?)\s*[:#-]\s*([^\n|;]+)",
            ),
            (
                "submission_status",
                r"(?:submission\s+status|declaration\s+status|acceptance\s+status|처리\s*상태|수리\s*상태)\s*[:#-]\s*([^\n|;]+)",
            ),
            (
                "invoice_no",
                r"(?:invoice\s*(?:no|number)|inv\s*no|(?:상업\s*)?송장\s*번호|인보이스\s*번호)\s*[:#-]\s*([^\n|;]+)",
            ),
            (
                "total_amount",
                r"(?:total\s+amount|declared\s+amount|invoice\s+total|총액|신고\s*금액)\s*[:#-]\s*([^\n|;]+)",
            ),
            (
                "origin_country",
                r"(?:country\s+of\s+origin|origin\s+country|origin|원산지)\s*[:#-]\s*([^\n|;]+)",
            ),
            (
                "destination_country",
                r"(?:destination\s+country|country\s+of\s+destination|destination|목적\s*국(?:가)?)\s*[:#-]\s*([^\n|;]+)",
            ),
            ("hs_code", r"\bhs\s*(?:code)?\s*[:#-]\s*([0-9]{6,10})\b"),
        ),
        "supporting_evidence": (
            (
                "submission_receipt_no",
                r"(?:receipt\s*(?:no|number)?|filing\s*(?:no|number)?|submission\s*(?:no|number)?|접수\s*번호|제출\s*번호)\s*[:#-]\s*([^\n|;]+)",
            ),
            (
                "filing_date",
                r"(?:filing\s+date|submission\s+date|submitted\s+on|제출\s*일(?:자)?|접수\s*일(?:자)?)\s*[:#-]\s*([^\n|;]+)",
            ),
            (
                "submission_status",
                r"(?:submission\s+status|acceptance\s+status|decision\s+status|처리\s*상태|수리\s*상태)\s*[:#-]\s*([^\n|;]+)",
            ),
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
        r"(?:\b(?:description|goods|item\s*\d+|qty|quantity|unit\s+price|hs\s*code)\b|품명|수량|단가|금액|HS\s*코드)",
        quote,
        re.IGNORECASE,
    ):
        return None
    field_rules: tuple[tuple[str, str], ...] = (
        ("description", r"(?:description|goods|item\s*\d+|품명)\s*[:#-]\s*([^|;]+)"),
        ("hs_code", r"\bhs\s*(?:code)?\s*[:#-]\s*([0-9]{6,10})\b"),
        ("quantity", r"(?:quantity|qty|수량)\s*[:#-]\s*([^|;]+)"),
        ("unit", r"(?:\bunit\b(?!\s*price)|단위)\s*[:#-]\s*([^|;]+)"),
        ("unit_price", r"(?:unit\s+price|price|단가)\s*[:#-]\s*([^|;]+)"),
        ("amount", r"(?:line\s+amount|amount|금액)\s*[:#-]\s*([^|;]+)"),
        (
            "origin_country",
            r"(?:origin\s+country|country\s+of\s+origin|origin|원산지)\s*[:#-]\s*([^|;]+)",
        ),
        ("gross_weight_kg", r"(?:gross\s+weight|총중량)\s*[:#-]\s*([^|;]+)"),
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


def _guidance_query_request(
    *,
    conversation_id: str,
    trade_type: Literal["import", "export", "both"],
    as_of: str,
    topic_hint: str,
    scope_hint: str,
    registry_version: Literal["1.0", "2.0"],
) -> dict[str, Any]:
    """Create one versioned lookup without forwarding user prose.

    Data's versioned registry already receives the bounded ``topic_hint``.
    The required query field is therefore a fixed Korean retrieval label, not
    a second copy of a user's question or conversation history.  This keeps
    normal user prose out of the Agent→Data transcript while preserving the
    shared Data wire contract and deterministic request identity.  V1 has no
    scope field, so its request uses only the explicit legacy topic.  The
    Agent rejects a V2-only field before it can reach that peer.
    """

    target = (topic_hint, scope_hint)
    if topic_hint not in _GUIDANCE_TOPIC_HINTS:
        raise RuntimeInputError("guidance_topic_invalid", "guidance topic is invalid")
    if registry_version == "1.0":
        legacy_topic = _GUIDANCE_LEGACY_TOPIC_BY_SCOPE.get(target)
        safe_query = (
            _GUIDANCE_LEGACY_SAFE_QUERY.get(legacy_topic) if legacy_topic is not None else None
        )
        if safe_query is None:
            raise RuntimeInputError("guidance_topic_invalid", "guidance topic is invalid")
        digest = hashlib.sha256(
            f"{conversation_id}\n{safe_query}\n{trade_type}\n{legacy_topic}\nlegacy_topic\n{as_of}".encode(
                "utf-8"
            )
        ).hexdigest()[:32]
        return {
            "schema_version": "3.0",
            "request_id": f"guide-{digest}",
            "query": safe_query,
            "trade_direction": trade_type,
            "topic_hint": legacy_topic,
            "as_of": as_of,
            "top_k": 3,
        }
    if registry_version != "2.0":
        raise RuntimeInputError("guidance_registry_invalid", "guidance registry is invalid")
    safe_query = _GUIDANCE_SAFE_QUERY.get(target)
    if safe_query is None:
        raise RuntimeInputError("guidance_topic_invalid", "guidance topic is invalid")
    digest = hashlib.sha256(
        f"{conversation_id}\n{safe_query}\n{trade_type}\n{topic_hint}\n{scope_hint}\n2.0\n{as_of}".encode(
            "utf-8"
        )
    ).hexdigest()[:32]
    return {
        "schema_version": "3.0",
        "request_id": f"guide-{digest}",
        "query": safe_query,
        "trade_direction": trade_type,
        "topic_hint": topic_hint,
        "scope_hint": scope_hint,
        "as_of": as_of,
        "top_k": 3,
    }


def _guidance_provider_snapshot(
    context: GuidanceContext,
    *,
    trade_direction: str,
) -> dict[str, Any]:
    """Pass only bounded presentation data to the standalone Claude adapter.

    In particular this payload deliberately has no user question or recent
    conversation text.  Claude may select a display enum and an order among
    the Agent-assigned public-source slots; Agent code owns every visible
    Korean sentence, citation, and suggested question.
    """

    return {
        "topic": context.topic,
        "scope": context.scope,
        "trade_direction": trade_direction,
        "snapshot_id": context.snapshot_id,
        "as_of": context.as_of.isoformat(),
        "coverage": context.coverage,
        "registry_version": context.registry_version,
        "allowed_presentation_styles": ["overview", "checklist", "source_focus"],
        "allowed_suggestion_topics": sorted(_GUIDANCE_TOPIC_HINTS),
        "source_slots": [
            {
                "slot": index,
                "title": item.title,
                "topic": item.topic,
                "direction": item.direction,
                "excerpt": item.excerpt,
                "law_name": item.law_name,
                "article_no": item.article_no,
            }
            for index, item in enumerate(context.evidence)
        ],
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
        self,
        settings: ServiceSettings,
        *,
        transcript_root: ScenarioTranscriptRoot | None = None,
        knowledge_transcript_root: KnowledgeTranscriptRoot | None = None,
    ) -> None:
        self.mode = service_mode(settings)
        self._closed = False
        self._transcript_root = _transcript_root(settings, transcript_root)
        self._knowledge_transcript_root = _knowledge_transcript_root(
            settings, knowledge_transcript_root
        )
        if self.mode == "local-development":
            self._analyzer: Any = _DeterministicDocumentAnalyzer()
            self._grounded_responder: ClaudeGroundedResponder | None = None
        else:
            try:
                claude_settings = ClaudeSettings.from_environment()
                self._transcript_root = self._transcript_root.with_sensitive_values(
                    (claude_settings.api_key.get_secret_value(),)
                )
                self._knowledge_transcript_root = (
                    self._knowledge_transcript_root.with_sensitive_values(
                        (claude_settings.api_key.get_secret_value(),)
                    )
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

    @staticmethod
    def _decode_knowledge(payload: bytes) -> KnowledgeAnswerRequestV1:
        try:
            parsed = KnowledgeAnswerRequestV1.model_validate(json.loads(payload.decode("utf-8")))
        except (UnicodeDecodeError, ValidationError, ValueError):
            raise RuntimeInputError(
                "knowledge_request_invalid",
                "The knowledge assistant request does not satisfy the service contract.",
            ) from None
        try:
            assert_knowledge_input_is_safe(parsed.question, parsed.recent_turns)
        except KnowledgeInputBoundaryError:
            # This guard also protects direct runtime callers that do not use
            # the HTTP admission layer.  It runs before a transcript is
            # allocated or either Data/Claude capability is considered.
            raise RuntimeInputError(
                "knowledge_input_boundary",
                "The knowledge request crosses the standalone assistant boundary.",
            ) from None
        if _SAFE_RUN_ID.fullmatch(parsed.conversation_id) is None:
            raise RuntimeInputError(
                "knowledge_request_invalid", "The knowledge assistant request is invalid."
            )
        return parsed.model_copy(
            update={
                "question": parsed.question.strip(),
                "recent_turns": [
                    turn.model_copy(update={"content": compact_knowledge_turn_text(turn.content)})
                    for turn in parsed.recent_turns
                ],
            }
        )

    def _engine(self) -> EngineV3:
        return EngineV3(
            kind="deterministic" if self.mode == "local-development" else "claude",
            model_id=str(self._analyzer.model_id),
        )

    def _guidance_lookup_is_enabled(self) -> bool:
        """Keep Functional-UAT generations explicit at the Agent boundary."""

        return (
            self.mode == "local-development"
            or self._retriever.guidance_registry_version is not None
        )

    def _guidance_request_registry_version(self) -> Literal["1.0", "2.0"]:
        """Select the request generation from the explicitly bound profile.

        Local development has one deterministic V2 fixture contract.  A
        Functional-UAT profile must be explicit; its selected registry
        generation controls whether ``scope_hint`` is present on the wire.
        """

        return self._retriever.guidance_registry_version or "2.0"

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

    def _knowledge_answer(
        self,
        request: KnowledgeAnswerRequestV1,
        *,
        context: Any,
        transcript: ScenarioTranscriptRun | None = None,
        registry_version: Literal["1.0", "2.0"] = "2.0",
    ) -> KnowledgeAnswerResponseV1:
        """Compose one source-bounded guidance answer, with a safe fallback."""

        fallback = deterministic_knowledge_answer(
            request,
            engine=self._engine(),
            context=context,
            registry_version=registry_version,
        )
        responder = self._grounded_responder
        provider_allowed = (
            fallback.answer_scope == "general_customs_guidance"
            and context is not None
            and bool(context.evidence)
        )
        if self.mode != "functional-uat" or responder is None or not provider_allowed:
            # The standalone Functional-UAT verifier requires the two
            # Agent-owned streams even when coverage is unavailable or the
            # optional wording adapter cannot run.  Record a safe non-call
            # event instead of leaving an ambiguous missing ``claude`` file.
            if transcript is not None and self.mode == "functional-uat":
                transcript.write(
                    "claude",
                    "knowledge_answer_not_called",
                    {
                        "provider": "claude",
                        "called": False,
                        "reason_code": (
                            "official_guidance_coverage_unavailable"
                            if not provider_allowed
                            else "claude_responder_unavailable"
                        ),
                    },
                )
            return fallback
        try:
            candidate = responder.respond(
                {
                    "guidance_presentation": _guidance_provider_snapshot(
                        context,
                        trade_direction=request.trade_type,
                    ),
                },
                transcript=transcript,
                purpose="knowledge_answer",
            )
            return response_from_knowledge_provider_candidate(
                candidate, request, engine=self._engine(), context=context
            )
        except ScenarioTranscriptError:
            # Functional-UAT transcript writing is part of the evidence
            # boundary.  A rejected or unwritable local record cannot quietly
            # downgrade to a successful answer.
            raise
        except Exception:
            if transcript is not None:
                transcript.write(
                    "claude",
                    "knowledge_answer_fallback",
                    {"reason_code": "provider_answer_unavailable_or_ungrounded"},
                )
            return fallback

    def assistant_answer_json(self, payload: bytes) -> AssistantAnswerResponseV1:
        """Answer once from an immutable review snapshot; never mutate Case state."""

        self._assert_open()
        sanitization = sanitize_case_assistant_request(self._decode_assistant(payload))
        request = sanitization.request
        if sanitization.current_input_was_document_shaped:
            # Do not allocate or write a review transcript for the pasted
            # body.  The safe response below is fully code-owned and does not
            # invoke Data or Claude.
            return unregistered_document_input_answer(request, engine=self._engine())
        transcript = self._transcript_root.for_run(request.review_run_id)
        registry_version = self._guidance_request_registry_version()
        policy = guidance_policy_for_registry(
            assistant_question_policy(request), registry_version=registry_version
        )
        guidance_context = None
        if (
            policy.general_topic is not None
            and policy.general_scope is not None
            and policy.query_direction is not None
            and not policy.direction_inapplicable
            and self._guidance_lookup_is_enabled()
        ):
            try:
                data_response = self._retriever.guidance_query(
                    _guidance_query_request(
                        conversation_id=request.conversation_id,
                        trade_type=policy.query_direction,
                        as_of=request.review_result.review_as_of.isoformat(),
                        topic_hint=policy.general_topic,
                        scope_hint=policy.general_scope,
                        registry_version=registry_version,
                    ),
                    review_run_id=request.review_run_id,
                )
                guidance_context = guidance_context_from_data(
                    data_response,
                    profile_mode=self.mode,
                    expected_as_of=request.review_result.review_as_of,
                    expected_trade_direction=policy.query_direction,
                    expected_topic=policy.general_topic,
                    expected_scope=policy.general_scope,
                )
            except (RetrievalError, ValueError):
                # A guidance lookup may not turn an otherwise usable Case
                # conversation into a dependency error.  The answer below
                # explicitly reports unavailable coverage without guessing.
                guidance_context = None
        if (
            policy.general_topic is not None
            and not policy.mixed
            and policy.scope in {"general_customs_guidance", "insufficient_evidence"}
        ):
            knowledge_request = KnowledgeAnswerRequestV1(
                conversation_id=request.conversation_id,
                question=request.question,
                trade_type=request.review_result.trade_type,
                as_of=request.review_result.review_as_of,
                recent_turns=request.recent_turns,
            )
            knowledge_answer = self._knowledge_answer(
                knowledge_request,
                context=guidance_context,
                transcript=transcript,
                registry_version=registry_version,
            )
            answer = guidance_answer_for_case(
                request,
                engine=self._engine(),
                context=guidance_context,
                standalone_answer=knowledge_answer,
            )
            if transcript is not None:
                transcript.write(
                    "claude",
                    "assistant_answer_deterministic",
                    {
                        "answer_scope": answer.answer_scope,
                        "citation_count": len(answer.citations),
                        "guidance_evidence_count": sum(
                            getattr(citation, "kind", None) == "guidance_evidence"
                            for citation in answer.citations
                        ),
                        "reason_code": (
                            "official_guidance_provider"
                            if knowledge_answer.engine.kind == "claude"
                            and knowledge_answer.direct_answer
                            != deterministic_knowledge_answer(
                                knowledge_request,
                                engine=self._engine(),
                                context=guidance_context,
                                registry_version=registry_version,
                            ).direct_answer
                            else "official_guidance_deterministic"
                        ),
                    },
                )
            return answer
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
        if policy.mixed:
            answer = combine_case_and_guidance(
                answer,
                context=guidance_context,
                policy=policy,
                as_of=request.review_result.review_as_of,
                registry_version=registry_version,
            )
        return answer

    def knowledge_answer_json(self, payload: bytes) -> KnowledgeAnswerResponseV1:
        """Answer one standalone, official-source-grounded knowledge question.

        No Case, Review Run, document text, Evidence IDs, or human-correction
        state crosses this method's boundary.  The Data call is optional only
        for a classified official topic and occurs at most once per turn.
        """

        self._assert_open()
        request = self._decode_knowledge(payload)
        registry_version = self._guidance_request_registry_version()
        policy = guidance_policy_for_registry(
            knowledge_question_policy(request.question.strip(), trade_type=request.trade_type),
            registry_version=registry_version,
        )
        guidance_context = None
        # A standalone transcript is allocated only for a permitted general
        # guidance lookup.  Ordinary development leaves this root disabled;
        # Functional-UAT uses the reserved scenario-runs/knowledge namespace;
        # an explicit dev-fixture may use that safe namespace or preserved
        # legacy knowledge-runs history, never a review's Case/Review tree.
        transcript: KnowledgeTranscriptTurn | None = None
        if (
            policy.general_topic is not None
            and policy.general_scope is not None
            and policy.query_direction is not None
            and not policy.direction_inapplicable
            and self._guidance_lookup_is_enabled()
        ):
            transcript = self._knowledge_transcript_root.start_turn(request.conversation_id)
            try:
                data_response = self._retriever.guidance_query(
                    _guidance_query_request(
                        conversation_id=request.conversation_id,
                        trade_type=policy.query_direction,
                        as_of=request.as_of.isoformat(),
                        topic_hint=policy.general_topic,
                        scope_hint=policy.general_scope,
                        registry_version=registry_version,
                    ),
                    transcript=transcript,
                )
                guidance_context = guidance_context_from_data(
                    data_response,
                    profile_mode=self.mode,
                    expected_as_of=request.as_of,
                    expected_trade_direction=policy.query_direction,
                    expected_topic=policy.general_topic,
                    expected_scope=policy.general_scope,
                )
            except (RetrievalError, ValueError):
                guidance_context = None
        return self._knowledge_answer(
            request,
            context=guidance_context,
            transcript=transcript,
            registry_version=registry_version,
        )

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
                self._retriever.record_retrieve_skipped(
                    review_run_id=request.review_run_id,
                    reason_code=(
                        workflow_output.retrieval_skip_reason or "no_hs_qualified_commodity"
                    ),
                )
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
            evidence_retrieval_skip_reason=workflow_output.retrieval_skip_reason,
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
