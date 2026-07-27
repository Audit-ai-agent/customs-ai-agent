from __future__ import annotations

import copy
import json
import math
import re
import time
from dataclasses import dataclass, field
from os import PathLike
from collections.abc import Callable
from typing import Any, Literal

from jsonschema import Draft202012Validator, FormatChecker
from pydantic import SecretStr, ValidationError

from customs_ai_agent.adapters.llm import (
    ClaudeDocumentAnalyzer,
    ClaudeSettings,
    LLMError,
    analysis_to_extracted_document,
)
from customs_ai_agent.adapters.ocr import OcrLayoutProvider, OcrLayoutProviderError
from customs_ai_agent.adapters.retrieval import (
    ProductionHttpRegulationRetriever,
    RetrievalError,
    validate_evidence_destination,
)
from customs_ai_agent.adapters.retrieval_v3 import ProductionHttpRegulationRetrieverV3
from customs_ai_agent.assertion_trust import (
    AssertionTrustStoreV1,
    SupportingDocumentAssertionVerifier,
)
from customs_ai_agent.build_info import package_code_snapshot_sha256
from customs_ai_agent.contracts import get_schema
from customs_ai_agent.contracts.evidence_runtime_release import (
    EvidenceRuntimeReleaseLockError,
    EvidenceRuntimeReleaseUnreleasedError,
    load_released_evidence_runtime_identity,
)
from customs_ai_agent.ocr_release import ReleaseLockedOcrProvider
from customs_ai_agent.runner import (
    CorrectionApplicationError,
    WorkflowStageObserver,
    WorkflowStageState,
    rerun_review,
    run_review_result_v2,
    run_review_v2,
    to_review_run_result_v2,
)
from customs_ai_agent.agentic import (
    AGENT_POLICY_ID,
    ActiveBoundedRuntimeOutcome,
    BoundedToolBroker,
    BoundedToolResult,
    DeterministicReevaluationResult,
    TrustedAssertionVerifier,
    answer_grounded_question,
    automatic_completion_unmet_reason_codes,
    assert_result_hash,
    execute_active_bounded_plan,
    presented_requirement_codes_by_item,
    runtime_release_identity,
    to_review_run_result_v3,
    verified_requirement_codes_by_item,
)
from customs_ai_agent.nodes.compliance import compliance_review
from customs_ai_agent.nodes.normalize import normalize_documents
from customs_ai_agent.nodes.pair import pair_documents
from customs_ai_agent.nodes.report import generate_xai_report
from customs_ai_agent.nodes.retrieve import retrieve_regulations
from customs_ai_agent.nodes.route import route_to_human_or_complete
from customs_ai_agent.nodes.validate import validate_cross_documents
from customs_ai_agent.policies.outcome import extraction_completion_blockers
from customs_ai_agent.schemas.models import (
    ReviewRunRequestV2,
    ReviewRunResultV2,
    ServiceReviewRunRequestV2,
    UserCorrection,
)
from customs_ai_agent.schemas.v3 import (
    GroundedAnswerV1,
    GroundedQuestionV1,
    FactLedgerV1,
    PlannedToolActionV1,
    ReviewPlanProposalV1,
    ReviewRevisionRequestV1,
    ReviewRevisionResultV1,
    ReviewRunResultV3,
    ServiceReviewRunRequestV3,
    review_result_v3_sha256,
    stable_object_id,
)
from customs_ai_agent.schemas.experience import (
    AgentUatCapabilityAttestationV1,
    AssistantAnswerV2,
    CaseConversationRequestV2,
    ReviewExperienceEnvelopeV1,
)
from customs_ai_agent.experience import answer_case_conversation, build_review_experience


_FORBIDDEN_RESULT_KEYS = frozenset(
    {
        "anthropic_api_key",
        "api_key",
        "authorization",
        "content_text",
        "data_base64",
        "document_bytes",
        "documents",
        "dsn",
        "file_path",
        "image_bytes",
        "image_pages",
        "manifest_path",
        "ocr_ledger_path",
        "pdf_pages",
        "raw",
        "raw_bytes",
        "raw_prompt",
        "raw_response",
        "secret",
        "service_token",
        "source_path",
        "token",
    }
)
_MAX_SERVICE_REQUEST_JSON_BYTES = 32 * 1024 * 1024
_MAX_SERVICE_REQUEST_JSON_NODES = 20_000
_MAX_SERVICE_REQUEST_JSON_DEPTH = 32
_SENSITIVE_FRAGMENT_CHARS = 256
_SENSITIVE_FRAGMENT_STEP = 128
_BASE64_RUN_PATTERN = re.compile(rf"[A-Za-z0-9+/=]{{{_SENSITIVE_FRAGMENT_CHARS},}}")
_SERVICE_REQUEST_VALIDATOR = Draft202012Validator(
    get_schema("backend", "v2", "review-request"),
    format_checker=FormatChecker(),
)
_SERVICE_REQUEST_V3_VALIDATOR = Draft202012Validator(
    get_schema("backend", "v3", "review-request"),
    format_checker=FormatChecker(),
)
_EVIDENCE_BEARER_TOKEN = re.compile(r"^[A-Za-z0-9\-._~+/]+={0,8}$")


def _key_safety_token(key: str) -> str:
    return "".join(
        character for character in key.casefold() if character.isascii() and character.isalnum()
    )


_FORBIDDEN_RESULT_KEY_TOKENS = frozenset(_key_safety_token(key) for key in _FORBIDDEN_RESULT_KEYS)


class _DuplicateJsonKeyError(ValueError):
    pass


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKeyError
        value[key] = item
    return value


def _reject_nonfinite_json_constant(_value: str) -> Any:
    raise ValueError


def _assert_plain_json_request(value: Any) -> None:
    """Reject constructed models and custom containers at the untrusted dict boundary."""

    stack: list[tuple[Any, int]] = [(value, 0)]
    node_count = 0
    string_char_count = 0
    while stack:
        item, depth = stack.pop()
        node_count += 1
        if node_count > _MAX_SERVICE_REQUEST_JSON_NODES or depth > _MAX_SERVICE_REQUEST_JSON_DEPTH:
            raise RuntimeInputError(
                "review_request_invalid",
                "The Agent request exceeds the service structure limit.",
            )
        if item is None or type(item) is bool:
            continue
        if type(item) is int:
            if item.bit_length() > 4096:
                raise RuntimeInputError(
                    "review_request_invalid",
                    "The Agent request contains an unsupported numeric value.",
                )
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise RuntimeInputError(
                    "review_request_invalid",
                    "The Agent request contains an unsupported numeric value.",
                )
            continue
        if type(item) is str:
            string_char_count += len(item)
            if string_char_count > _MAX_SERVICE_REQUEST_JSON_BYTES:
                raise RuntimeInputError(
                    "review_request_too_large",
                    "The Agent request exceeds the service safety limit.",
                )
            continue
        if type(item) is list:
            if len(item) > 1_000:
                raise RuntimeInputError(
                    "review_request_invalid",
                    "The Agent request exceeds the service structure limit.",
                )
            stack.extend((nested, depth + 1) for nested in item)
            continue
        if type(item) is dict:
            if len(item) > 1_000 or any(type(key) is not str for key in item):
                raise RuntimeInputError(
                    "review_request_invalid",
                    "The Agent request must contain only JSON object keys.",
                )
            stack.extend((key, depth + 1) for key in item)
            stack.extend((nested, depth + 1) for nested in item.values())
            continue
        raise RuntimeInputError(
            "review_request_invalid",
            "The Agent request must contain only plain JSON values.",
        )


def _decode_serialized_json_object(
    payload: bytes | bytearray | memoryview,
    *,
    invalid_code: str,
) -> dict[str, Any]:
    if type(payload) not in {bytes, bytearray, memoryview}:
        raise RuntimeInputError(invalid_code, "The request must be UTF-8 JSON bytes.")
    try:
        payload_size = payload.nbytes if isinstance(payload, memoryview) else len(payload)
        raw = bytes(payload)
    except (BufferError, TypeError, ValueError):
        raise RuntimeInputError(invalid_code, "The request bytes are unavailable.") from None
    if payload_size > _MAX_SERVICE_REQUEST_JSON_BYTES or len(raw) > _MAX_SERVICE_REQUEST_JSON_BYTES:
        del raw
        raise RuntimeInputError("review_request_too_large", "The request exceeds the safety limit.")
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise RuntimeInputError(
            invalid_code,
            "The request is not valid duplicate-key-free JSON.",
        ) from None
    finally:
        del raw
    if not isinstance(decoded, dict):
        raise RuntimeInputError(invalid_code, "The request must contain one JSON object.")
    _assert_plain_json_request(decoded)
    return decoded


def _contains_sensitive_fragment(value: str, marker: str) -> bool:
    """Detect full secrets and substantial raw-media fragments without echoing them."""

    if not marker:
        return False
    if value == marker or (len(marker) >= 16 and marker in value):
        return True
    if len(marker) < _SENSITIVE_FRAGMENT_CHARS or len(value) < _SENSITIVE_FRAGMENT_CHARS:
        return False
    for match in _BASE64_RUN_PATTERN.finditer(value):
        candidate = match.group(0)
        if candidate in marker:
            return True
        last_start = len(candidate) - _SENSITIVE_FRAGMENT_CHARS
        starts = set(range(0, last_start + 1, _SENSITIVE_FRAGMENT_STEP))
        starts.add(last_start)
        if any(candidate[start : start + _SENSITIVE_FRAGMENT_CHARS] in marker for start in starts):
            return True
    return False


class AgentRuntimeError(RuntimeError):
    """A content-free error safe to return across a service boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class RuntimeConfigurationError(AgentRuntimeError):
    pass


class RuntimeInputError(AgentRuntimeError):
    pass


class RuntimeConflictError(AgentRuntimeError):
    pass


class RuntimeExecutionError(AgentRuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Validated dependencies for the production V2 Agent runtime.

    Visual service traffic accepts only a release-attested OCR wrapper. Raw OCR
    providers belong to evaluation-only composition outside this public
    runtime; the runtime never guesses an engine or silently falls back to
    Claude-only extraction.
    """

    claude_settings: ClaudeSettings = field(repr=False)
    evidence_base_url: str
    ocr_layout_provider: OcrLayoutProvider | None = field(default=None, repr=False)
    evidence_service_token: SecretStr | None = field(default=None, repr=False)
    evidence_allowed_origins: frozenset[str] = field(default_factory=frozenset)
    evidence_timeout_seconds: float = 60.0
    evidence_max_retries: int = 1
    verification_assertion_verifier: TrustedAssertionVerifier | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.claude_settings, ClaudeSettings):
            raise RuntimeConfigurationError(
                "claude_settings_invalid",
                "RuntimeConfig requires validated ClaudeSettings.",
            )
        if self.verification_assertion_verifier is not None and not callable(
            self.verification_assertion_verifier
        ):
            raise RuntimeConfigurationError(
                "verification_assertion_verifier_invalid",
                "Supporting-document assertion verifier is invalid.",
            )
        if self.ocr_layout_provider is not None and not callable(
            getattr(self.ocr_layout_provider, "analyze", None)
        ):
            raise RuntimeConfigurationError(
                "ocr_layout_provider_invalid",
                "The configured OCR provider does not implement the required interface.",
            )
        if self.ocr_layout_provider is not None:
            if not isinstance(
                self.ocr_layout_provider,
                ReleaseLockedOcrProvider,
            ):
                raise RuntimeConfigurationError(
                    "ocr_release_gate_required",
                    "Service OCR requires a release-attested provider wrapper.",
                )
            release_error_code: str | None = None
            try:
                self.ocr_layout_provider.assert_service_ready()
            except OcrLayoutProviderError as exc:
                release_error_code = exc.code
            if release_error_code is not None:
                raise RuntimeConfigurationError(
                    release_error_code,
                    "The configured OCR release Gate is not ready.",
                )
        if not isinstance(self.evidence_service_token, SecretStr):
            raise RuntimeConfigurationError(
                "evidence_service_token_required",
                "Production Evidence credentials must be supplied as SecretStr.",
            )
        evidence_token = self.evidence_service_token.get_secret_value()
        if (
            _EVIDENCE_BEARER_TOKEN.fullmatch(evidence_token) is None
            or len(set(evidence_token)) < 12
        ):
            raise RuntimeConfigurationError(
                "evidence_service_token_invalid",
                "Production Evidence credentials are invalid.",
            )
        if not isinstance(self.evidence_allowed_origins, frozenset) or not all(
            isinstance(origin, str) for origin in self.evidence_allowed_origins
        ):
            raise RuntimeConfigurationError(
                "evidence_allowed_origins_invalid",
                "Evidence allowed origins must be a frozenset of exact origins.",
            )
        if (
            isinstance(self.evidence_timeout_seconds, bool)
            or not isinstance(self.evidence_timeout_seconds, (int, float))
            or self.evidence_timeout_seconds <= 0
            or self.evidence_timeout_seconds > 300
        ):
            raise RuntimeConfigurationError(
                "evidence_timeout_invalid",
                "Evidence timeout must be greater than zero and at most 300 seconds.",
            )
        if (
            isinstance(self.evidence_max_retries, bool)
            or not isinstance(self.evidence_max_retries, int)
            or not 0 <= self.evidence_max_retries <= 3
        ):
            raise RuntimeConfigurationError(
                "evidence_retries_invalid",
                "Evidence retries must be between zero and three.",
            )
        if not isinstance(self.evidence_base_url, str):
            raise RuntimeConfigurationError(
                "evidence_base_url_invalid",
                "Evidence Service base URL is invalid.",
            )
        invalid_destination = False
        try:
            normalized_url = validate_evidence_destination(
                self.evidence_base_url,
                self.evidence_allowed_origins,
            )
        except (TypeError, ValueError):
            invalid_destination = True
        if invalid_destination:
            raise RuntimeConfigurationError(
                "evidence_base_url_invalid",
                "Evidence Service base URL is invalid or not explicitly allowed.",
            )
        object.__setattr__(self, "evidence_base_url", normalized_url)

    @classmethod
    def from_environment(
        cls,
        *,
        evidence_base_url: str,
        env_file: str | PathLike[str] | None = None,
        ocr_layout_provider: OcrLayoutProvider | None = None,
        evidence_service_token: SecretStr | None = None,
        evidence_allowed_origins: frozenset[str] = frozenset(),
        evidence_timeout_seconds: float = 60.0,
        evidence_max_retries: int = 1,
        verification_assertion_verifier: TrustedAssertionVerifier | None = None,
    ) -> RuntimeConfig:
        """Load only allowlisted Claude settings; Evidence settings stay explicit."""

        return cls(
            claude_settings=ClaudeSettings.from_environment(env_file),
            evidence_base_url=evidence_base_url,
            ocr_layout_provider=ocr_layout_provider,
            evidence_service_token=evidence_service_token,
            evidence_allowed_origins=evidence_allowed_origins,
            evidence_timeout_seconds=evidence_timeout_seconds,
            evidence_max_retries=evidence_max_retries,
            verification_assertion_verifier=verification_assertion_verifier,
        )


def _has_visual_documents(
    request: ServiceReviewRunRequestV2 | ServiceReviewRunRequestV3,
) -> bool:
    return any(
        document.image_pages or document.pdf_pages is not None
        for document in (request.documents or [])
    )


def _project_service_request_v3_to_v2(
    request: ServiceReviewRunRequestV3,
) -> ServiceReviewRunRequestV2:
    raw = request.model_dump(mode="python", warnings=False)
    projected = {name: raw[name] for name in ServiceReviewRunRequestV2.model_fields if name in raw}
    # Presence-only requirement codes are never derived from caller assertions.
    # Evidence v3 policy and verification assertions are evaluated separately.
    projected["uploaded_requirement_document_codes"] = []
    return ServiceReviewRunRequestV2.model_validate(projected)


def _input_base64_markers(request: ServiceReviewRunRequestV2) -> tuple[str, ...]:
    markers: list[str] = []
    for document in request.documents or []:
        pages = document.pdf_pages.pages if document.pdf_pages is not None else document.image_pages
        markers.extend(page.data_base64 for page in pages)
    return tuple(markers)


def _assert_result_correlated(
    request: ServiceReviewRunRequestV2,
    result: ReviewRunResultV2,
) -> None:
    if request.case_id is None or request.documents is None:
        raise RuntimeExecutionError(
            "review_result_correlation_failed",
            "The Agent result could not be correlated to its request.",
        )
    if (
        result.case_id != request.case_id
        or result.review_run_id != request.review_run_id
        or result.review_as_of != request.review_as_of
        or result.trade_type != request.trade_type
        or result.jurisdiction != request.jurisdiction
    ):
        raise RuntimeExecutionError(
            "review_result_correlation_failed",
            "The Agent result identity differs from its request.",
        )
    expected_modalities = {
        document.document_id: ("text" if document.content_text is not None else "image")
        for document in request.documents
    }
    observed_modalities = {
        document.document_id: document.input_modality for document in result.document_results
    }
    if observed_modalities != expected_modalities:
        raise RuntimeExecutionError(
            "review_result_correlation_failed",
            "The Agent result documents differ from its request.",
        )


def _assert_public_corrections_applied(
    result: ReviewRunResultV3,
    corrections: list[UserCorrection],
) -> None:
    """Bind every claimed human correction to the exact public child value."""

    for correction in corrections:
        documents = [
            document
            for document in result.document_results
            if document.document_id == correction.document_id
        ]
        if len(documents) != 1:
            raise RuntimeConflictError(
                "correction_application_failed",
                "An approved correction is not present exactly once in the child result.",
            )
        document = documents[0]
        parts = correction.field_path.split(".")
        if len(parts) == 2 and parts[0] == "fields":
            fields = [field for field in document.fields if field.name == parts[1]]
        elif (
            len(parts) == 3
            and parts[0] == "line_items"
            and parts[1].isascii()
            and parts[1].isdigit()
            and 0 <= int(parts[1]) < len(document.line_items)
        ):
            fields = [
                field
                for field in document.line_items[int(parts[1])].fields
                if field.name == parts[2]
            ]
        else:
            fields = []
        if len(fields) != 1:
            raise RuntimeConflictError(
                "correction_application_failed",
                "An approved correction target is absent or ambiguous in the child result.",
            )
        public_field = fields[0]
        try:
            actual = json.dumps(
                public_field.value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            expected = json.dumps(
                correction.value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError):
            raise RuntimeConflictError(
                "correction_application_failed",
                "An approved correction is not representable in the child result.",
            ) from None
        if (
            type(public_field.value) is not type(correction.value)
            or actual != expected
            or public_field.confidence != correction.confidence
        ):
            raise RuntimeConflictError(
                "correction_application_failed",
                "An approved correction was not reflected exactly in the child result.",
            )


def _assert_public_result_safe(
    value: Any,
    *,
    sensitive_markers: tuple[str, ...],
    depth: int = 0,
) -> None:
    if depth > 32:
        raise RuntimeExecutionError(
            "public_result_safety_violation",
            "The Agent result exceeded the public safety inspection depth.",
        )
    if isinstance(value, dict):
        normalized_keys = {_key_safety_token(str(key)) for key in value}
        if normalized_keys.intersection(_FORBIDDEN_RESULT_KEY_TOKENS):
            raise RuntimeExecutionError(
                "public_result_safety_violation",
                "Sensitive input material reached the public Agent result.",
            )
        for item in value.values():
            _assert_public_result_safe(
                item,
                sensitive_markers=sensitive_markers,
                depth=depth + 1,
            )
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_public_result_safe(
                item,
                sensitive_markers=sensitive_markers,
                depth=depth + 1,
            )
    elif isinstance(value, str):
        for marker in sensitive_markers:
            if _contains_sensitive_fragment(value, marker):
                raise RuntimeExecutionError(
                    "public_result_safety_violation",
                    "Sensitive input material reached the public Agent result.",
                )


class _RuntimeBoundedAdaptiveExecutor:
    """Runtime-owned active Tool composition over the mandatory workflow.

    The request cannot inject handlers. A field reread is bound to one
    canonical FactLedger field reference, re-invokes the pinned Claude
    analyzer for only the owning immutable document, and accepts only an exact
    re-observation of that field's value and source span. Evidence retry reuses
    only the already-pinned v3 retriever. Both paths then run code-owned
    deterministic stages.
    """

    def __init__(
        self,
        *,
        service_request: ServiceReviewRunRequestV2,
        request: ServiceReviewRunRequestV3,
        retriever: ProductionHttpRegulationRetrieverV3,
        analyzer: ClaudeDocumentAnalyzer,
        state: dict[str, Any],
        result: ReviewRunResultV2,
        supporting_document_rejections: list[str],
        assertion_verifier: TrustedAssertionVerifier | None,
    ) -> None:
        self._service_request = service_request
        self._request = request
        self._retriever = retriever
        self._analyzer = analyzer
        self._state = copy.deepcopy(state)
        self._result = result
        self._supporting_rejections = list(supporting_document_rejections)
        self._assertion_verifier = assertion_verifier
        self._reference_kinds: dict[str, str] = {}
        self._reference_documents: dict[str, str | None] = {}
        self._field_targets: dict[str, dict[str, Any]] = {}
        self._pending_reobserved_references: set[str] = set()
        self._pending_state: dict[str, Any] | None = None
        self._next_retrieval_attempt = int(state.get("retrieval_attempt") or 0)

    def _reserve_retrieval_attempt(self) -> int:
        """Consume one process-local Evidence attempt before any Tool I/O.

        A Tool observation can be rejected during deterministic reevaluation.
        The HTTP request still happened, so its attempt number must never be
        reused by a later Tool action even when the pending state is discarded.
        """

        attempt = self._next_retrieval_attempt
        self._next_retrieval_attempt += 1
        return attempt

    def _index_field_targets(self) -> dict[str, dict[str, Any]]:
        targets: dict[str, dict[str, Any]] = {}
        for document in self._result.document_results:
            for document_field in document.fields:
                reference_id = stable_object_id(
                    "field",
                    {
                        "document_id": document.document_id,
                        "field_path": f"fields.{document_field.name}",
                        "source": document_field.source.model_dump(mode="json"),
                    },
                )
                targets[reference_id] = {
                    "document_id": document.document_id,
                    "field_name": document_field.name,
                    "line_item_id": None,
                    # A reread observes the source value, not the later
                    # code-normalized projection. Normalization is rerun as a
                    # separate deterministic stage after the observation.
                    "value": document_field.value,
                    "source": document_field.source.model_dump(mode="json"),
                }
            for line_item in document.line_items:
                for item_field in line_item.fields:
                    reference_id = stable_object_id(
                        "field",
                        {
                            "document_id": document.document_id,
                            "field_path": (f"line_items.{line_item.item_id}.{item_field.name}"),
                            "source": item_field.source.model_dump(mode="json"),
                        },
                    )
                    targets[reference_id] = {
                        "document_id": document.document_id,
                        "field_name": item_field.name,
                        "line_item_id": line_item.item_id,
                        "value": item_field.value,
                        "source": item_field.source.model_dump(mode="json"),
                    }
        return targets

    @staticmethod
    def _mapped_field(
        mapped_document: dict[str, Any],
        target: dict[str, Any],
    ) -> dict[str, Any] | None:
        line_item_id = target["line_item_id"]
        field_name = target["field_name"]
        if line_item_id is None:
            value = (mapped_document.get("fields") or {}).get(field_name)
            return value if isinstance(value, dict) else None
        for line_item in mapped_document.get("line_items") or []:
            if (
                isinstance(line_item, dict)
                and line_item.get("item_id") == line_item_id
                and isinstance(line_item.get(field_name), dict)
            ):
                return line_item[field_name]
        return None

    @staticmethod
    def _same_public_value(left: Any, right: Any) -> bool:
        return json.dumps(
            left,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ) == json.dumps(
            right,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )

    def _finalize_state(
        self,
        state: dict[str, Any],
    ) -> tuple[dict[str, Any], ReviewRunResultV2, list[str]]:
        retrieval_request = state.get("retrieval_request") or {}
        request_id = retrieval_request.get("request_id")
        policies = (
            self._retriever.policy_bindings(request_id) if isinstance(request_id, str) else {}
        )
        verified_by_item, supporting_rejections = verified_requirement_codes_by_item(
            self._request.supporting_documents,
            case_id=self._request.case_id,
            review_as_of=self._request.review_as_of,
            assertion_cutoff_at=self._request.assertion_cutoff_at,
            retrieval_results=list(state.get("retrieval_results") or []),
            policy_bindings=policies,
            assertion_verifier=self._assertion_verifier,
        )
        next_state = copy.deepcopy(state)
        next_state["verified_requirement_documents_by_item"] = verified_by_item
        next_state["presented_requirement_documents_by_item"] = presented_requirement_codes_by_item(
            self._request.supporting_documents,
            retrieval_results=list(state.get("retrieval_results") or []),
        )
        # A prior route result is not an independent source of truth. Rebuild
        # the human flag from provider/model facts, then let compliance/outcome
        # derive it again from the complete state.
        next_state["requires_human_review"] = bool(
            next_state.get("model_errors")
            or next_state.get("model_conflicts")
            or extraction_completion_blockers(next_state)
        )
        next_state.update(compliance_review(next_state))
        next_state.update(generate_xai_report(next_state))
        next_state.update(route_to_human_or_complete(next_state))
        provisional = to_review_run_result_v2(next_state)
        observed_model = (
            provisional.model_calls[0].model_id
            if provisional.model_calls
            else self._analyzer.model_id
        )
        runtime_release = runtime_release_identity(
            model_id=observed_model,
            evidence_snapshot_id=provisional.knowledge_snapshot_id,
            evidence_runtime_v3_sha256=(
                self._retriever.expected_identity.runtime_release_v3_sha256
            ),
            assertion_trust_store=(
                self._assertion_verifier.store
                if isinstance(
                    self._assertion_verifier,
                    SupportingDocumentAssertionVerifier,
                )
                else None
            ),
        )
        calibration_blockers = automatic_completion_unmet_reason_codes(
            provisional,
            self._request,
            runtime_release,
            supporting_rejections=supporting_rejections,
            require_completed_outcome=False,
        )
        # Calibration approval may remove only the release-calibration hold.
        # It must never erase deterministic document-set/pairing blockers that
        # were reconstructed above (for example duplicate required documents or
        # a declared/observed document-type conflict).
        if not calibration_blockers and not extraction_completion_blockers(next_state):
            next_state["requires_human_review"] = False
            next_state.update(
                route_to_human_or_complete(
                    next_state,
                    extraction_release_approved=True,
                )
            )
        final_result = to_review_run_result_v2(next_state)
        return next_state, final_result, supporting_rejections

    def _reread_span(self, reference_ids: tuple[str, ...]) -> BoundedToolResult:
        self._pending_reobserved_references = set()
        if len(reference_ids) != 1:
            return BoundedToolResult(status="failed", reason_code="span_reference_not_grounded")
        reference_id = reference_ids[0]
        target = self._field_targets.get(reference_id)
        if self._reference_kinds.get(reference_id) != "field_source" or target is None:
            return BoundedToolResult(status="failed", reason_code="span_reference_not_grounded")
        document = next(
            (
                item
                for item in self._service_request.documents or []
                if item.document_id == target["document_id"]
            ),
            None,
        )
        if document is None or not document.content_text:
            return BoundedToolResult(status="failed", reason_code="span_document_not_in_request")
        source = target["source"]
        text_start = source.get("text_start")
        text_end = source.get("text_end")
        if (
            isinstance(text_start, bool)
            or isinstance(text_end, bool)
            or not isinstance(text_start, int)
            or not isinstance(text_end, int)
            or not 0 <= text_start < text_end <= len(document.content_text)
        ):
            return BoundedToolResult(
                status="failed",
                reason_code="span_coordinates_not_grounded",
            )
        analysis_document = document.model_dump(mode="python", warnings=False)
        try:
            analysis = self._analyzer.analyze(analysis_document)
            mapped = analysis_to_extracted_document(
                analysis_document,
                analysis,
            )
            fresh_field = self._mapped_field(mapped, target)
        except (LLMError, RetrievalError, ValidationError, ValueError):
            self._pending_state = None
            return BoundedToolResult(status="failed", reason_code="span_reread_failed")
        except Exception:
            self._pending_state = None
            return BoundedToolResult(status="failed", reason_code="span_reread_failed")
        if fresh_field is None:
            return BoundedToolResult(
                status="failed",
                reason_code="span_target_not_reobserved",
            )
        fresh_value = fresh_field.get("value")
        fresh_evidence = fresh_field.get("evidence")
        fresh_start = fresh_evidence.get("text_start") if isinstance(fresh_evidence, dict) else None
        fresh_end = fresh_evidence.get("text_end") if isinstance(fresh_evidence, dict) else None
        source_overlap = (
            isinstance(fresh_start, int)
            and not isinstance(fresh_start, bool)
            and isinstance(fresh_end, int)
            and not isinstance(fresh_end, bool)
            and 0 <= fresh_start < fresh_end <= len(document.content_text)
            and max(text_start, fresh_start) < min(text_end, fresh_end)
        )
        if not self._same_public_value(fresh_value, target["value"]) or not source_overlap:
            return BoundedToolResult(
                status="failed",
                reason_code="span_reobservation_conflicted",
            )
        next_state = copy.deepcopy(self._state)
        model_calls = list(next_state.get("model_calls") or [])
        model_calls.append(
            {
                "document_id": document.document_id,
                **analysis.metadata.as_audit_dict(),
            }
        )
        model_calls.extend(
            {
                "document_id": document.document_id,
                **item.as_audit_dict(),
            }
            for item in analysis.additional_model_calls
        )
        next_state["model_calls"] = model_calls
        next_state["model_errors"] = [
            *(next_state.get("model_errors") or []),
            *analysis.additional_model_errors,
        ]
        next_state["model_conflicts"] = [
            *(next_state.get("model_conflicts") or []),
            *analysis.model_conflicts,
        ]
        if analysis.additional_model_errors or analysis.model_conflicts:
            self._pending_state = None
            return BoundedToolResult(
                status="failed",
                reason_code="span_reobservation_provider_conflict",
            )
        # The value is intentionally not replaced and its fixed 0.85
        # uncalibrated confidence is never increased. Re-run only the
        # deterministic downstream stages over the immutable extraction.
        next_state.update(normalize_documents(next_state))
        next_state.update(pair_documents(next_state))
        next_state.update(validate_cross_documents(next_state))
        self._pending_state = next_state
        self._pending_reobserved_references = {reference_id}
        return BoundedToolResult(
            status="succeeded",
            reason_code="source_bound_field_reobserved_exactly",
            output_reference_ids=reference_ids,
        )

    def _retry_evidence(self, reference_ids: tuple[str, ...]) -> BoundedToolResult:
        if any(
            self._reference_kinds.get(reference_id) not in {"evidence", "outcome"}
            for reference_id in reference_ids
        ):
            return BoundedToolResult(status="failed", reason_code="evidence_reference_not_grounded")
        retrieval_attempt = self._reserve_retrieval_attempt()
        try:
            next_state = copy.deepcopy(self._state)
            next_state["retrieval_attempt"] = retrieval_attempt
            next_state.update(retrieve_regulations(next_state, self._retriever))
        except (RetrievalError, ValidationError, ValueError):
            self._pending_state = None
            return BoundedToolResult(status="failed", reason_code="evidence_retry_failed")
        except Exception:
            self._pending_state = None
            return BoundedToolResult(status="failed", reason_code="evidence_retry_failed")
        self._pending_state = next_state
        return BoundedToolResult(
            status="succeeded",
            reason_code="pinned_evidence_retry_completed",
            output_reference_ids=reference_ids,
        )

    def _reevaluate(
        self,
        *,
        action: PlannedToolActionV1,
        observation: BoundedToolResult,
        affected_stages: tuple[str, ...],
    ) -> DeterministicReevaluationResult:
        pending = self._pending_state
        self._pending_state = None
        expected = (
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
        if observation.status != "succeeded" or pending is None or affected_stages != expected:
            self._pending_reobserved_references = set()
            return DeterministicReevaluationResult(
                status="failed",
                reason_code="affected_stage_rerun_not_authorized",
            )
        try:
            next_state, next_result, supporting_rejections = self._finalize_state(pending)
        except (LLMError, RetrievalError, ValidationError, ValueError):
            self._pending_reobserved_references = set()
            return DeterministicReevaluationResult(
                status="failed",
                reason_code="deterministic_stage_rerun_failed",
            )
        except Exception:
            self._pending_reobserved_references = set()
            return DeterministicReevaluationResult(
                status="failed",
                reason_code="deterministic_stage_rerun_failed",
            )
        if action.tool_name == "document.reread_span/v1":
            reobserved_references = self._pending_reobserved_references
            self._pending_reobserved_references = set()
            exact_reobservation = (
                bool(action.input_reference_ids)
                and set(action.input_reference_ids) == reobserved_references
                and not next_result.model_errors
                and not next_result.model_conflicts
            )
            if not exact_reobservation:
                return DeterministicReevaluationResult(
                    status="failed",
                    reason_code="deterministic_field_reobservation_not_grounded",
                )
            # A second observation is useful evidence that the deterministic
            # downstream stages were rerun, but the current Claude extraction
            # release remains fixed at an uncalibrated confidence of 0.85.
            # Therefore the field_uncertainty Gap must stay open. Claiming it
            # as resolved here would manufacture confidence that the provider
            # did not supply.
            self._state = next_state
            self._result = next_result
            self._supporting_rejections = supporting_rejections
            return DeterministicReevaluationResult(
                status="succeeded",
                reason_code="affected_deterministic_stages_reexecuted_gap_open",
                resolved_gap_ids=(),
                output_reference_ids=tuple(sorted(reobserved_references)),
            )
        else:
            resolved = (
                next_result.retrieval_status == "ok"
                and next_result.retrieval_coverage == "complete"
                and not next_result.retrieval_errors
            )
        if not resolved:
            return DeterministicReevaluationResult(
                status="failed",
                reason_code="deterministic_gap_remains_open",
            )
        self._state = next_state
        self._result = next_result
        self._supporting_rejections = supporting_rejections
        return DeterministicReevaluationResult(
            status="succeeded",
            reason_code="affected_deterministic_stages_reexecuted",
            resolved_gap_ids=(action.gap_id,),
            output_reference_ids=(action.gap_id,),
        )

    def execute(
        self,
        *,
        plan: ReviewPlanProposalV1,
        fact_ledger: FactLedgerV1,
        uat_capability: AgentUatCapabilityAttestationV1 | None = None,
    ) -> ActiveBoundedRuntimeOutcome:
        self._reference_kinds = {entry.reference_id: entry.kind for entry in fact_ledger.entries}
        self._reference_documents = {
            entry.reference_id: entry.document_id for entry in fact_ledger.entries
        }
        self._field_targets = self._index_field_targets()
        self._pending_reobserved_references = set()
        observed_model = (
            self._result.model_calls[0].model_id
            if self._result.model_calls
            else self._analyzer.model_id
        )
        active_runtime_release = runtime_release_identity(
            model_id=observed_model,
            evidence_snapshot_id=self._result.knowledge_snapshot_id,
            evidence_runtime_v3_sha256=(
                self._retriever.expected_identity.runtime_release_v3_sha256
            ),
            assertion_trust_store=(
                self._assertion_verifier.store
                if isinstance(
                    self._assertion_verifier,
                    SupportingDocumentAssertionVerifier,
                )
                else None
            ),
        )
        traces, observations, resolved = execute_active_bounded_plan(
            plan,
            broker=BoundedToolBroker(
                span_reread=self._reread_span,
                evidence_retry=self._retry_evidence,
            ),
            reevaluate=self._reevaluate,
            policy_id=AGENT_POLICY_ID,
            uat_capability=uat_capability,
            runtime_release=active_runtime_release,
        )
        failures: set[str] = set()
        if len(traces) < len(plan.actions):
            failures.add("agent_budget_exhausted")
        second_cycle_evidence_failed = any(
            trace.tool_name == "evidence.retrieve/v3"
            and trace.reason_code == "evidence_retry_cycle_2_if_unresolved"
            and trace.cycle == 1
            and trace.status == "failed"
            for trace in traces
        )
        if second_cycle_evidence_failed:
            failures.add("agent_budget_exhausted")
        if any(
            item.status == "failed"
            or (item.status == "deferred" and item.reason_code != "gap_already_resolved")
            for item in observations
        ):
            failures.add("active_tool_execution_incomplete")
        return ActiveBoundedRuntimeOutcome(
            result=self._result,
            supporting_document_rejections=tuple(self._supporting_rejections),
            action_trace=tuple(traces),
            observations=tuple(observations),
            resolved_gap_ids=tuple(resolved),
            failure_reason_codes=tuple(sorted(failures)),
        )


class AgentRuntime:
    """Service-facing Claude + OCR + Evidence HTTP composition root."""

    __slots__ = (
        "_analyzer",
        "_closed",
        "_config",
        "_retriever",
        "_retriever_v3",
        "_secret_markers",
        "_assertion_verifier",
        "_uat_capability",
        "_uat_expected_agent_commit",
    )

    def __init__(self, config: RuntimeConfig) -> None:
        if not isinstance(config, RuntimeConfig):
            raise RuntimeConfigurationError(
                "runtime_config_invalid",
                "AgentRuntime requires a validated RuntimeConfig.",
            )
        token = (
            config.evidence_service_token.get_secret_value()
            if config.evidence_service_token is not None
            else None
        )
        try:
            load_released_evidence_runtime_identity()
        except EvidenceRuntimeReleaseUnreleasedError:
            raise RuntimeConfigurationError(
                "evidence_runtime_release_unreleased",
                "No sealed Evidence runtime release is approved for production.",
            ) from None
        except EvidenceRuntimeReleaseLockError:
            raise RuntimeConfigurationError(
                "evidence_runtime_release_lock_invalid",
                "The Evidence runtime release lock is invalid.",
            ) from None
        dependency_configuration_failed = False
        try:
            retriever = ProductionHttpRegulationRetriever(
                config.evidence_base_url,
                service_token=token,
                allowed_origins=config.evidence_allowed_origins,
                timeout_seconds=config.evidence_timeout_seconds,
                max_retries=config.evidence_max_retries,
            )
            analyzer = ClaudeDocumentAnalyzer.from_settings(
                config.claude_settings,
                ocr_layout_provider=config.ocr_layout_provider,
            )
        except Exception:
            dependency_configuration_failed = True
        if dependency_configuration_failed:
            raise RuntimeConfigurationError(
                "runtime_dependency_configuration_failed",
                "Claude, OCR, or Evidence runtime configuration is invalid.",
            )
        self._config = config
        self._retriever = retriever
        self._retriever_v3: ProductionHttpRegulationRetrieverV3 | None = None
        self._analyzer = analyzer
        self._assertion_verifier = config.verification_assertion_verifier
        self._uat_capability: AgentUatCapabilityAttestationV1 | None = None
        self._uat_expected_agent_commit: str | None = None
        self._closed = False
        self._secret_markers = tuple(
            marker
            for marker in (
                config.claude_settings.api_key.get_secret_value(),
                token,
            )
            if marker
        )

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeExecutionError(
                "runtime_closed",
                "The Agent runtime is closed.",
            )

    def assert_uat_capability_current(
        self,
        *,
        require_active: bool = False,
    ) -> AgentUatCapabilityAttestationV1:
        """Revalidate the time-bound local-UAT authority without I/O.

        The service transport calls this before dependency readiness, and the
        direct runtime methods call it again before Claude or Evidence. This
        prevents a capability that expired after process startup from becoming
        a reusable in-memory bearer capability.
        """

        self._assert_open()
        capability = getattr(self, "_uat_capability", None)
        if not isinstance(capability, AgentUatCapabilityAttestationV1):
            raise RuntimeConfigurationError(
                "uat_capability_invalid_or_expired",
                "The local-UAT capability is absent, expired, or identity-mismatched.",
            )
        retriever = getattr(self, "_retriever_v3", None)
        expected_identity = getattr(retriever, "expected_identity", None)
        evidence_identity = getattr(
            expected_identity,
            "runtime_release_v3_sha256",
            None,
        )
        from customs_ai_agent.uat_capability import assert_runtime_uat_capability

        if not assert_runtime_uat_capability(
            capability,
            agent_code_snapshot_sha256=package_code_snapshot_sha256(),
            model_id=self._analyzer.model_id,
            evidence_runtime_v3_sha256=evidence_identity,
            expected_agent_commit=getattr(
                self,
                "_uat_expected_agent_commit",
                None,
            ),
            require_active=require_active,
        ):
            raise RuntimeConfigurationError(
                "uat_capability_invalid_or_expired",
                "The local-UAT capability is absent, expired, or identity-mismatched.",
            )
        return capability

    def _runtime_assertion_trust_store(self) -> AssertionTrustStoreV1 | None:
        """Return only the store owned by the concrete runtime verifier.

        Request data can never supply this identity.  Alternate verifier
        implementations remain fail-closed on the packaged unreleased store.
        """

        if isinstance(self._assertion_verifier, SupportingDocumentAssertionVerifier):
            return self._assertion_verifier.store
        return None

    def close(self) -> None:
        """Close the runtime-owned Anthropic client.

        ``RuntimeConfig.ocr_layout_provider`` is injected by the caller and
        remains caller-owned. Closing the Agent runtime therefore never closes
        that OCR provider.
        """

        if self._closed:
            return
        self._closed = True
        close_analyzer = getattr(self._analyzer, "close", None)
        if not callable(close_analyzer):
            return
        try:
            close_analyzer()
        except Exception:
            raise RuntimeExecutionError(
                "runtime_close_failed",
                "The Agent runtime could not close its provider client safely.",
            ) from None

    def __enter__(self) -> AgentRuntime:
        self._assert_open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def review_json(
        self,
        payload: bytes | bytearray | memoryview,
    ) -> ReviewRunResultV2:
        """Validate a bounded, duplicate-key-free UTF-8 JSON service request."""

        self._assert_open()
        if type(payload) not in {bytes, bytearray, memoryview}:
            raise RuntimeInputError(
                "review_request_invalid",
                "The serialized Agent request must be UTF-8 JSON bytes.",
            )
        payload_bytes_unavailable = False
        try:
            payload_size = payload.nbytes if isinstance(payload, memoryview) else len(payload)
        except ValueError:
            payload_bytes_unavailable = True
        if payload_bytes_unavailable:
            raise RuntimeInputError(
                "review_request_invalid",
                "The serialized Agent request bytes are unavailable.",
            )
        if payload_size > _MAX_SERVICE_REQUEST_JSON_BYTES:
            raise RuntimeInputError(
                "review_request_too_large",
                "The serialized Agent request exceeds the service safety limit.",
            )
        payload_bytes_unavailable = False
        try:
            raw = bytes(payload)
        except (BufferError, TypeError, ValueError):
            payload_bytes_unavailable = True
        if payload_bytes_unavailable:
            raise RuntimeInputError(
                "review_request_invalid",
                "The serialized Agent request bytes are unavailable.",
            )
        if len(raw) > _MAX_SERVICE_REQUEST_JSON_BYTES:
            del raw
            raise RuntimeInputError(
                "review_request_too_large",
                "The serialized Agent request exceeds the service safety limit.",
            )
        invalid_serialized_request = False
        try:
            decoded = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (UnicodeDecodeError, ValueError, RecursionError):
            invalid_serialized_request = True
        finally:
            del raw
        if invalid_serialized_request:
            raise RuntimeInputError(
                "review_request_invalid",
                "The serialized Agent request is not valid duplicate-key-free JSON.",
            )
        if not isinstance(decoded, dict):
            raise RuntimeInputError(
                "review_request_invalid",
                "The serialized Agent request must contain one JSON object.",
            )
        return self.review(decoded)

    def review(
        self,
        request: ServiceReviewRunRequestV2 | dict[str, Any],
    ) -> ReviewRunResultV2:
        """Run one real V2 review and return only the redacted public contract."""

        self._assert_open()
        if not isinstance(request, (ServiceReviewRunRequestV2, dict)):
            raise RuntimeInputError(
                "review_request_invalid",
                "The Agent runtime accepts ServiceReviewRunRequestV2 or a request object.",
            )
        # A Pydantic instance can have been created with ``model_construct`` or
        # mutated after validation. Re-parse a detached dump at the service
        # boundary instead of trusting the caller-owned object.
        request_serialization_failed = False
        try:
            request_payload = (
                request.model_dump(mode="json", warnings=False)
                if isinstance(request, ServiceReviewRunRequestV2)
                else request
            )
        except Exception:
            request_serialization_failed = True
        if request_serialization_failed:
            raise RuntimeInputError(
                "review_request_invalid",
                "The V2 Agent request could not be serialized safely.",
            )
        _assert_plain_json_request(request_payload)
        if next(_SERVICE_REQUEST_VALIDATOR.iter_errors(request_payload), None) is not None:
            raise RuntimeInputError(
                "review_request_invalid",
                "The V2 Agent request does not satisfy the service contract.",
            )
        request_validation_failed = False
        try:
            service_request = ServiceReviewRunRequestV2.model_validate(request_payload)
        except ValidationError:
            request_validation_failed = True
        if request_validation_failed:
            raise RuntimeInputError(
                "review_request_invalid",
                "The V2 Agent request does not satisfy the service contract.",
            )

        if _has_visual_documents(service_request) and self._config.ocr_layout_provider is None:
            raise RuntimeConfigurationError(
                "ocr_layout_provider_required",
                "Image and PDF reviews require an explicitly configured OCR/layout provider.",
            )

        return self._execute_review(service_request, retriever=self._retriever)

    def _get_retriever_v3(self) -> ProductionHttpRegulationRetrieverV3:
        if self._retriever_v3 is not None:
            return self._retriever_v3
        token = (
            self._config.evidence_service_token.get_secret_value()
            if self._config.evidence_service_token is not None
            else None
        )
        try:
            retriever = ProductionHttpRegulationRetrieverV3(
                self._config.evidence_base_url,
                service_token=token,
                allowed_origins=self._config.evidence_allowed_origins,
                timeout_seconds=self._config.evidence_timeout_seconds,
                max_retries=self._config.evidence_max_retries,
            )
        except Exception:
            raise RuntimeConfigurationError(
                "evidence_runtime_v3_unreleased",
                "No exact approved Evidence v3 runtime is pinned for this Agent build.",
            ) from None
        self._retriever_v3 = retriever
        return retriever

    def _execute_review(
        self,
        service_request: ServiceReviewRunRequestV2,
        *,
        retriever: Any,
    ) -> ReviewRunResultV2:

        sensitive_markers = self._secret_markers + _input_base64_markers(service_request)
        execution_failure: RuntimeExecutionError | None = None
        try:
            # Classification/extraction may invoke Claude. Production must
            # therefore attest the sealed Data dependency before entering the
            # workflow, even though each later retrieval re-attests independently.
            retriever.preflight(service_request.review_as_of)
            workflow_request = ReviewRunRequestV2.model_validate(
                service_request.model_dump(mode="python", warnings=False)
            )
            result = run_review_result_v2(
                workflow_request,
                retriever=retriever,
                analyzer=self._analyzer,
            )
            if not isinstance(result, ReviewRunResultV2):
                raise RuntimeExecutionError(
                    "review_result_invalid",
                    "The Agent workflow did not return the V2 public result contract.",
                )
            result_validation_failed = False
            try:
                # Re-validate even a model instance so a bypassed or mutated
                # result can never cross the public service boundary.
                result = ReviewRunResultV2.model_validate(
                    result.model_dump(mode="python", warnings=False)
                )
            except ValidationError:
                result_validation_failed = True
            if result_validation_failed:
                raise RuntimeExecutionError(
                    "review_result_invalid",
                    "The Agent workflow returned an invalid V2 public result.",
                )
            _assert_result_correlated(service_request, result)
            _assert_public_result_safe(
                result.model_dump(mode="python"),
                sensitive_markers=sensitive_markers,
            )
            return result
        except AgentRuntimeError:
            raise
        except (LLMError, OcrLayoutProviderError, RetrievalError):
            execution_failure = RuntimeExecutionError(
                "review_execution_failed",
                "The Agent review failed without exposing private execution details.",
            )
        except Exception:
            execution_failure = RuntimeExecutionError(
                "review_execution_failed",
                "The Agent review failed without exposing private execution details.",
            )
        if execution_failure is None:
            execution_failure = RuntimeExecutionError(
                "review_execution_failed",
                "The Agent review failed without exposing private execution details.",
            )
        raise execution_failure

    def review_v3_json(
        self,
        payload: bytes | bytearray | memoryview,
    ) -> ReviewRunResultV3:
        decoded = _decode_serialized_json_object(payload, invalid_code="review_v3_request_invalid")
        return self.review_v3(decoded)

    def review_v3(
        self,
        request: ServiceReviewRunRequestV3 | dict[str, Any],
        *,
        _forced_human_review_reasons: tuple[str, ...] = (),
        _stage_observer: WorkflowStageObserver | None = None,
    ) -> ReviewRunResultV3:
        """Execute mandatory policy stages plus release-gated bounded adaptation."""

        self._assert_open()
        try:
            payload = (
                request.model_dump(mode="json", warnings=False)
                if isinstance(request, ServiceReviewRunRequestV3)
                else request
            )
        except Exception:
            raise RuntimeInputError(
                "review_v3_request_invalid",
                "The V3 Agent request could not be serialized safely.",
            ) from None
        _assert_plain_json_request(payload)
        if next(_SERVICE_REQUEST_V3_VALIDATOR.iter_errors(payload), None) is not None:
            raise RuntimeInputError(
                "review_v3_request_invalid",
                "The V3 Agent request does not satisfy the service contract.",
            )
        try:
            service_request = ServiceReviewRunRequestV3.model_validate(payload)
        except ValidationError:
            raise RuntimeInputError(
                "review_v3_request_invalid",
                "The V3 Agent request does not satisfy the service contract.",
            ) from None
        if _has_visual_documents(service_request):
            raise RuntimeInputError(
                "visual_input_not_supported",
                "The V3 Production-candidate service is text-only.",
            )

        # V3 never projects self-asserted SupportingDocument codes into the V2
        # presence-only field.  Only the provider-owned Evidence v3 policy can
        # determine whether a requirement is verifiable; until an approved policy
        # is pinned, constructing this retriever fails closed before Claude runs.
        retriever_v3 = self._get_retriever_v3()
        v2_request = _project_service_request_v3_to_v2(service_request)
        v2_result, supporting_rejections, state = self._execute_review_v3(
            v2_request,
            request=service_request,
            retriever=retriever_v3,
            stage_observer=_stage_observer,
        )
        supporting_rejections = sorted({*supporting_rejections, *_forced_human_review_reasons})
        active_executor = _RuntimeBoundedAdaptiveExecutor(
            service_request=v2_request,
            request=service_request,
            retriever=retriever_v3,
            analyzer=self._analyzer,
            state=state,
            result=v2_result,
            supporting_document_rejections=supporting_rejections,
            assertion_verifier=self._assertion_verifier,
        )
        adaptive_started_ns = time.perf_counter_ns()
        if _stage_observer is not None:
            _stage_observer("__adaptive__", "started", {}, None)
        try:
            result = to_review_run_result_v3(
                v2_result,
                service_request,
                analyzer=self._analyzer,
                evidence_runtime_v3_sha256=(
                    retriever_v3.expected_identity.runtime_release_v3_sha256
                ),
                assertion_trust_store=self._runtime_assertion_trust_store(),
                supporting_document_rejections=supporting_rejections,
                active_executor=active_executor,
                uat_capability=getattr(self, "_uat_capability", None),
            )
        except Exception:
            if _stage_observer is not None:
                _stage_observer(
                    "__adaptive__",
                    "failed",
                    {},
                    max(
                        1,
                        min(
                            300_000,
                            round((time.perf_counter_ns() - adaptive_started_ns) / 1_000_000),
                        ),
                    ),
                )
            raise
        if _stage_observer is not None:
            _stage_observer(
                "__adaptive__",
                "completed",
                {
                    "issues": len(result.validation_issues),
                    "findings": len(result.regulatory_findings),
                    "evidence": len(result.rag_evidence),
                    "actions": len(result.action_trace),
                },
                max(
                    1,
                    min(
                        300_000,
                        round((time.perf_counter_ns() - adaptive_started_ns) / 1_000_000),
                    ),
                ),
            )
        _assert_public_result_safe(
            result.model_dump(mode="python", warnings=False),
            sensitive_markers=self._secret_markers,
        )
        return result

    def _execute_review_v3(
        self,
        service_request: ServiceReviewRunRequestV2,
        *,
        request: ServiceReviewRunRequestV3,
        retriever: ProductionHttpRegulationRetrieverV3,
        corrections: list[Any] | None = None,
        stage_observer: WorkflowStageObserver | None = None,
    ) -> tuple[ReviewRunResultV2, list[str], dict[str, Any]]:
        try:
            retriever.preflight(service_request.review_as_of)
            workflow_request = ReviewRunRequestV2.model_validate(
                service_request.model_dump(mode="python", warnings=False)
            )
            state = run_review_v2(
                workflow_request,
                retriever=retriever,
                analyzer=self._analyzer,
                stage_observer=stage_observer,
            )
            if corrections:
                # Human corrections never mutate the immutable parent request.
                # They are applied to the extracted child state and all affected
                # deterministic stages, including Evidence retrieval, are rerun.
                state = rerun_review(state, corrections, retriever=retriever)
            retrieval_request = state.get("retrieval_request") or {}
            request_id = retrieval_request.get("request_id")
            policies = retriever.policy_bindings(request_id) if isinstance(request_id, str) else {}
            verified_by_item, supporting_rejections = verified_requirement_codes_by_item(
                request.supporting_documents,
                case_id=request.case_id,
                review_as_of=request.review_as_of,
                assertion_cutoff_at=request.assertion_cutoff_at,
                retrieval_results=list(state.get("retrieval_results") or []),
                policy_bindings=policies,
                assertion_verifier=self._assertion_verifier,
            )
            state["verified_requirement_documents_by_item"] = verified_by_item
            state["presented_requirement_documents_by_item"] = presented_requirement_codes_by_item(
                request.supporting_documents,
                retrieval_results=list(state.get("retrieval_results") or []),
            )
            for node in (compliance_review, generate_xai_report, route_to_human_or_complete):
                state.update(node(state))
            result = to_review_run_result_v2(state)
            # Break the release-calibration bootstrap cycle without weakening a
            # business Gate.  The default V2 route correctly holds every LLM
            # extraction for human review.  For V3, evaluate all independently
            # signed non-Outcome Gates first; only when every one passes may the
            # code-owned Outcome policy be rerun with extraction calibration.
            observed_model = (
                result.model_calls[0].model_id if result.model_calls else self._analyzer.model_id
            )
            release_identity = runtime_release_identity(
                model_id=observed_model,
                evidence_snapshot_id=result.knowledge_snapshot_id,
                evidence_runtime_v3_sha256=retriever.expected_identity.runtime_release_v3_sha256,
                assertion_trust_store=self._runtime_assertion_trust_store(),
            )
            release_blockers = automatic_completion_unmet_reason_codes(
                result,
                request,
                release_identity,
                supporting_rejections=supporting_rejections,
                require_completed_outcome=False,
            )
            if not release_blockers and not extraction_completion_blockers(state):
                state["requires_human_review"] = False
                state.update(
                    route_to_human_or_complete(
                        state,
                        extraction_release_approved=True,
                    )
                )
                result = to_review_run_result_v2(state)
            _assert_result_correlated(service_request, result)
            _assert_public_result_safe(
                result.model_dump(mode="python"),
                sensitive_markers=self._secret_markers,
            )
            return result, supporting_rejections, state
        except AgentRuntimeError:
            raise
        except CorrectionApplicationError:
            raise RuntimeConflictError(
                "correction_application_failed",
                "An approved correction could not be applied exactly to the child review.",
            ) from None
        except (LLMError, OcrLayoutProviderError, RetrievalError, ValidationError, ValueError):
            raise RuntimeExecutionError(
                "review_execution_failed",
                "The V3 Agent review failed without exposing private execution details.",
            ) from None

    def answer_v3_json(
        self,
        payload: bytes | bytearray | memoryview,
    ) -> GroundedAnswerV1:
        self._assert_open()
        decoded = _decode_serialized_json_object(payload, invalid_code="answer_request_invalid")
        try:
            question = GroundedQuestionV1.model_validate(decoded)
            assert_result_hash(question.review_result)
        except (ValidationError, ValueError):
            raise RuntimeInputError(
                "answer_request_invalid",
                "The grounded-answer request is stale or invalid.",
            ) from None
        answer = answer_grounded_question(question, analyzer=self._analyzer)
        _assert_public_result_safe(
            answer.model_dump(mode="python", warnings=False),
            sensitive_markers=self._secret_markers,
        )
        return answer

    def review_experience_json(
        self,
        payload: bytes | bytearray | memoryview,
    ) -> ReviewExperienceEnvelopeV1:
        """Run one canonical review and add a verified user presentation.

        This additive boundary never mutates or replaces ``ReviewRunResultV3``.
        It is disabled unless the local-UAT composition injected an exact,
        unexpired capability attestation.
        """

        self._assert_open()
        capability = self.assert_uat_capability_current()
        result = self.review_v3_json(payload)
        envelope = build_review_experience(
            result,
            analyzer=self._analyzer,
            capability=capability,
        )
        _assert_public_result_safe(
            envelope.model_dump(mode="python", warnings=False),
            sensitive_markers=self._secret_markers,
        )
        return envelope

    def review_experience_stream_json(
        self,
        payload: bytes | bytearray | memoryview,
        stage_callback: Callable[[dict[str, Any]], None],
        *,
        _result_transform: Callable[[ReviewRunResultV3], ReviewRunResultV3] | None = None,
    ) -> ReviewExperienceEnvelopeV1:
        """Execute v3 while publishing only safe, actually observed stage boundaries."""

        self._assert_open()
        capability = self.assert_uat_capability_current()
        document_analysis_started_ns: int | None = None

        def emit_started(stage_code: str) -> int:
            started_ns = time.perf_counter_ns()
            stage_callback(
                {
                    "stage_code": stage_code,
                    "state": "started",
                    "duration_ms": None,
                    "safe_counts": {},
                }
            )
            return started_ns

        def emit_finished(
            stage_code: str,
            started_ns: int,
            *,
            state: str,
            safe_counts: dict[str, int] | None = None,
            duration_ms: int | None = None,
        ) -> None:
            stage_callback(
                {
                    "stage_code": stage_code,
                    "state": state,
                    "duration_ms": (
                        duration_ms
                        if duration_ms is not None
                        else max(
                            1,
                            min(
                                300_000,
                                round((time.perf_counter_ns() - started_ns) / 1_000_000),
                            ),
                        )
                    ),
                    "safe_counts": safe_counts or {},
                }
            )

        request_validation_started = emit_started("request_validation")
        try:
            decoded = _decode_serialized_json_object(
                payload,
                invalid_code="review_v3_request_invalid",
            )
            service_request = ServiceReviewRunRequestV3.model_validate(decoded)
            emit_finished(
                "request_validation",
                request_validation_started,
                state="completed",
                safe_counts={"documents": len(service_request.documents)},
            )
        except Exception:
            emit_finished(
                "request_validation",
                request_validation_started,
                state="failed",
            )
            raise

        public_by_internal = {
            "normalize_documents": "normalization",
            "pair_documents": "document_pairing",
            "validate_cross_documents": "cross_validation",
            "retrieve_regulations": "evidence_retrieval",
            "compliance_review": "compliance_review",
        }

        def observe(
            internal_stage: str,
            state: WorkflowStageState,
            counts: dict[str, int],
            duration_ms: int | None,
        ) -> None:
            nonlocal document_analysis_started_ns
            if internal_stage == "classify_documents" and state == "started":
                document_analysis_started_ns = emit_started("document_analysis")
                return
            if internal_stage in {"classify_documents", "extract_documents"}:
                if state == "failed" and document_analysis_started_ns is not None:
                    emit_finished(
                        "document_analysis",
                        document_analysis_started_ns,
                        state="failed",
                        safe_counts=counts,
                    )
                    document_analysis_started_ns = None
                elif (
                    internal_stage == "extract_documents"
                    and state == "completed"
                    and document_analysis_started_ns is not None
                ):
                    emit_finished(
                        "document_analysis",
                        document_analysis_started_ns,
                        state="completed",
                        safe_counts=counts,
                    )
                    document_analysis_started_ns = None
                return
            if internal_stage == "__adaptive__":
                stage_callback(
                    {
                        "stage_code": "adaptive_investigation",
                        "state": state,
                        "duration_ms": duration_ms,
                        "safe_counts": counts,
                    }
                )
                return
            public_stage = public_by_internal.get(internal_stage)
            if public_stage is None:
                return
            stage_callback(
                {
                    "stage_code": public_stage,
                    "state": state,
                    "duration_ms": duration_ms,
                    "safe_counts": counts,
                }
            )

        result = self.review_v3(service_request, _stage_observer=observe)
        if _result_transform is not None:
            result = _result_transform(result)
        envelope = build_review_experience(
            result,
            analyzer=self._analyzer,
            capability=capability,
            stage_callback=stage_callback,
        )
        result_validation_started = emit_started("result_validation")
        try:
            _assert_public_result_safe(
                envelope.model_dump(mode="python", warnings=False),
                sensitive_markers=self._secret_markers,
            )
            emit_finished(
                "result_validation",
                result_validation_started,
                state="completed",
                safe_counts={
                    "statements": sum(
                        len(section.statements) for section in envelope.user_presentation.sections
                    )
                },
            )
        except Exception:
            emit_finished(
                "result_validation",
                result_validation_started,
                state="failed",
            )
            raise
        stage_callback(
            {
                "stage_code": "completed",
                "state": "completed",
                "duration_ms": None,
                "safe_counts": {
                    "documents": len(result.document_results),
                    "issues": len(result.validation_issues),
                    "findings": len(result.regulatory_findings),
                    "evidence": len(result.rag_evidence),
                },
            }
        )
        return envelope

    def answer_v2_json(
        self,
        payload: bytes | bytearray | memoryview,
    ) -> AssistantAnswerV2:
        self._assert_open()
        self.assert_uat_capability_current()
        decoded = _decode_serialized_json_object(
            payload,
            invalid_code="answer_v2_request_invalid",
        )
        try:
            request = CaseConversationRequestV2.model_validate(decoded)
        except ValidationError:
            raise RuntimeInputError(
                "answer_v2_request_invalid",
                "The conversation request is stale or invalid.",
            ) from None
        answer = answer_case_conversation(request, analyzer=self._analyzer)
        valid_fact_refs = {
            (entry.kind, entry.reference_id) for entry in request.review_result.fact_ledger.entries
        }
        if any(
            (citation.kind, citation.reference_id) not in valid_fact_refs
            for citation in answer.citations
        ):
            raise RuntimeExecutionError(
                "answer_grounding_invalid",
                "The Assistant answer failed its canonical grounding check.",
            )
        _assert_public_result_safe(
            answer.model_dump(mode="python", warnings=False),
            sensitive_markers=self._secret_markers,
        )
        return answer

    def answer_v2_stream_json(
        self,
        payload: bytes | bytearray | memoryview,
        stage_callback: Callable[[dict[str, Any]], None],
    ) -> AssistantAnswerV2:
        """Publish safe Chat boundaries; answer prose is released only after verification."""

        self._assert_open()
        self.assert_uat_capability_current()

        def observed_stage(
            stage_code: str,
            state: Literal["started", "completed", "failed"],
            *,
            started_ns: int | None = None,
            safe_counts: dict[str, int] | None = None,
        ) -> int:
            now_ns = time.perf_counter_ns()
            stage_callback(
                {
                    "stage_code": stage_code,
                    "state": state,
                    "duration_ms": (
                        None
                        if state == "started" or started_ns is None
                        else max(
                            1,
                            min(
                                300_000,
                                round((now_ns - started_ns) / 1_000_000),
                            ),
                        )
                    ),
                    "safe_counts": safe_counts or {},
                }
            )
            return now_ns

        validation_started = observed_stage("request_validation", "started")
        try:
            decoded = _decode_serialized_json_object(
                payload,
                invalid_code="answer_v2_request_invalid",
            )
            request = CaseConversationRequestV2.model_validate(decoded)
            observed_stage(
                "request_validation",
                "completed",
                started_ns=validation_started,
                safe_counts={"turns": len(request.recent_turns)},
            )
        except Exception:
            observed_stage(
                "request_validation",
                "failed",
                started_ns=validation_started,
            )
            raise

        binding_started = observed_stage("conversation_binding", "started")
        try:
            assert_result_hash(request.review_result)
            if request.bound_result_sha256 != request.review_result.result_sha256:
                raise RuntimeConflictError(
                    "stale_conversation",
                    "The conversation is bound to a stale review result.",
                )
            observed_stage(
                "conversation_binding",
                "completed",
                started_ns=binding_started,
                safe_counts={"turns": len(request.recent_turns)},
            )
        except Exception:
            observed_stage(
                "conversation_binding",
                "failed",
                started_ns=binding_started,
            )
            raise

        answer = answer_case_conversation(
            request,
            analyzer=self._analyzer,
            stage_callback=stage_callback,
        )
        result_validation_started = observed_stage("result_validation", "started")
        try:
            valid_fact_refs = {
                (entry.kind, entry.reference_id)
                for entry in request.review_result.fact_ledger.entries
            }
            if any(
                (citation.kind, citation.reference_id) not in valid_fact_refs
                for citation in answer.citations
            ):
                raise RuntimeExecutionError(
                    "answer_grounding_invalid",
                    "The Assistant answer failed its canonical grounding check.",
                )
            _assert_public_result_safe(
                answer.model_dump(mode="python", warnings=False),
                sensitive_markers=self._secret_markers,
            )
            observed_stage(
                "result_validation",
                "completed",
                started_ns=result_validation_started,
                safe_counts={"citations": len(answer.citations)},
            )
        except Exception:
            observed_stage(
                "result_validation",
                "failed",
                started_ns=result_validation_started,
            )
            raise
        stage_callback(
            {
                "stage_code": "completed",
                "state": "completed",
                "duration_ms": None,
                "safe_counts": {"citations": len(answer.citations)},
            }
        )
        return answer

    def revision_v3_json(
        self,
        payload: bytes | bytearray | memoryview,
    ) -> ReviewRevisionResultV1:
        self._assert_open()
        decoded = _decode_serialized_json_object(payload, invalid_code="revision_request_invalid")
        raw_parent = decoded.get("parent_result")
        if isinstance(raw_parent, dict) and (
            decoded.get("parent_result_sha256") != raw_parent.get("result_sha256")
            or decoded.get("parent_input_manifest_sha256")
            != raw_parent.get("input_manifest_sha256")
        ):
            raise RuntimeConflictError(
                "stale_revision",
                "The revision parent identity is stale.",
            )
        if isinstance(raw_parent, dict) and raw_parent.get(
            "result_sha256"
        ) != review_result_v3_sha256(raw_parent):
            raise RuntimeConflictError(
                "stale_revision",
                "The revision parent result was modified after completion.",
            )
        try:
            revision = ReviewRevisionRequestV1.model_validate(decoded)
        except ValidationError:
            raise RuntimeInputError(
                "revision_request_invalid",
                "The revision request is stale or invalid.",
            ) from None
        try:
            assert_result_hash(revision.parent_result)
        except ValueError:
            raise RuntimeConflictError(
                "stale_revision",
                "The revision parent result was modified after completion.",
            ) from None
        corrections = [
            decision.correction
            for decision in revision.decisions
            if decision.correction is not None
        ]
        forced_reasons = (
            ("human_reviewer_rejected",)
            if any(decision.decision == "reject_review" for decision in revision.decisions)
            else ()
        )
        if corrections:
            child = self._review_v3_with_corrections(
                revision.revised_request,
                corrections,
                forced_human_review_reasons=forced_reasons,
            )
            _assert_public_corrections_applied(child, corrections)
        else:
            child = self.review_v3(
                revision.revised_request,
                _forced_human_review_reasons=forced_reasons,
            )
        result = ReviewRevisionResultV1(
            parent_result_sha256=revision.parent_result_sha256,
            child_result=child,
            applied_decision_ids=[item.decision_id for item in revision.decisions],
        )
        _assert_public_result_safe(
            result.model_dump(mode="python", warnings=False),
            sensitive_markers=self._secret_markers,
        )
        return result

    def _review_v3_with_corrections(
        self,
        request: ServiceReviewRunRequestV3,
        corrections: list[Any],
        *,
        forced_human_review_reasons: tuple[str, ...] = (),
    ) -> ReviewRunResultV3:
        if _has_visual_documents(request):
            raise RuntimeInputError(
                "visual_input_not_supported",
                "The V3 Production-candidate service is text-only.",
            )
        retriever = self._get_retriever_v3()
        service_request = _project_service_request_v3_to_v2(request)
        v2_result, supporting_rejections, state = self._execute_review_v3(
            service_request,
            request=request,
            retriever=retriever,
            corrections=corrections,
        )
        supporting_rejections = sorted({*supporting_rejections, *forced_human_review_reasons})
        active_executor = _RuntimeBoundedAdaptiveExecutor(
            service_request=service_request,
            request=request,
            retriever=retriever,
            analyzer=self._analyzer,
            state=state,
            result=v2_result,
            supporting_document_rejections=supporting_rejections,
            assertion_verifier=self._assertion_verifier,
        )
        return to_review_run_result_v3(
            v2_result,
            request,
            analyzer=self._analyzer,
            evidence_runtime_v3_sha256=(retriever.expected_identity.runtime_release_v3_sha256),
            assertion_trust_store=self._runtime_assertion_trust_store(),
            supporting_document_rejections=supporting_rejections,
            active_executor=active_executor,
            uat_capability=getattr(self, "_uat_capability", None),
        )


__all__ = [
    "AgentRuntime",
    "AgentRuntimeError",
    "RuntimeConfig",
    "RuntimeConfigurationError",
    "RuntimeConflictError",
    "RuntimeExecutionError",
    "RuntimeInputError",
]
