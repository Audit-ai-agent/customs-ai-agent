from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Literal

from jsonschema import Draft202012Validator, FormatChecker
from pydantic import SecretStr, ValidationError

from customs_ai_agent.adapters.llm import (
    ClaudeDocumentAnalyzer,
    ClaudeSettings,
    LLMError,
)
from customs_ai_agent.adapters.ocr import OcrLayoutProvider, OcrLayoutProviderError
from customs_ai_agent.adapters.retrieval import (
    HttpRegulationRetriever,
    RetrievalError,
    validate_evidence_destination,
)
from customs_ai_agent.contracts import get_schema
from customs_ai_agent.ocr_release import ReleaseLockedOcrProvider
from customs_ai_agent.runner import run_review_result_v2
from customs_ai_agent.schemas.models import (
    ReviewRunRequestV2,
    ReviewRunResultV2,
    ServiceReviewRunRequestV2,
)


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
    # This is a public HTTP header name, never a credential value.
    evidence_token_header: str = "X-Demo-Token"  # noqa: S105
    evidence_allowed_origins: frozenset[str] = field(default_factory=frozenset)
    evidence_timeout_seconds: float = 60.0
    evidence_max_retries: int = 1
    # The service-facing runtime never permits the non-release compatibility
    # mode. Evaluation-only callers may compose HttpRegulationRetriever
    # directly with an explicit optional mode, outside AgentRuntime.
    evidence_contract_identity_mode: Literal["required"] = "required"

    def __post_init__(self) -> None:
        if not isinstance(self.claude_settings, ClaudeSettings):
            raise RuntimeConfigurationError(
                "claude_settings_invalid",
                "RuntimeConfig requires validated ClaudeSettings.",
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
        if self.evidence_service_token is not None and not isinstance(
            self.evidence_service_token,
            SecretStr,
        ):
            raise RuntimeConfigurationError(
                "evidence_service_token_invalid",
                "Evidence credentials must be supplied as SecretStr.",
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
        if self.evidence_contract_identity_mode != "required":
            raise RuntimeConfigurationError(
                "evidence_contract_identity_mode_invalid",
                "Service Runtime requires Evidence contract identity verification.",
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
        # This is a public HTTP header name, never a credential value.
        evidence_token_header: str = "X-Demo-Token",  # noqa: S107
        evidence_allowed_origins: frozenset[str] = frozenset(),
        evidence_timeout_seconds: float = 60.0,
        evidence_max_retries: int = 1,
        evidence_contract_identity_mode: Literal["required"] = "required",
    ) -> RuntimeConfig:
        """Load only allowlisted Claude settings; Evidence settings stay explicit."""

        return cls(
            claude_settings=ClaudeSettings.from_environment(env_file),
            evidence_base_url=evidence_base_url,
            ocr_layout_provider=ocr_layout_provider,
            evidence_service_token=evidence_service_token,
            evidence_token_header=evidence_token_header,
            evidence_allowed_origins=evidence_allowed_origins,
            evidence_timeout_seconds=evidence_timeout_seconds,
            evidence_max_retries=evidence_max_retries,
            evidence_contract_identity_mode=evidence_contract_identity_mode,
        )


def _has_visual_documents(request: ServiceReviewRunRequestV2) -> bool:
    return any(
        document.image_pages or document.pdf_pages is not None
        for document in (request.documents or [])
    )


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


class AgentRuntime:
    """Service-facing Claude + OCR + Evidence HTTP composition root."""

    __slots__ = (
        "_analyzer",
        "_closed",
        "_config",
        "_retriever",
        "_secret_markers",
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
        dependency_configuration_failed = False
        try:
            retriever = HttpRegulationRetriever(
                config.evidence_base_url,
                service_token=token,
                token_header=config.evidence_token_header,
                allowed_origins=config.evidence_allowed_origins,
                timeout_seconds=config.evidence_timeout_seconds,
                max_retries=config.evidence_max_retries,
                contract_identity_mode=config.evidence_contract_identity_mode,
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
        self._analyzer = analyzer
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

        sensitive_markers = self._secret_markers + _input_base64_markers(service_request)
        execution_failure: RuntimeExecutionError | None = None
        try:
            workflow_request = ReviewRunRequestV2.model_validate(
                service_request.model_dump(mode="python", warnings=False)
            )
            result = run_review_result_v2(
                workflow_request,
                retriever=self._retriever,
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


__all__ = [
    "AgentRuntime",
    "AgentRuntimeError",
    "RuntimeConfig",
    "RuntimeConfigurationError",
    "RuntimeExecutionError",
    "RuntimeInputError",
]
