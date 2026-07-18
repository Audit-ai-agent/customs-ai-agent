from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Literal

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
from customs_ai_agent.ocr_release import ReleaseLockedOcrProvider
from customs_ai_agent.runner import run_review_result_v2
from customs_ai_agent.schemas.models import ReviewRunRequestV2, ReviewRunResultV2


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
            try:
                self.ocr_layout_provider.assert_service_ready()
            except OcrLayoutProviderError as exc:
                raise RuntimeConfigurationError(
                    exc.code,
                    "The configured OCR release Gate is not ready.",
                ) from None
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
        try:
            normalized_url = validate_evidence_destination(
                self.evidence_base_url,
                self.evidence_allowed_origins,
            )
        except (TypeError, ValueError):
            raise RuntimeConfigurationError(
                "evidence_base_url_invalid",
                "Evidence Service base URL is invalid or not explicitly allowed.",
            ) from None
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


def _has_visual_documents(request: ReviewRunRequestV2) -> bool:
    return any(
        document.image_pages or document.pdf_pages is not None
        for document in (request.documents or [])
    )


def _input_base64_markers(request: ReviewRunRequestV2) -> tuple[str, ...]:
    markers: list[str] = []
    for document in request.documents or []:
        pages = document.pdf_pages.pages if document.pdf_pages is not None else document.image_pages
        markers.extend(page.data_base64 for page in pages)
    return tuple(markers)


def _assert_service_request(request: ReviewRunRequestV2) -> None:
    if request.scenario_id is not None or request.documents is None or request.case_id is None:
        raise RuntimeInputError(
            "service_request_identity_required",
            "Service reviews require an explicit case and document set.",
        )


def _assert_result_correlated(
    request: ReviewRunRequestV2,
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
        normalized_keys = {str(key).strip().casefold().replace("-", "_") for key in value}
        if normalized_keys.intersection(_FORBIDDEN_RESULT_KEYS):
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
            if marker and (value == marker or (len(marker) >= 16 and marker in value)):
                raise RuntimeExecutionError(
                    "public_result_safety_violation",
                    "Sensitive input material reached the public Agent result.",
                )


class AgentRuntime:
    """Service-facing Claude + OCR + Evidence HTTP composition root."""

    __slots__ = (
        "_analyzer",
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
        except LLMError:
            raise
        except Exception:
            raise RuntimeConfigurationError(
                "runtime_dependency_configuration_failed",
                "Claude, OCR, or Evidence runtime configuration is invalid.",
            ) from None
        self._config = config
        self._retriever = retriever
        self._analyzer = analyzer
        self._secret_markers = tuple(
            marker
            for marker in (
                config.claude_settings.api_key.get_secret_value(),
                token,
            )
            if marker
        )

    def review(
        self,
        request: ReviewRunRequestV2 | dict[str, Any],
    ) -> ReviewRunResultV2:
        """Run one real V2 review and return only the redacted public contract."""

        if not isinstance(request, (ReviewRunRequestV2, dict)):
            raise RuntimeInputError(
                "review_request_invalid",
                "The Agent runtime accepts ReviewRunRequestV2 or a request object.",
            )
        # A Pydantic instance can have been created with ``model_construct`` or
        # mutated after validation. Re-parse a detached dump at the service
        # boundary instead of trusting the caller-owned object.
        request_payload = (
            request.model_dump(mode="python", warnings=False)
            if isinstance(request, ReviewRunRequestV2)
            else request
        )
        try:
            parsed_request = ReviewRunRequestV2.model_validate(request_payload)
        except ValidationError:
            raise RuntimeInputError(
                "review_request_invalid",
                "The V2 Agent request does not satisfy the public contract.",
            ) from None

        _assert_service_request(parsed_request)
        if parsed_request.llm_mode != "llm":
            raise RuntimeInputError(
                "real_llm_mode_required",
                "The service runtime accepts only explicit llm mode requests.",
            )
        if _has_visual_documents(parsed_request) and self._config.ocr_layout_provider is None:
            raise RuntimeConfigurationError(
                "ocr_layout_provider_required",
                "Image and PDF reviews require an explicitly configured OCR/layout provider.",
            )

        sensitive_markers = self._secret_markers + _input_base64_markers(parsed_request)
        try:
            result = run_review_result_v2(
                parsed_request,
                retriever=self._retriever,
                analyzer=self._analyzer,
            )
            if not isinstance(result, ReviewRunResultV2):
                raise RuntimeExecutionError(
                    "review_result_invalid",
                    "The Agent workflow did not return the V2 public result contract.",
                )
            try:
                # Re-validate even a model instance so a bypassed or mutated
                # result can never cross the public service boundary.
                result = ReviewRunResultV2.model_validate(
                    result.model_dump(mode="python", warnings=False)
                )
            except ValidationError:
                raise RuntimeExecutionError(
                    "review_result_invalid",
                    "The Agent workflow returned an invalid V2 public result.",
                ) from None
            _assert_result_correlated(parsed_request, result)
            _assert_public_result_safe(
                result.model_dump(mode="python"),
                sensitive_markers=sensitive_markers,
            )
            return result
        except (AgentRuntimeError, LLMError, OcrLayoutProviderError, RetrievalError):
            raise
        except Exception:
            raise RuntimeExecutionError(
                "review_execution_failed",
                "The Agent review failed without exposing private execution details.",
            ) from None


__all__ = [
    "AgentRuntime",
    "AgentRuntimeError",
    "RuntimeConfig",
    "RuntimeConfigurationError",
    "RuntimeExecutionError",
    "RuntimeInputError",
]
