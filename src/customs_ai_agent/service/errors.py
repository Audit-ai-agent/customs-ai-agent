from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


_MESSAGES = {
    "authentication_invalid": "Service authentication failed.",
    "authentication_required": "Service authentication is required.",
    "content_encoding_unsupported": "The request content encoding is not supported.",
    "content_length_invalid": "The request content length is invalid.",
    "document_decode_invalid": "The document could not be safely decoded for preview.",
    "document_hash_mismatch": "The document integrity check failed.",
    "document_image_limits_exceeded": "The document image exceeds the configured safety limits.",
    "document_magic_mismatch": "The document bytes do not match the declared media type.",
    "document_page_limit_exceeded": "The document exceeds the configured page limit.",
    "document_preview_failed": "The local document preview could not be completed.",
    "document_preview_timeout": "The local document preview exceeded the execution deadline.",
    "document_request_invalid": "The document preview request is invalid.",
    "document_too_large": "The document exceeds the service limit.",
    "dependency_unready": "A required service dependency is not ready.",
    "idempotency_conflict": "The review run identifier is already bound to another request.",
    "internal_error": "The request could not be processed.",
    "knowledge_input_boundary": (
        "통관 지식 도우미에는 사건·심사·문서 원문을 입력할 수 없습니다. "
        "일반적인 통관 지식 질문을 한 문장으로 입력해 주세요."
    ),
    "request_invalid": "The review request is invalid.",
    "request_body_timeout": "The review request body exceeded the receive deadline.",
    "request_too_large": "The review request exceeds the service limit.",
    "required_field_missing": "A required service field is missing.",
    "review_execution_failed": "The review could not be completed.",
    "review_timeout": "The review exceeded the service execution deadline.",
    "service_busy": "The service is at its configured processing capacity.",
    "service_draining": "The service is shutting down.",
    "service_mode_mismatch": "The caller service mode does not match this Agent profile.",
    "unsupported_media_type": "The request media type is not supported.",
    "visual_input_not_supported": "This service accepts text documents only.",
}


@dataclass(frozen=True, slots=True)
class ServiceHttpError(Exception):
    code: str
    status_code: int
    retryable: bool = False
    retry_after_seconds: int | None = None
    execution_state: Literal["not-started"] | None = None

    def __post_init__(self) -> None:
        if self.execution_state is not None and (
            self.execution_state != "not-started"
            or self.code != "dependency_unready"
            or self.status_code != 503
            or not self.retryable
        ):
            raise ValueError("not-started is reserved for retryable dependency admission failures")

    @property
    def public_message(self) -> str:
        return _MESSAGES.get(self.code, _MESSAGES["internal_error"])


__all__ = ["ServiceHttpError"]
