from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Mapping

from pydantic import SecretStr

from customs_ai_agent.transport_limits import (
    TEXT_FIRST_MAX_BODY_BYTES,
    TEXT_FIRST_MAX_DOCUMENTS,
    TEXT_FIRST_MAX_TEXT_CHARACTERS,
)


_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[\x21-\x7e]{43,512}$")
_BEARER_TOKEN68 = re.compile(r"^[A-Za-z0-9\-._~+/]{43,504}={0,8}$")
_HEADER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9-]{0,63}$")
_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO"})


class ServiceConfigurationError(RuntimeError):
    """A content-free startup configuration error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _required(source: Mapping[str, str], name: str) -> str:
    value = source.get(name, "").strip()
    if not value:
        raise ServiceConfigurationError(f"{name.casefold()}_missing")
    return value


def _integer(
    source: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = source.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        raise ServiceConfigurationError(f"{name.casefold()}_invalid") from None
    if not minimum <= value <= maximum:
        raise ServiceConfigurationError(f"{name.casefold()}_invalid")
    return value


def _number(
    source: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = source.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        raise ServiceConfigurationError(f"{name.casefold()}_invalid") from None
    if not minimum <= value <= maximum:
        raise ServiceConfigurationError(f"{name.casefold()}_invalid")
    return value


def _boolean(
    source: Mapping[str, str],
    name: str,
    default: bool,
) -> bool:
    raw = source.get(name, "true" if default else "false").strip().casefold()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ServiceConfigurationError(f"{name.casefold()}_invalid")


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    """Validated settings for one Agent service process.

    The service token protects this internal process. End-user identity and
    authorization remain Backend responsibilities.
    """

    bearer_token: SecretStr
    evidence_base_url: str
    evidence_allowed_origins: frozenset[str] = frozenset()
    evidence_service_token: SecretStr | None = None
    evidence_token_header: str = "Authorization"  # noqa: S105 - header name, not a secret
    environment: str = "development"
    build_sha: str = "unknown"
    wheel_sha256: str = "unknown"
    host: str = "127.0.0.1"
    port: int = 8080
    max_body_bytes: int = TEXT_FIRST_MAX_BODY_BYTES
    max_documents: int = TEXT_FIRST_MAX_DOCUMENTS
    max_text_characters: int = TEXT_FIRST_MAX_TEXT_CHARACTERS
    body_read_timeout_seconds: float = 15.0
    max_concurrency: int = 2
    queue_wait_seconds: float = 1.0
    run_timeout_seconds: float = 420.0
    replay_ttl_seconds: float = 600.0
    replay_max_entries: int = 1_000
    replay_max_bytes: int = 64 * 1024 * 1024
    readiness_ttl_seconds: float = 5.0
    readiness_timeout_seconds: float = 5.0
    shutdown_grace_seconds: float = 30.0
    log_level: str = "INFO"
    enhanced_agent_experience: bool = False
    uat_adaptive_active: bool = False

    def __post_init__(self) -> None:
        token = self.bearer_token.get_secret_value()
        if (
            _TOKEN.fullmatch(token) is None
            or len(set(token)) < 12
            or token.casefold() in {"changeme", "placeholder", "development"}
        ):
            raise ServiceConfigurationError("service_bearer_token_weak")
        if not self.evidence_base_url.strip():
            raise ServiceConfigurationError("evidence_service_base_url_missing")
        if _HEADER_NAME.fullmatch(self.evidence_token_header) is None:
            raise ServiceConfigurationError("evidence_service_token_header_invalid")
        evidence_token = (
            self.evidence_service_token.get_secret_value()
            if self.evidence_service_token is not None
            else None
        )
        if self.evidence_token_header.casefold() == "authorization" and evidence_token is not None:
            if _BEARER_TOKEN68.fullmatch(evidence_token) is None or len(set(evidence_token)) < 12:
                raise ServiceConfigurationError("evidence_service_token_weak")
        if self.environment not in {"development", "test", "production"}:
            raise ServiceConfigurationError("customs_agent_environment_invalid")
        if self.environment == "production":
            if self.enhanced_agent_experience or self.uat_adaptive_active:
                raise ServiceConfigurationError("production_uat_capability_forbidden")
            if self.evidence_token_header.casefold() != "authorization":
                raise ServiceConfigurationError("production_evidence_auth_must_use_bearer")
            if evidence_token is None:
                raise ServiceConfigurationError("evidence_service_token_missing")
            if _GIT_SHA.fullmatch(self.build_sha) is None:
                raise ServiceConfigurationError("customs_agent_build_sha_invalid")
            if _SHA256.fullmatch(self.wheel_sha256) is None:
                raise ServiceConfigurationError("customs_agent_wheel_sha256_invalid")
            if (
                self.max_body_bytes != TEXT_FIRST_MAX_BODY_BYTES
                or self.max_documents != TEXT_FIRST_MAX_DOCUMENTS
                or self.max_text_characters != TEXT_FIRST_MAX_TEXT_CHARACTERS
            ):
                raise ServiceConfigurationError("production_transport_limits_mismatch")
        elif self.build_sha != "unknown" and _GIT_SHA.fullmatch(self.build_sha) is None:
            raise ServiceConfigurationError("customs_agent_build_sha_invalid")
        elif self.wheel_sha256 != "unknown" and _SHA256.fullmatch(self.wheel_sha256) is None:
            raise ServiceConfigurationError("customs_agent_wheel_sha256_invalid")
        if not self.host or any(character.isspace() for character in self.host):
            raise ServiceConfigurationError("customs_agent_host_invalid")
        if not 1 <= self.port <= 65_535:
            raise ServiceConfigurationError("customs_agent_port_invalid")
        if not 1_024 <= self.max_body_bytes <= 32 * 1024 * 1024:
            raise ServiceConfigurationError("customs_agent_max_body_bytes_invalid")
        if not 1 <= self.max_documents <= TEXT_FIRST_MAX_DOCUMENTS:
            raise ServiceConfigurationError("customs_agent_max_documents_invalid")
        if not 1 <= self.max_text_characters <= TEXT_FIRST_MAX_TEXT_CHARACTERS:
            raise ServiceConfigurationError("customs_agent_max_text_characters_invalid")
        if not 0.1 <= self.body_read_timeout_seconds <= 120:
            raise ServiceConfigurationError("customs_agent_body_read_timeout_seconds_invalid")
        if not 1 <= self.max_concurrency <= 32:
            raise ServiceConfigurationError("customs_agent_max_concurrency_invalid")
        if not 0.01 <= self.queue_wait_seconds <= 60:
            raise ServiceConfigurationError("customs_agent_queue_wait_seconds_invalid")
        if not 1 <= self.run_timeout_seconds <= 3_600:
            raise ServiceConfigurationError("customs_agent_run_timeout_seconds_invalid")
        if not 1 <= self.replay_ttl_seconds <= 86_400:
            raise ServiceConfigurationError("customs_agent_replay_ttl_seconds_invalid")
        if not 1 <= self.replay_max_entries <= 10_000:
            raise ServiceConfigurationError("customs_agent_replay_max_entries_invalid")
        if not 1 * 1024 * 1024 <= self.replay_max_bytes <= 256 * 1024 * 1024:
            raise ServiceConfigurationError("customs_agent_replay_max_bytes_invalid")
        if not 0.1 <= self.readiness_ttl_seconds <= 60:
            raise ServiceConfigurationError("customs_agent_readiness_ttl_seconds_invalid")
        if not 0.1 <= self.readiness_timeout_seconds <= 120:
            raise ServiceConfigurationError("customs_agent_readiness_timeout_seconds_invalid")
        if not 1 <= self.shutdown_grace_seconds <= 900:
            raise ServiceConfigurationError("customs_agent_shutdown_grace_seconds_invalid")
        if self.log_level not in _LOG_LEVELS:
            raise ServiceConfigurationError("customs_agent_log_level_invalid")

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> ServiceSettings:
        source = os.environ if environ is None else environ
        raw_token = _required(source, "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN")
        evidence_url = _required(source, "EVIDENCE_SERVICE_BASE_URL")
        allowed_origins = frozenset(
            value.strip()
            for value in source.get("EVIDENCE_SERVICE_ALLOWED_ORIGINS", "").split(",")
            if value.strip()
        )
        evidence_token = source.get("EVIDENCE_SERVICE_TOKEN", "").strip()
        return cls(
            bearer_token=SecretStr(raw_token),
            evidence_base_url=evidence_url,
            evidence_allowed_origins=allowed_origins,
            evidence_service_token=SecretStr(evidence_token) if evidence_token else None,
            evidence_token_header=source.get(
                "EVIDENCE_SERVICE_TOKEN_HEADER",
                "Authorization",
            ).strip(),
            environment=source.get("CUSTOMS_AGENT_ENVIRONMENT", "development").strip().casefold(),
            build_sha=source.get("CUSTOMS_AGENT_BUILD_SHA", "unknown").strip().casefold(),
            wheel_sha256=source.get("CUSTOMS_AGENT_WHEEL_SHA256", "unknown").strip().casefold(),
            host=source.get("CUSTOMS_AGENT_HOST", "127.0.0.1").strip(),
            port=_integer(source, "CUSTOMS_AGENT_PORT", 8080, minimum=1, maximum=65_535),
            max_body_bytes=_integer(
                source,
                "CUSTOMS_AGENT_MAX_BODY_BYTES",
                TEXT_FIRST_MAX_BODY_BYTES,
                minimum=1_024,
                maximum=32 * 1024 * 1024,
            ),
            max_documents=_integer(
                source,
                "CUSTOMS_AGENT_MAX_DOCUMENTS",
                TEXT_FIRST_MAX_DOCUMENTS,
                minimum=1,
                maximum=TEXT_FIRST_MAX_DOCUMENTS,
            ),
            max_text_characters=_integer(
                source,
                "CUSTOMS_AGENT_MAX_TEXT_CHARACTERS",
                TEXT_FIRST_MAX_TEXT_CHARACTERS,
                minimum=1,
                maximum=TEXT_FIRST_MAX_TEXT_CHARACTERS,
            ),
            body_read_timeout_seconds=_number(
                source,
                "CUSTOMS_AGENT_BODY_READ_TIMEOUT_SECONDS",
                15,
                minimum=0.1,
                maximum=120,
            ),
            max_concurrency=_integer(
                source,
                "CUSTOMS_AGENT_MAX_CONCURRENCY",
                2,
                minimum=1,
                maximum=32,
            ),
            queue_wait_seconds=_number(
                source,
                "CUSTOMS_AGENT_QUEUE_WAIT_SECONDS",
                1,
                minimum=0.01,
                maximum=60,
            ),
            run_timeout_seconds=_number(
                source,
                "CUSTOMS_AGENT_RUN_TIMEOUT_SECONDS",
                420,
                minimum=1,
                maximum=3_600,
            ),
            replay_ttl_seconds=_number(
                source,
                "CUSTOMS_AGENT_REPLAY_TTL_SECONDS",
                600,
                minimum=1,
                maximum=86_400,
            ),
            replay_max_entries=_integer(
                source,
                "CUSTOMS_AGENT_REPLAY_MAX_ENTRIES",
                1_000,
                minimum=1,
                maximum=10_000,
            ),
            replay_max_bytes=_integer(
                source,
                "CUSTOMS_AGENT_REPLAY_MAX_BYTES",
                64 * 1024 * 1024,
                minimum=1 * 1024 * 1024,
                maximum=256 * 1024 * 1024,
            ),
            readiness_ttl_seconds=_number(
                source,
                "CUSTOMS_AGENT_READINESS_TTL_SECONDS",
                5,
                minimum=0.1,
                maximum=60,
            ),
            readiness_timeout_seconds=_number(
                source,
                "CUSTOMS_AGENT_READINESS_TIMEOUT_SECONDS",
                5,
                minimum=0.1,
                maximum=120,
            ),
            shutdown_grace_seconds=_number(
                source,
                "CUSTOMS_AGENT_SHUTDOWN_GRACE_SECONDS",
                30,
                minimum=1,
                maximum=900,
            ),
            log_level=source.get("CUSTOMS_AGENT_LOG_LEVEL", "INFO").strip().upper(),
            enhanced_agent_experience=_boolean(
                source,
                "CUSTOMS_ENHANCED_AGENT_EXPERIENCE",
                False,
            ),
            uat_adaptive_active=_boolean(
                source,
                "CUSTOMS_UAT_ADAPTIVE_ACTIVE",
                False,
            ),
        )


__all__ = [
    "ServiceConfigurationError",
    "ServiceSettings",
]
