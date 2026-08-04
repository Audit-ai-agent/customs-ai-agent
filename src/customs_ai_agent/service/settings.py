"""Small loopback-only settings surface for development and functional UAT."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal, Mapping, cast

from pydantic import SecretStr

_TOKEN = re.compile(r"^[\x21-\x7e]{43,512}$")
_B64_TOKEN = re.compile(r"^[A-Za-z0-9\-._~+/]{43,504}={0,8}$")
_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO"})
_MAX_BODY_BYTES = 2 * 1024 * 1024
_MAX_DOCUMENTS = 20
_MAX_TEXT_CHARACTERS = 500_000
ServiceMode = Literal["local-development", "functional-uat"]


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


def _integer(source: Mapping[str, str], name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(source.get(name, str(default)).strip())
    except ValueError:
        raise ServiceConfigurationError(f"{name.casefold()}_invalid") from None
    if not minimum <= value <= maximum:
        raise ServiceConfigurationError(f"{name.casefold()}_invalid")
    return value


def _number(
    source: Mapping[str, str], name: str, default: float, minimum: float, maximum: float
) -> float:
    try:
        value = float(source.get(name, str(default)).strip())
    except ValueError:
        raise ServiceConfigurationError(f"{name.casefold()}_invalid") from None
    if not minimum <= value <= maximum:
        raise ServiceConfigurationError(f"{name.casefold()}_invalid")
    return value


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    """All service traffic stays on loopback and uses one peer-mode handshake."""

    bearer_token: SecretStr
    evidence_base_url: str
    evidence_service_token: SecretStr
    service_mode: ServiceMode = "local-development"
    host: str = "127.0.0.1"
    # Backend owns 8080.  Keeping Agent on 8090 makes the sibling default
    # topology work without a per-developer port override.
    port: int = 8090
    max_body_bytes: int = _MAX_BODY_BYTES
    max_documents: int = _MAX_DOCUMENTS
    max_text_characters: int = _MAX_TEXT_CHARACTERS
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

    def __post_init__(self) -> None:
        service_token = self.bearer_token.get_secret_value()
        evidence_token = self.evidence_service_token.get_secret_value()
        if _TOKEN.fullmatch(service_token) is None or len(set(service_token)) < 12:
            raise ServiceConfigurationError("service_bearer_token_weak")
        if _B64_TOKEN.fullmatch(evidence_token) is None or len(set(evidence_token)) < 12:
            raise ServiceConfigurationError("evidence_service_token_weak")
        if self.service_mode not in {"local-development", "functional-uat"}:
            raise ServiceConfigurationError("customs_agent_service_mode_invalid")
        if self.host != "127.0.0.1":
            raise ServiceConfigurationError("local_profile_bind_host_not_loopback")
        if not self.evidence_base_url.strip() or any(
            char.isspace() for char in self.evidence_base_url
        ):
            raise ServiceConfigurationError("evidence_service_base_url_invalid")
        if not 1 <= self.port <= 65_535:
            raise ServiceConfigurationError("customs_agent_port_invalid")
        if not 1_024 <= self.max_body_bytes <= 32 * 1024 * 1024:
            raise ServiceConfigurationError("customs_agent_max_body_bytes_invalid")
        if not 1 <= self.max_documents <= _MAX_DOCUMENTS:
            raise ServiceConfigurationError("customs_agent_max_documents_invalid")
        if not 1 <= self.max_text_characters <= _MAX_TEXT_CHARACTERS:
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
        *,
        service_mode_override: ServiceMode | None = None,
    ) -> "ServiceSettings":
        source = os.environ if environ is None else environ
        configured_mode = (
            service_mode_override
            or source.get("CUSTOMS_AGENT_SERVICE_MODE", "local-development").strip().casefold()
        )
        if configured_mode not in {"local-development", "functional-uat"}:
            raise ServiceConfigurationError("customs_agent_service_mode_invalid")
        if (
            source.get("EVIDENCE_SERVICE_TOKEN_HEADER", "Authorization").strip().casefold()
            != "authorization"
        ):
            raise ServiceConfigurationError("local_profile_evidence_bearer_required")
        return cls(
            bearer_token=SecretStr(_required(source, "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN")),
            evidence_base_url=_required(source, "EVIDENCE_SERVICE_BASE_URL"),
            evidence_service_token=SecretStr(_required(source, "EVIDENCE_SERVICE_TOKEN")),
            service_mode=cast(ServiceMode, configured_mode),
            host=source.get("CUSTOMS_AGENT_HOST", "127.0.0.1").strip(),
            port=_integer(source, "CUSTOMS_AGENT_PORT", 8090, 1, 65_535),
            max_body_bytes=_integer(
                source, "CUSTOMS_AGENT_MAX_BODY_BYTES", _MAX_BODY_BYTES, 1_024, 32 * 1024 * 1024
            ),
            max_documents=_integer(
                source, "CUSTOMS_AGENT_MAX_DOCUMENTS", _MAX_DOCUMENTS, 1, _MAX_DOCUMENTS
            ),
            max_text_characters=_integer(
                source,
                "CUSTOMS_AGENT_MAX_TEXT_CHARACTERS",
                _MAX_TEXT_CHARACTERS,
                1,
                _MAX_TEXT_CHARACTERS,
            ),
            body_read_timeout_seconds=_number(
                source, "CUSTOMS_AGENT_BODY_READ_TIMEOUT_SECONDS", 15, 0.1, 120
            ),
            max_concurrency=_integer(source, "CUSTOMS_AGENT_MAX_CONCURRENCY", 2, 1, 32),
            queue_wait_seconds=_number(source, "CUSTOMS_AGENT_QUEUE_WAIT_SECONDS", 1, 0.01, 60),
            run_timeout_seconds=_number(source, "CUSTOMS_AGENT_RUN_TIMEOUT_SECONDS", 420, 1, 3_600),
            replay_ttl_seconds=_number(source, "CUSTOMS_AGENT_REPLAY_TTL_SECONDS", 600, 1, 86_400),
            replay_max_entries=_integer(
                source, "CUSTOMS_AGENT_REPLAY_MAX_ENTRIES", 1_000, 1, 10_000
            ),
            replay_max_bytes=_integer(
                source,
                "CUSTOMS_AGENT_REPLAY_MAX_BYTES",
                64 * 1024 * 1024,
                1 * 1024 * 1024,
                256 * 1024 * 1024,
            ),
            readiness_ttl_seconds=_number(
                source, "CUSTOMS_AGENT_READINESS_TTL_SECONDS", 5, 0.1, 60
            ),
            readiness_timeout_seconds=_number(
                source, "CUSTOMS_AGENT_READINESS_TIMEOUT_SECONDS", 5, 0.1, 120
            ),
            shutdown_grace_seconds=_number(
                source, "CUSTOMS_AGENT_SHUTDOWN_GRACE_SECONDS", 30, 1, 900
            ),
            log_level=source.get("CUSTOMS_AGENT_LOG_LEVEL", "INFO").strip().upper(),
        )


__all__ = ["ServiceConfigurationError", "ServiceMode", "ServiceSettings"]
