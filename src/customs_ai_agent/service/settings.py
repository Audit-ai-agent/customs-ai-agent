"""Small loopback-only settings surface for development and functional UAT."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, cast

from pydantic import SecretStr

_TOKEN = re.compile(r"^[\x21-\x7e]{43,512}$")
_B64_TOKEN = re.compile(r"^[A-Za-z0-9\-._~+/]{43,504}={0,8}$")
_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO"})
_MAX_BODY_BYTES = 2 * 1024 * 1024
_MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
_MAX_DOCUMENT_PAGES = 50
_MAX_DOCUMENT_IMAGE_PIXELS = 40_000_000
_MAX_DOCUMENTS = 20
_MAX_TEXT_CHARACTERS = 500_000
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
ServiceMode = Literal["local-development", "functional-uat"]
GuidanceRegistryVersion = Literal["1.0", "2.0"]
# This is intentionally the Data-owned local profile name, rather than an
# Agent runtime profile name.  The composed V2 + law-history Agent profile
# therefore resolves to the same expected Evidence profile as review-only
# law-history Functional-UAT.
EvidenceProfile = Literal[
    "dev-fixture",
    "functional-uat-3law",
    "functional-uat-law-history",
]
_EVIDENCE_PROFILES = frozenset({"dev-fixture", "functional-uat-3law", "functional-uat-law-history"})
_TEMPORAL_EVIDENCE_PROFILES = frozenset({"functional-uat-law-history"})


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


def _document_preview_approval_path(value: str) -> Path:
    """Accept only the fixed Git-excluded active-pointer location.

    The runtime repeats this check with symlink and regular-file validation.
    Settings deliberately do not require the file to exist, so an unavailable
    optional preview profile cannot stop the ordinary text-only service.
    """

    path = Path(value)
    if (
        not path.is_absolute()
        or path.name != "pdf-preview-active.json"
        or path.parent.name != "approvals"
        or tuple(part.casefold() for part in path.parent.parent.parts[-3:])
        != ("_local", "data", "document-preview")
    ):
        raise ServiceConfigurationError("customs_agent_document_preview_approval_path_invalid")
    return path


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    """All service traffic stays on loopback and uses one peer-mode handshake.

    ``evidence_*`` always belongs to document-review retrieval.  A guidance
    peer is opt-in as a complete URL/token pair.  Functional-UAT guidance
    profiles also declare their registry version so a V2 scope request cannot
    silently land on the preserved V1 peer.
    """

    bearer_token: SecretStr
    evidence_base_url: str
    evidence_service_token: SecretStr
    # The local importer sets this non-secret Data profile identity.  It is
    # optional only so existing manually configured legacy environments and
    # immutable historical test fixtures remain readable.
    evidence_profile: EvidenceProfile | None = None
    guidance_base_url: str | None = None
    guidance_service_token: SecretStr | None = None
    guidance_registry_version: GuidanceRegistryVersion | None = None
    service_mode: ServiceMode = "local-development"
    host: str = "127.0.0.1"
    # Backend owns 18081 and Data fixture owns 18083.  18090 avoids the
    # Windows-reserved 80xx range while remaining loopback-only.
    port: int = 18090
    max_body_bytes: int = _MAX_BODY_BYTES
    # The preview endpoint has an isolated binary budget.  It is intentionally
    # separate from the small JSON text-review budget so enabling preview never
    # broadens the review request surface.
    max_document_bytes: int = _MAX_DOCUMENT_BYTES
    max_document_pages: int = _MAX_DOCUMENT_PAGES
    max_document_image_pixels: int = _MAX_DOCUMENT_IMAGE_PIXELS
    # The sole production opt-in for document previews.  Both values are
    # required; there is intentionally no environment setting for an engine,
    # executable, model path, network source, or automatic fallback.
    document_preview_approval_path: Path | None = None
    document_preview_approval_sha256: str | None = None
    max_documents: int = _MAX_DOCUMENTS
    max_text_characters: int = _MAX_TEXT_CHARACTERS
    body_read_timeout_seconds: float = 15.0
    max_concurrency: int = 2
    queue_wait_seconds: float = 1.0
    run_timeout_seconds: float = 420.0
    replay_ttl_seconds: float = 600.0
    replay_max_entries: int = 1_000
    readiness_timeout_seconds: float = 5.0
    evidence_retrieve_timeout_seconds: float = 90.0
    shutdown_grace_seconds: float = 30.0
    log_level: str = "INFO"

    @property
    def requires_temporal_context(self) -> bool:
        """Whether this selected Evidence profile must return temporal provenance."""

        return self.evidence_profile in _TEMPORAL_EVIDENCE_PROFILES

    def __post_init__(self) -> None:
        service_token = self.bearer_token.get_secret_value()
        evidence_token = self.evidence_service_token.get_secret_value()
        has_guidance_base_url = self.guidance_base_url is not None
        has_guidance_token = self.guidance_service_token is not None
        has_guidance_registry_version = self.guidance_registry_version is not None
        if _TOKEN.fullmatch(service_token) is None or len(set(service_token)) < 12:
            raise ServiceConfigurationError("service_bearer_token_weak")
        if _B64_TOKEN.fullmatch(evidence_token) is None or len(set(evidence_token)) < 12:
            raise ServiceConfigurationError("evidence_service_token_weak")
        if self.evidence_profile not in {None, *_EVIDENCE_PROFILES}:
            raise ServiceConfigurationError("customs_agent_evidence_profile_invalid")
        has_preview_approval_path = self.document_preview_approval_path is not None
        has_preview_approval_sha256 = self.document_preview_approval_sha256 is not None
        if has_preview_approval_path != has_preview_approval_sha256:
            raise ServiceConfigurationError("customs_agent_document_preview_approval_incomplete")
        if has_preview_approval_path:
            assert self.document_preview_approval_path is not None
            assert self.document_preview_approval_sha256 is not None
            _document_preview_approval_path(str(self.document_preview_approval_path))
            if _SHA256.fullmatch(self.document_preview_approval_sha256) is None:
                raise ServiceConfigurationError("customs_agent_document_preview_approval_sha256_invalid")
        if has_guidance_base_url != has_guidance_token:
            raise ServiceConfigurationError("guidance_service_peer_incomplete")
        if has_guidance_registry_version and not has_guidance_base_url:
            raise ServiceConfigurationError("guidance_registry_version_without_peer")
        if self.guidance_registry_version not in {None, "1.0", "2.0"}:
            raise ServiceConfigurationError("guidance_registry_version_invalid")
        if has_guidance_base_url:
            assert self.guidance_base_url is not None
            assert self.guidance_service_token is not None
            guidance_token = self.guidance_service_token.get_secret_value()
            if not self.guidance_base_url.strip() or any(
                char.isspace() for char in self.guidance_base_url
            ):
                raise ServiceConfigurationError("guidance_service_base_url_invalid")
            if _B64_TOKEN.fullmatch(guidance_token) is None or len(set(guidance_token)) < 12:
                raise ServiceConfigurationError("guidance_service_token_weak")
        if self.service_mode not in {"local-development", "functional-uat"}:
            raise ServiceConfigurationError("customs_agent_service_mode_invalid")
        if (
            self.service_mode == "functional-uat"
            and has_guidance_base_url
            and self.guidance_registry_version is None
        ):
            raise ServiceConfigurationError("guidance_registry_version_peer_incomplete")
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
        if not 1_024 <= self.max_document_bytes <= 32 * 1024 * 1024:
            raise ServiceConfigurationError("customs_agent_max_document_bytes_invalid")
        if not 1 <= self.max_document_pages <= 1_000:
            raise ServiceConfigurationError("customs_agent_max_document_pages_invalid")
        if not 1_000_000 <= self.max_document_image_pixels <= 100_000_000:
            raise ServiceConfigurationError("customs_agent_max_document_image_pixels_invalid")
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
        if not 0.1 <= self.readiness_timeout_seconds <= 120:
            raise ServiceConfigurationError("customs_agent_readiness_timeout_seconds_invalid")
        if not 1 <= self.evidence_retrieve_timeout_seconds <= 120:
            raise ServiceConfigurationError(
                "customs_agent_evidence_retrieve_timeout_seconds_invalid"
            )
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
        if (
            source.get("GUIDANCE_SERVICE_TOKEN_HEADER", "Authorization").strip().casefold()
            != "authorization"
        ):
            raise ServiceConfigurationError("local_profile_guidance_bearer_required")
        has_guidance_base_url = "GUIDANCE_SERVICE_BASE_URL" in source
        has_guidance_token = "GUIDANCE_SERVICE_TOKEN" in source
        has_guidance_registry_version = "GUIDANCE_SERVICE_REGISTRY_VERSION" in source
        has_preview_approval_path = "CUSTOMS_AGENT_DOCUMENT_PREVIEW_APPROVAL_PATH" in source
        has_preview_approval_sha256 = "CUSTOMS_AGENT_DOCUMENT_PREVIEW_APPROVAL_SHA256" in source
        if has_guidance_base_url != has_guidance_token:
            raise ServiceConfigurationError("guidance_service_peer_incomplete")
        if has_guidance_registry_version != has_guidance_base_url:
            raise ServiceConfigurationError("guidance_registry_version_peer_incomplete")
        if has_preview_approval_path != has_preview_approval_sha256:
            raise ServiceConfigurationError("customs_agent_document_preview_approval_incomplete")
        guidance_base_url: str | None = None
        guidance_service_token: SecretStr | None = None
        guidance_registry_version: GuidanceRegistryVersion | None = None
        evidence_profile: EvidenceProfile | None = None
        document_preview_approval_path: Path | None = None
        document_preview_approval_sha256: str | None = None
        if "CUSTOMS_AGENT_EVIDENCE_PROFILE" in source:
            configured_evidence_profile = _required(source, "CUSTOMS_AGENT_EVIDENCE_PROFILE")
            if configured_evidence_profile not in _EVIDENCE_PROFILES:
                raise ServiceConfigurationError("customs_agent_evidence_profile_invalid")
            evidence_profile = cast(EvidenceProfile, configured_evidence_profile)
        if configured_mode == "functional-uat" and evidence_profile is None:
            raise ServiceConfigurationError("customs_agent_evidence_profile_missing")
        if has_guidance_base_url:
            guidance_base_url = _required(source, "GUIDANCE_SERVICE_BASE_URL")
            guidance_service_token = SecretStr(_required(source, "GUIDANCE_SERVICE_TOKEN"))
            configured_registry_version = _required(source, "GUIDANCE_SERVICE_REGISTRY_VERSION")
            if configured_registry_version not in {"1.0", "2.0"}:
                raise ServiceConfigurationError("guidance_registry_version_invalid")
            guidance_registry_version = cast(GuidanceRegistryVersion, configured_registry_version)
        if has_preview_approval_path:
            document_preview_approval_path = _document_preview_approval_path(
                _required(source, "CUSTOMS_AGENT_DOCUMENT_PREVIEW_APPROVAL_PATH")
            )
            document_preview_approval_sha256 = _required(
                source, "CUSTOMS_AGENT_DOCUMENT_PREVIEW_APPROVAL_SHA256"
            )
        return cls(
            bearer_token=SecretStr(_required(source, "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN")),
            evidence_base_url=_required(source, "EVIDENCE_SERVICE_BASE_URL"),
            evidence_service_token=SecretStr(_required(source, "EVIDENCE_SERVICE_TOKEN")),
            evidence_profile=evidence_profile,
            guidance_base_url=guidance_base_url,
            guidance_service_token=guidance_service_token,
            guidance_registry_version=guidance_registry_version,
            service_mode=cast(ServiceMode, configured_mode),
            host=source.get("CUSTOMS_AGENT_HOST", "127.0.0.1").strip(),
            port=_integer(source, "CUSTOMS_AGENT_PORT", 18090, 1, 65_535),
            max_body_bytes=_integer(
                source, "CUSTOMS_AGENT_MAX_BODY_BYTES", _MAX_BODY_BYTES, 1_024, 32 * 1024 * 1024
            ),
            max_document_bytes=_integer(
                source,
                "CUSTOMS_AGENT_MAX_DOCUMENT_BYTES",
                _MAX_DOCUMENT_BYTES,
                1_024,
                32 * 1024 * 1024,
            ),
            max_document_pages=_integer(
                source,
                "CUSTOMS_AGENT_MAX_DOCUMENT_PAGES",
                _MAX_DOCUMENT_PAGES,
                1,
                1_000,
            ),
            max_document_image_pixels=_integer(
                source,
                "CUSTOMS_AGENT_MAX_DOCUMENT_IMAGE_PIXELS",
                _MAX_DOCUMENT_IMAGE_PIXELS,
                1_000_000,
                100_000_000,
            ),
            document_preview_approval_path=document_preview_approval_path,
            document_preview_approval_sha256=document_preview_approval_sha256,
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
            readiness_timeout_seconds=_number(
                source, "CUSTOMS_AGENT_READINESS_TIMEOUT_SECONDS", 5, 0.1, 120
            ),
            evidence_retrieve_timeout_seconds=_number(
                source, "CUSTOMS_AGENT_EVIDENCE_RETRIEVE_TIMEOUT_SECONDS", 90, 1, 120
            ),
            shutdown_grace_seconds=_number(
                source, "CUSTOMS_AGENT_SHUTDOWN_GRACE_SECONDS", 30, 1, 900
            ),
            log_level=source.get("CUSTOMS_AGENT_LOG_LEVEL", "INFO").strip().upper(),
        )


__all__ = [
    "EvidenceProfile",
    "GuidanceRegistryVersion",
    "ServiceConfigurationError",
    "ServiceMode",
    "ServiceSettings",
]
