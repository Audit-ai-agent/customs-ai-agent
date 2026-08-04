"""Safe URL validation for the local Data HTTP boundary."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit, urlunsplit


class RetrievalError(RuntimeError):
    """A Data request could not produce a safe response."""


def _is_loopback(hostname: str) -> bool:
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return hostname.casefold() == "localhost"


def validate_evidence_destination(base_url: str) -> str:
    """Normalize one credential-free loopback Data origin."""

    value = base_url.strip()
    if not value or any(character.isspace() for character in value):
        raise ValueError("Data service base URL is invalid")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Data service base URL is invalid") from exc
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Data service base URL is invalid")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Data service base URL cannot contain credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Data service base URL must be an origin")
    host = parsed.hostname.casefold()
    if not _is_loopback(host):
        raise ValueError("Data service must be loopback-only")
    rendered_host = f"[{host}]" if ":" in host else host
    netloc = f"{rendered_host}:{port}" if port is not None else rendered_host
    return urlunsplit((parsed.scheme.casefold(), netloc, "", "", ""))


__all__ = ["RetrievalError", "validate_evidence_destination"]
