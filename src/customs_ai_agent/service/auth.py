from __future__ import annotations

import hashlib
import hmac

from customs_ai_agent.service.settings import ServiceSettings


def bearer_authorized(authorization: str | None, settings: ServiceSettings) -> bool:
    """Validate the internal bearer credential without length-dependent comparison."""

    if not isinstance(authorization, str):
        presented = ""
    else:
        scheme, separator, value = authorization.partition(" ")
        presented = value if separator and scheme.casefold() == "bearer" else ""
    expected_digest = hashlib.sha256(
        settings.bearer_token.get_secret_value().encode("utf-8")
    ).digest()
    presented_digest = hashlib.sha256(presented.encode("utf-8")).digest()
    return bool(presented) and hmac.compare_digest(expected_digest, presented_digest)


__all__ = ["bearer_authorized"]
