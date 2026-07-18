from __future__ import annotations

from typing import Any

from customs_ai_agent.nodes.common import audit


def without_image_payloads(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: without_image_payloads(item) for key, item in value.items() if key != "data_base64"
        }
    if isinstance(value, list):
        return [without_image_payloads(item) for item in value]
    return value


def redact_sensitive_inputs(state: dict[str, Any]) -> dict[str, Any]:
    """Remove raw image bytes before any compiled graph returns its state."""

    return {
        "documents": without_image_payloads(state.get("documents") or []),
        "sensitive_inputs_redacted": True,
        "current_stage": "redact_sensitive_inputs",
        "audit_trail": audit(
            state,
            "redact_sensitive_inputs",
            "completed",
            "Removed raw image payloads from the completed review state.",
        ),
    }


__all__ = ["redact_sensitive_inputs", "without_image_payloads"]
