from __future__ import annotations

from typing import Any


def requires_human_review(state: dict[str, Any]) -> bool:
    if state.get("requires_human_review"):
        return True
    issues = state.get("validation_issues") or []
    findings = state.get("regulatory_findings") or []
    if any(item.get("severity") in {"medium", "high", "critical"} for item in issues):
        return True
    if any(item.get("severity") in {"medium", "high", "critical"} for item in findings):
        return True
    if state.get("retry_count", 0) > 0:
        return True
    return False
