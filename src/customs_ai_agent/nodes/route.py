from __future__ import annotations

from typing import Any

from customs_ai_agent.nodes.common import audit
from customs_ai_agent.policies.outcome import determine_outcome


def route_to_human_or_complete(state: dict[str, Any]) -> dict[str, Any]:
    final_status, human, reasons = determine_outcome(state)
    report = dict(state.get("report_json") or {})
    report["overall_status"] = final_status
    if report.get("summary") is not None:
        report["summary"] = {**report["summary"], "outcome_reasons": reasons}
    return {
        "final_status": final_status,
        "requires_human_review": human,
        "report_json": report,
        "current_stage": "route_to_human_or_complete",
        "audit_trail": audit(
            state,
            "route_to_human_or_complete",
            "completed",
            "Routed case to completion or human review.",
            {
                "final_status": final_status,
                "requires_human_review": human,
                "outcome_reasons": reasons,
            },
        ),
    }
