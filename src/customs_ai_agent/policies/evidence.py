from __future__ import annotations

from customs_ai_agent.schemas.models import RegulatoryFinding


def enforce_evidence_gate(
    finding: RegulatoryFinding,
    available_evidence_ids: set[str],
) -> bool:
    if not finding.evidence_ids:
        return False
    if not set(finding.evidence_ids).issubset(available_evidence_ids):
        return False
    return True
