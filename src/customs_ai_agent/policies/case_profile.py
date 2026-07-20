"""Versioned document-set requirements for deterministic customs review.

This profile deliberately requires the three transaction documents needed for
the CI/PL/B/L rules.  A certificate of origin remains conditional: when one is
provided it is validated against the invoice, and when Evidence requires one
its absence is handled by the Evidence/Compliance Gate.
"""

from __future__ import annotations

from typing import Literal

from customs_ai_agent.schemas.models import TradeType


CASE_PROFILE_ID = "standard-trade-document-set"
CASE_PROFILE_VERSION = "1.0"
RequiredDocumentType = Literal["invoice", "packing_list", "bill_of_lading"]

_REQUIRED_DOCUMENT_TYPES: dict[TradeType, tuple[RequiredDocumentType, ...]] = {
    "import": ("invoice", "packing_list", "bill_of_lading"),
    "export": ("invoice", "packing_list", "bill_of_lading"),
}


def required_document_types(trade_type: TradeType | str) -> tuple[RequiredDocumentType, ...]:
    """Return the immutable required document set for one supported trade type."""

    if trade_type == "import":
        return _REQUIRED_DOCUMENT_TYPES["import"]
    if trade_type == "export":
        return _REQUIRED_DOCUMENT_TYPES["export"]
    raise ValueError("unsupported trade type for the active case profile")


__all__ = [
    "CASE_PROFILE_ID",
    "CASE_PROFILE_VERSION",
    "RequiredDocumentType",
    "required_document_types",
]
