"""The one text-extraction prompt used by functional UAT."""

from __future__ import annotations

from importlib.resources import files


DOCUMENT_ANALYSIS_PROMPT_ID = "customs_document_analysis"
DOCUMENT_ANALYSIS_PROMPT_VERSION = "1.3.0"


def load_document_analysis_system_prompt() -> str:
    return (
        files("customs_ai_agent.prompts")
        .joinpath("document_analysis_v1.txt")
        .read_text(encoding="utf-8")
    )


__all__ = [
    "DOCUMENT_ANALYSIS_PROMPT_ID",
    "DOCUMENT_ANALYSIS_PROMPT_VERSION",
    "load_document_analysis_system_prompt",
]
