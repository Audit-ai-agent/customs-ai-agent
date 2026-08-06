"""Small, fixed prompts used only by explicit functional UAT."""

from __future__ import annotations

from importlib.resources import files


DOCUMENT_ANALYSIS_PROMPT_ID = "customs_document_analysis"
DOCUMENT_ANALYSIS_PROMPT_VERSION = "1.3.1"
ASSISTANT_PROMPT_ID = "customs_grounded_assistant"
ASSISTANT_PROMPT_VERSION = "1.0.1"
REPORT_PROMPT_ID = "customs_grounded_review_report"
REPORT_PROMPT_VERSION = "1.0.2"


def load_document_analysis_system_prompt() -> str:
    return (
        files("customs_ai_agent.prompts")
        .joinpath("document_analysis_v1.txt")
        .read_text(encoding="utf-8")
    )


def load_grounded_assistant_system_prompt() -> str:
    return (
        files("customs_ai_agent.prompts")
        .joinpath("grounded_assistant_v1.txt")
        .read_text(encoding="utf-8")
    )


def load_grounded_report_system_prompt() -> str:
    return (
        files("customs_ai_agent.prompts")
        .joinpath("grounded_report_v1.txt")
        .read_text(encoding="utf-8")
    )


__all__ = [
    "DOCUMENT_ANALYSIS_PROMPT_ID",
    "DOCUMENT_ANALYSIS_PROMPT_VERSION",
    "ASSISTANT_PROMPT_ID",
    "ASSISTANT_PROMPT_VERSION",
    "REPORT_PROMPT_ID",
    "REPORT_PROMPT_VERSION",
    "load_document_analysis_system_prompt",
    "load_grounded_assistant_system_prompt",
    "load_grounded_report_system_prompt",
]
