"""Small, fixed prompts used only by explicit functional UAT."""

from __future__ import annotations

from importlib.resources import files


DOCUMENT_ANALYSIS_PROMPT_ID = "customs_document_analysis"
DOCUMENT_ANALYSIS_PROMPT_VERSION = "1.3.1"
ASSISTANT_PROMPT_ID = "customs_grounded_assistant"
ASSISTANT_PROMPT_VERSION = "1.0.3"
REPORT_PROMPT_ID = "customs_grounded_review_report"
REPORT_PROMPT_VERSION = "1.0.3"
KNOWLEDGE_PROMPT_ID = "customs_official_trade_guidance"
KNOWLEDGE_PROMPT_VERSION = "1.1.0"
EVIDENCE_RELEVANCE_PROMPT_ID = "customs_evidence_relevance"
EVIDENCE_RELEVANCE_PROMPT_VERSION = "1.0.0"
FINDING_CLASSIFICATION_PROMPT_ID = "customs_finding_classification"
FINDING_CLASSIFICATION_PROMPT_VERSION = "1.0.0"
EXPERT_REVIEW_PROMPT_ID = "customs_expert_review"
EXPERT_REVIEW_PROMPT_VERSION = "1.0.0"


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


def load_evidence_relevance_system_prompt() -> str:
    return (
        files("customs_ai_agent.prompts")
        .joinpath("evidence_relevance_v1.txt")
        .read_text(encoding="utf-8")
    )


def load_official_guidance_system_prompt() -> str:
    return (
        files("customs_ai_agent.prompts")
        .joinpath("official_guidance_v1.txt")
        .read_text(encoding="utf-8")
    )


def load_finding_classification_system_prompt() -> str:
    return (
        files("customs_ai_agent.prompts")
        .joinpath("finding_classification_v1.txt")
        .read_text(encoding="utf-8")
    )


def load_expert_review_system_prompt() -> str:
    return (
        files("customs_ai_agent.prompts")
        .joinpath("expert_review_v1.txt")
        .read_text(encoding="utf-8")
    )


__all__ = [
    "DOCUMENT_ANALYSIS_PROMPT_ID",
    "DOCUMENT_ANALYSIS_PROMPT_VERSION",
    "ASSISTANT_PROMPT_ID",
    "ASSISTANT_PROMPT_VERSION",
    "REPORT_PROMPT_ID",
    "REPORT_PROMPT_VERSION",
    "KNOWLEDGE_PROMPT_ID",
    "KNOWLEDGE_PROMPT_VERSION",
    "EVIDENCE_RELEVANCE_PROMPT_ID",
    "EVIDENCE_RELEVANCE_PROMPT_VERSION",
    "FINDING_CLASSIFICATION_PROMPT_ID",
    "FINDING_CLASSIFICATION_PROMPT_VERSION",
    "EXPERT_REVIEW_PROMPT_ID",
    "EXPERT_REVIEW_PROMPT_VERSION",
    "load_document_analysis_system_prompt",
    "load_evidence_relevance_system_prompt",
    "load_expert_review_system_prompt",
    "load_finding_classification_system_prompt",
    "load_grounded_assistant_system_prompt",
    "load_grounded_report_system_prompt",
    "load_official_guidance_system_prompt",
]
