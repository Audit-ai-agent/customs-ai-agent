from __future__ import annotations

from importlib.resources import files


DOCUMENT_ANALYSIS_PROMPT_ID = "customs_document_analysis"
DOCUMENT_ANALYSIS_PROMPT_VERSION = "1.2.0"
DOCUMENT_VISION_ANALYSIS_PROMPT_ID = "customs_document_vision_analysis"
DOCUMENT_VISION_ANALYSIS_PROMPT_VERSION = "1.3.0"
DOCUMENT_CROP_RECONCILIATION_PROMPT_ID = "customs_document_crop_reconciliation"
DOCUMENT_CROP_RECONCILIATION_PROMPT_VERSION = "1.0.0"
REVIEW_COORDINATOR_PROMPT_ID = "customs_review_coordinator"
REVIEW_COORDINATOR_PROMPT_VERSION = "1.0.0"
NARRATIVE_REPORT_PROMPT_ID = "customs_narrative_report"
NARRATIVE_REPORT_PROMPT_VERSION = "1.0.0"
CASE_ANSWER_PROMPT_ID = "customs_case_answer"
CASE_ANSWER_PROMPT_VERSION = "1.0.0"
EXPERIENCE_REPORT_PROMPT_ID = "customs_experience_report"
EXPERIENCE_REPORT_PROMPT_VERSION = "2.0.0"
EXPERIENCE_ENTAILMENT_PROMPT_ID = "customs_experience_entailment"
EXPERIENCE_ENTAILMENT_PROMPT_VERSION = "1.0.0"
CASE_ANSWER_V2_PROMPT_ID = "customs_case_answer_v2"
CASE_ANSWER_V2_PROMPT_VERSION = "2.0.0"
CASE_ANSWER_VERIFIER_PROMPT_ID = "customs_case_answer_verifier"
CASE_ANSWER_VERIFIER_PROMPT_VERSION = "1.0.0"
CASE_SCOPE_CLASSIFIER_PROMPT_ID = "customs_case_scope_classifier"
CASE_SCOPE_CLASSIFIER_PROMPT_VERSION = "1.0.0"
_CROP_PROMPT_MARKER = "\n=== CROP_RECONCILIATION_PROMPT ===\n"


def load_document_analysis_system_prompt() -> str:
    return (
        files("customs_ai_agent.prompts")
        .joinpath("document_analysis_v1.txt")
        .read_text(encoding="utf-8")
    )


def load_document_vision_analysis_system_prompt() -> str:
    content = (
        files("customs_ai_agent.prompts")
        .joinpath("document_vision_analysis_v1.txt")
        .read_text(encoding="utf-8")
    )
    main_prompt, marker, _ = content.partition(_CROP_PROMPT_MARKER)
    if not marker:
        raise RuntimeError("vision prompt is missing its crop prompt section")
    return main_prompt.strip() + "\n"


def load_document_crop_reconciliation_system_prompt() -> str:
    content = (
        files("customs_ai_agent.prompts")
        .joinpath("document_vision_analysis_v1.txt")
        .read_text(encoding="utf-8")
    )
    _, marker, crop_prompt = content.partition(_CROP_PROMPT_MARKER)
    if not marker or not crop_prompt.strip():
        raise RuntimeError("vision prompt is missing its crop prompt section")
    return crop_prompt.strip() + "\n"


def _load_prompt(name: str) -> str:
    return files("customs_ai_agent.prompts").joinpath(name).read_text(encoding="utf-8")


def load_review_coordinator_system_prompt() -> str:
    return _load_prompt("review_coordinator_v1.txt")


def load_narrative_report_system_prompt() -> str:
    return _load_prompt("narrative_report_v1.txt")


def load_case_answer_system_prompt() -> str:
    return _load_prompt("case_answer_v1.txt")


def load_experience_report_system_prompt() -> str:
    return _load_prompt("experience_report_v2.txt")


def load_experience_entailment_system_prompt() -> str:
    return _load_prompt("experience_entailment_v1.txt")


def load_case_answer_v2_system_prompt() -> str:
    return _load_prompt("case_answer_v2.txt")


def load_case_answer_verifier_system_prompt() -> str:
    return _load_prompt("case_answer_verifier_v1.txt")


def load_case_scope_classifier_system_prompt() -> str:
    return _load_prompt("case_scope_classifier_v1.txt")


__all__ = [
    "DOCUMENT_ANALYSIS_PROMPT_ID",
    "DOCUMENT_ANALYSIS_PROMPT_VERSION",
    "DOCUMENT_CROP_RECONCILIATION_PROMPT_ID",
    "DOCUMENT_CROP_RECONCILIATION_PROMPT_VERSION",
    "DOCUMENT_VISION_ANALYSIS_PROMPT_ID",
    "DOCUMENT_VISION_ANALYSIS_PROMPT_VERSION",
    "load_document_analysis_system_prompt",
    "load_document_crop_reconciliation_system_prompt",
    "load_document_vision_analysis_system_prompt",
    "REVIEW_COORDINATOR_PROMPT_ID",
    "REVIEW_COORDINATOR_PROMPT_VERSION",
    "NARRATIVE_REPORT_PROMPT_ID",
    "NARRATIVE_REPORT_PROMPT_VERSION",
    "CASE_ANSWER_PROMPT_ID",
    "CASE_ANSWER_PROMPT_VERSION",
    "EXPERIENCE_REPORT_PROMPT_ID",
    "EXPERIENCE_REPORT_PROMPT_VERSION",
    "EXPERIENCE_ENTAILMENT_PROMPT_ID",
    "EXPERIENCE_ENTAILMENT_PROMPT_VERSION",
    "CASE_ANSWER_V2_PROMPT_ID",
    "CASE_ANSWER_V2_PROMPT_VERSION",
    "CASE_ANSWER_VERIFIER_PROMPT_ID",
    "CASE_ANSWER_VERIFIER_PROMPT_VERSION",
    "CASE_SCOPE_CLASSIFIER_PROMPT_ID",
    "CASE_SCOPE_CLASSIFIER_PROMPT_VERSION",
    "load_review_coordinator_system_prompt",
    "load_narrative_report_system_prompt",
    "load_case_answer_system_prompt",
    "load_experience_report_system_prompt",
    "load_experience_entailment_system_prompt",
    "load_case_answer_v2_system_prompt",
    "load_case_answer_verifier_system_prompt",
    "load_case_scope_classifier_system_prompt",
]
