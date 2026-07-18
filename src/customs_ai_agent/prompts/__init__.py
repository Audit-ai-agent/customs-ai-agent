from __future__ import annotations

from importlib.resources import files


DOCUMENT_ANALYSIS_PROMPT_ID = "customs_document_analysis"
DOCUMENT_ANALYSIS_PROMPT_VERSION = "1.2.0"
DOCUMENT_VISION_ANALYSIS_PROMPT_ID = "customs_document_vision_analysis"
DOCUMENT_VISION_ANALYSIS_PROMPT_VERSION = "1.3.0"
DOCUMENT_CROP_RECONCILIATION_PROMPT_ID = "customs_document_crop_reconciliation"
DOCUMENT_CROP_RECONCILIATION_PROMPT_VERSION = "1.0.0"
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
]
