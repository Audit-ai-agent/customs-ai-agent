from __future__ import annotations

import os

from dotenv import load_dotenv


def ensure_llm_mode_available(mode: str) -> None:
    if mode == "mock":
        return
    load_dotenv()
    if not os.getenv("CUSTOMS_AGENT_ENABLE_LLM"):
        raise RuntimeError(
            "AGENT_LLM_MODE=llm was requested, but CUSTOMS_AGENT_ENABLE_LLM is not set. "
            "Use mock mode for deterministic tests or configure an actual LLM adapter."
        )
    raise RuntimeError(
        "LLM mode is intentionally isolated in this MVP. Implement a concrete adapter "
        "before using live document extraction."
    )
