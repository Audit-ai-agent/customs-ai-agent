from __future__ import annotations

import logging
import os
from typing import Literal

import uvicorn

from customs_ai_agent.service.app import create_app
from customs_ai_agent.service.document_extraction import LocalDocumentPreviewEngine
from customs_ai_agent.service.settings import ServiceSettings


def _demo_document_preview_engine(settings: ServiceSettings) -> LocalDocumentPreviewEngine | None:
    """Build the explicit demo-only preview engine, or None for normal resolution.

    This entry-point opt-in stays outside ``ServiceSettings`` and the approved
    engine resolution: it selects only the fixed in-repo demo composition and
    never an engine, executable, or model path value from the environment.
    """

    if os.environ.get("CUSTOMS_AGENT_DOCUMENT_PREVIEW_DEMO", "").strip().casefold() != "claude-vlm":
        return None
    from customs_ai_agent.service.document_preview_vlm_demo import build_claude_vlm_demo_engine

    return build_claude_vlm_demo_engine(settings)


def _serve(mode: Literal["local-development", "functional-uat"] | None = None) -> None:
    settings = ServiceSettings.from_environment(service_mode_override=mode)
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(message)s",
    )
    app = create_app(settings, document_preview_engine=_demo_document_preview_engine(settings))
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        access_log=False,
        backlog=max(32, settings.max_concurrency * 4),
        limit_concurrency=settings.max_concurrency + 16,
        proxy_headers=False,
        server_header=False,
        timeout_graceful_shutdown=round(settings.shutdown_grace_seconds),
    )


def main() -> None:
    """Run the configured local-development profile (the compatibility command)."""

    _serve()


def dev_main() -> None:
    """Run deterministic local-development Agent service on loopback."""

    _serve("local-development")


def functional_uat_main() -> None:
    """Run the actual LLM(Claude or OpenAI)/Data functional-UAT Agent service on loopback."""

    _serve("functional-uat")


if __name__ == "__main__":
    main()
