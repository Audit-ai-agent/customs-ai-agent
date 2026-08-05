from __future__ import annotations

import logging
from typing import Literal

import uvicorn

from customs_ai_agent.service.app import create_app
from customs_ai_agent.service.settings import ServiceSettings


def _serve(mode: Literal["local-development", "functional-uat"] | None = None) -> None:
    settings = ServiceSettings.from_environment(service_mode_override=mode)
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(message)s",
    )
    app = create_app(settings)
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
    """Run the actual Claude/Data functional-UAT Agent service on loopback."""

    _serve("functional-uat")


if __name__ == "__main__":
    main()
