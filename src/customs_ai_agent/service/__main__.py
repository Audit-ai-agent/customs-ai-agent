from __future__ import annotations

import logging

import uvicorn

from customs_ai_agent.service.app import create_app
from customs_ai_agent.service.settings import ServiceSettings


def main() -> None:
    settings = ServiceSettings.from_environment()
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


if __name__ == "__main__":
    main()
