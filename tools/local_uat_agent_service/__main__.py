from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from tools.local_uat_agent_service.app import (
    LOCAL_UAT_ACKNOWLEDGEMENT,
    LocalUatConfigurationError,
    create_local_uat_app,
    load_local_uat_manifest,
    local_uat_service_settings,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run loopback-only LOCAL USER UAT Agent")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument(
        "--acknowledge-local-uat",
        action="store_true",
        help=f"Acknowledge {LOCAL_UAT_ACKNOWLEDGEMENT}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.acknowledge_local_uat is not True:
        raise LocalUatConfigurationError("uat_acknowledgement_required")
    manifest = load_local_uat_manifest(args.manifest)
    settings = local_uat_service_settings(manifest, args.env_file)
    logging.basicConfig(level=getattr(logging, settings.log_level), format="%(message)s")
    app = create_local_uat_app(
        manifest=manifest,
        settings=settings,
        acknowledge_local_uat=True,
        env_file=args.env_file,
    )
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=settings.port,
        access_log=False,
        backlog=32,
        limit_concurrency=17,
        proxy_headers=False,
        server_header=False,
        timeout_graceful_shutdown=round(settings.shutdown_grace_seconds),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
