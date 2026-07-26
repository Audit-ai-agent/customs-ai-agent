from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from customs_ai_agent.uat_capability import capability_file_sha256, issue_uat_capability


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise argparse.ArgumentTypeError("timestamp must be ISO-8601") from None
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue one self-hashed, loopback-only Agent UAT capability."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--agent-commit", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--evidence-runtime-v3-sha256", required=True)
    parser.add_argument("--issued-at", required=True, type=_timestamp)
    parser.add_argument("--expires-at", required=True, type=_timestamp)
    parser.add_argument("--adaptive-active", action="store_true")
    args = parser.parse_args()
    capability = issue_uat_capability(
        agent_commit=args.agent_commit,
        model_id=args.model_id,
        evidence_runtime_v3_sha256=args.evidence_runtime_v3_sha256,
        issued_at=args.issued_at,
        expires_at=args.expires_at,
        adaptive_active=args.adaptive_active,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            capability.model_dump(mode="json", warnings=False),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "capability_sha256": capability.capability_sha256,
                "file_sha256": capability_file_sha256(output),
                "output": str(output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
