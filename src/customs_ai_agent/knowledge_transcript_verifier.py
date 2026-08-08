"""Fail-closed verifier for one standalone knowledge-assistant transcript turn.

This verifier is intentionally owned by the Agent because only the Agent
writes the ``agent-data`` and ``claude`` streams for a standalone knowledge
turn.  It prints no transcript path, body, prompt, model output, secret, or
provider error.  Callers receive only a stable status and a content-free error
code, while the full local-only record stays below ``_local/state``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .scenario_transcript import (
    _SAFE_KNOWLEDGE_TURN_ID,
    _SAFE_RUN_ID,
    _SENSITIVE_FIELD,
    _TEXT_REDACTIONS,
    _is_functional_uat_knowledge_transcript_root,
    _is_local_knowledge_transcript_root,
    _knowledge_payload_is_safe,
)


KnowledgeTranscriptServiceMode = Literal["local-development", "functional-uat"]
_STREAM_FILENAMES = frozenset({"agent-data.jsonl", "claude.jsonl"})
_RECORD_FIELDS = frozenset({"schema_version", "recorded_at", "event", "payload"})


class KnowledgeTranscriptVerificationError(RuntimeError):
    """A content-free reason why a local knowledge turn cannot be verified."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise KnowledgeTranscriptVerificationError(code)


def _turn_root(
    turn_directory: Path, *, service_mode: KnowledgeTranscriptServiceMode
) -> tuple[Path, Path]:
    """Return the resolved turn/root only when it has the exact safe layout."""

    candidate = turn_directory.expanduser()
    if not candidate.is_absolute():
        _fail("knowledge_transcript_turn_not_absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        _fail("knowledge_transcript_turn_unavailable")
    if not resolved.is_dir():
        _fail("knowledge_transcript_turn_invalid")

    conversation_directory = resolved.parent
    root = conversation_directory.parent
    if (
        _SAFE_KNOWLEDGE_TURN_ID.fullmatch(resolved.name) is None
        or _SAFE_RUN_ID.fullmatch(conversation_directory.name) is None
    ):
        _fail("knowledge_transcript_turn_layout_invalid")

    if service_mode == "functional-uat":
        if not _is_functional_uat_knowledge_transcript_root(root):
            _fail("knowledge_transcript_root_invalid")
    elif not _is_local_knowledge_transcript_root(root):
        _fail("knowledge_transcript_root_invalid")
    return resolved, root


def _contains_unredacted_secret(value: Any) -> bool:
    """Identify secret-shaped values without retaining or displaying them."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            # The writer retains a sensitive field name only after replacing
            # its value.  A verifier accepts that explicit marker but rejects
            # every other nested value under such a key.
            if _SENSITIVE_FIELD.search(key_text) and child != "[REDACTED]":
                return True
            if _contains_unredacted_secret(child):
                return True
        return False
    if isinstance(value, list | tuple):
        return any(_contains_unredacted_secret(item) for item in value)
    if isinstance(value, str):
        return any(pattern.search(value) is not None for pattern, _replacement in _TEXT_REDACTIONS)
    return False


def _validate_record(record: Any) -> None:
    if not isinstance(record, Mapping):
        _fail("knowledge_transcript_record_invalid")
    if set(record) != _RECORD_FIELDS:
        _fail("knowledge_transcript_record_shape_invalid")
    if record.get("schema_version") != "scenario-transcript/v1":
        _fail("knowledge_transcript_record_schema_invalid")
    if (
        not isinstance(record.get("event"), str)
        or not record["event"]
        or len(record["event"]) > 128
    ):
        _fail("knowledge_transcript_record_event_invalid")
    if not isinstance(record.get("payload"), Mapping):
        _fail("knowledge_transcript_record_payload_invalid")
    recorded_at = record.get("recorded_at")
    if not isinstance(recorded_at, str):
        _fail("knowledge_transcript_record_time_invalid")
    try:
        datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
    except ValueError:
        _fail("knowledge_transcript_record_time_invalid")
    if not _knowledge_payload_is_safe(record) or _contains_unredacted_secret(record):
        _fail("knowledge_transcript_record_sensitive_or_case_data")


def _validate_stream(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            count = 0
            for raw_line in handle:
                if not raw_line.endswith("\n") or not raw_line.strip():
                    _fail("knowledge_transcript_jsonl_invalid")
                try:
                    record = json.loads(raw_line)
                except (TypeError, ValueError):
                    _fail("knowledge_transcript_jsonl_invalid")
                _validate_record(record)
                count += 1
    except UnicodeDecodeError:
        _fail("knowledge_transcript_jsonl_invalid")
    except OSError:
        _fail("knowledge_transcript_stream_unavailable")
    if count == 0:
        _fail("knowledge_transcript_stream_empty")


def verify_knowledge_transcript_turn(
    turn_directory: Path,
    *,
    service_mode: KnowledgeTranscriptServiceMode,
) -> None:
    """Verify one safe standalone knowledge turn or raise a content-free error.

    Functional-UAT requires both Agent-owned streams.  Explicit local
    development may not invoke Claude, so its ``claude`` stream is optional;
    if present it is verified by the same rules.
    """

    resolved, _root = _turn_root(turn_directory, service_mode=service_mode)
    try:
        entries = tuple(resolved.iterdir())
    except OSError:
        _fail("knowledge_transcript_turn_unavailable")
    filenames = {entry.name for entry in entries}
    if any(not entry.is_file() or entry.name not in _STREAM_FILENAMES for entry in entries):
        _fail("knowledge_transcript_turn_entries_invalid")
    required = {"agent-data.jsonl"}
    if service_mode == "functional-uat":
        required.add("claude.jsonl")
    if not required.issubset(filenames):
        _fail("knowledge_transcript_required_stream_missing")
    for filename in sorted(filenames):
        _validate_stream(resolved / filename)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify one local-only standalone knowledge transcript turn without printing it."
    )
    parser.add_argument("--turn-directory", required=True)
    parser.add_argument(
        "--service-mode",
        required=True,
        choices=("local-development", "functional-uat"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the safe verifier CLI and emit status/code only."""

    args = _parser().parse_args(argv)
    try:
        verify_knowledge_transcript_turn(Path(args.turn_directory), service_mode=args.service_mode)
    except KnowledgeTranscriptVerificationError as exc:
        print(json.dumps({"status": "failed", "code": exc.code}, separators=(",", ":")))
        return 2
    print(json.dumps({"status": "passed"}, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())


__all__ = [
    "KnowledgeTranscriptVerificationError",
    "KnowledgeTranscriptServiceMode",
    "main",
    "verify_knowledge_transcript_turn",
]
