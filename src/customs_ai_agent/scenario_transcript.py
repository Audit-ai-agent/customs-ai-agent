"""Local-only detailed records for a deliberately run scenario.

The normal service and its structured logs never retain document bodies or
provider output.  Development stays opt-in by setting
``CUSTOMS_SCENARIO_TRANSCRIPT_ROOT`` to the workspace-local
``_local/state/scenario-runs`` directory before starting the Agent;
``functional-uat`` always selects that same fixed directory.  This module
deliberately rejects every other destination so a transcript cannot be
accidentally written into a repository or a shared output path.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRANSCRIPT_ROOT_ENV = "CUSTOMS_SCENARIO_TRANSCRIPT_ROOT"
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SENSITIVE_FIELD = re.compile(
    r"(?:authorization|cookie|token|api[_-]?key|secret|password|credential|dsn)", re.IGNORECASE
)
_TEXT_REDACTIONS = (
    (
        re.compile(r"(?i)\b(sk-ant-[A-Za-z0-9_-]{12,})\b"),
        "[REDACTED]",
    ),
    (
        re.compile(r"(?i)\b(gh[pousr]_[A-Za-z0-9]{12,})\b"),
        "[REDACTED]",
    ),
    (
        re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/-]{16,}"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(
            r"(?i)(\b(?:anthropic_api_key|law_api_oc|api[_-]?key|token|secret|password)\s*[:=]\s*)[^\s,;]+"
        ),
        r"\1[REDACTED]",
    ),
)
# The Backend owns ``backend-agent.jsonl``.  This package may only append the
# two streams for which it is the source of truth.
_STREAMS = frozenset({"agent-data", "claude"})
_WRITE_LOCK = threading.Lock()


class ScenarioTranscriptError(RuntimeError):
    """An opt-in transcript cannot be recorded safely."""


def _is_local_transcript_root(path: Path) -> bool:
    return (
        path.name == "scenario-runs"
        and path.parent.name == "state"
        and path.parent.parent.name == "_local"
    )


def _normalise_sensitive_values(values: Iterable[str | None]) -> tuple[str, ...]:
    """Return stable, usable literal redactions without retaining empty values."""

    # Four characters avoids replacing ordinary language when a caller gives a
    # malformed/empty setting.  All supported service and provider credentials
    # are substantially longer than this minimum.
    unique = {value for value in values if isinstance(value, str) and len(value) >= 4}
    return tuple(sorted(unique, key=len, reverse=True))


def _redact_text(value: str, *, sensitive_values: tuple[str, ...] = ()) -> str:
    for secret in sensitive_values:
        value = value.replace(secret, "[REDACTED]")
    for pattern, replacement in _TEXT_REDACTIONS:
        value = pattern.sub(replacement, value)
    return value


def redact_for_transcript(value: Any, *, sensitive_values: Iterable[str | None] = ()) -> Any:
    """Retain useful bodies while never retaining credential-shaped values."""

    literals = _normalise_sensitive_values(sensitive_values)

    def redact(nested: Any) -> Any:
        if isinstance(nested, Mapping):
            redacted: dict[str, Any] = {}
            for key, child in nested.items():
                rendered_key = _redact_text(str(key), sensitive_values=literals)
                redacted[rendered_key] = (
                    "[REDACTED]" if _SENSITIVE_FIELD.search(rendered_key) else redact(child)
                )
            return redacted
        if isinstance(nested, list | tuple):
            return [redact(item) for item in nested]
        if isinstance(nested, str):
            return _redact_text(nested, sensitive_values=literals)
        return nested

    return redact(value)


def _is_same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.normpath(str(left))) == os.path.normcase(
        os.path.normpath(str(right))
    )


def _functional_uat_directory() -> Path:
    """Resolve the one workspace-local Functional-UAT transcript destination."""

    source = Path(__file__).resolve()
    try:
        repository = source.parents[2]
    except IndexError:
        raise ScenarioTranscriptError("scenario_transcript_workspace_unavailable") from None
    if repository.name != "customs-ai-agent":
        raise ScenarioTranscriptError("scenario_transcript_workspace_unavailable")
    return repository.parent / "_local" / "state" / "scenario-runs"


def _validated_root(
    path: Path, *, sensitive_values: Iterable[str | None]
) -> "ScenarioTranscriptRoot":
    candidate = path.expanduser()
    if not candidate.is_absolute():
        raise ScenarioTranscriptError("scenario_transcript_root_not_absolute")
    candidate = candidate.resolve(strict=False)
    if not _is_local_transcript_root(candidate):
        raise ScenarioTranscriptError("scenario_transcript_root_invalid")
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ScenarioTranscriptError("scenario_transcript_root_unavailable") from exc
    if not _is_local_transcript_root(resolved):
        raise ScenarioTranscriptError("scenario_transcript_root_invalid")
    return ScenarioTranscriptRoot(
        directory=resolved,
        sensitive_values=_normalise_sensitive_values(sensitive_values),
    )


@dataclass(frozen=True, slots=True)
class ScenarioTranscriptRun:
    """One run directory with a constrained set of append-only streams."""

    directory: Path
    sensitive_values: tuple[str, ...] = field(default=(), repr=False)

    def with_sensitive_values(self, values: Iterable[str | None]) -> "ScenarioTranscriptRun":
        return ScenarioTranscriptRun(
            directory=self.directory,
            sensitive_values=_normalise_sensitive_values((*self.sensitive_values, *values)),
        )

    def write(self, stream: str, event: str, payload: Mapping[str, Any]) -> None:
        if stream not in _STREAMS:
            raise ScenarioTranscriptError("scenario_transcript_stream_invalid")
        if not event or len(event) > 128:
            raise ScenarioTranscriptError("scenario_transcript_event_invalid")
        record = {
            "schema_version": "scenario-transcript/v1",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": redact_for_transcript(payload, sensitive_values=self.sensitive_values),
        }
        try:
            line = json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            )
            with _WRITE_LOCK:
                with (self.directory / f"{stream}.jsonl").open(
                    "a", encoding="utf-8", newline="\n"
                ) as handle:
                    handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        except OSError as exc:
            raise ScenarioTranscriptError("scenario_transcript_write_failed") from exc


@dataclass(frozen=True, slots=True)
class ScenarioTranscriptRoot:
    """Disabled unless an explicit safe root, or Functional-UAT, selects one."""

    directory: Path | None = None
    sensitive_values: tuple[str, ...] = field(default=(), repr=False)

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        sensitive_values: Iterable[str | None] = (),
    ) -> "ScenarioTranscriptRoot":
        source = os.environ if environ is None else environ
        configured = source.get(TRANSCRIPT_ROOT_ENV, "").strip()
        if not configured:
            return cls()
        return _validated_root(Path(configured), sensitive_values=sensitive_values)

    @classmethod
    def for_service_mode(
        cls,
        service_mode: str,
        environ: Mapping[str, str] | None = None,
        *,
        sensitive_values: Iterable[str | None] = (),
    ) -> "ScenarioTranscriptRoot":
        """Select the fixed UAT root; development remains explicit opt-in."""

        if service_mode != "functional-uat":
            return cls.from_environment(environ, sensitive_values=sensitive_values)
        expected = _functional_uat_directory().resolve(strict=False)
        source = os.environ if environ is None else environ
        configured = source.get(TRANSCRIPT_ROOT_ENV, "").strip()
        if configured:
            configured_path = Path(configured).expanduser().resolve(strict=False)
            if not _is_same_path(configured_path, expected):
                raise ScenarioTranscriptError("scenario_transcript_root_invalid")
        return _validated_root(expected, sensitive_values=sensitive_values)

    def with_sensitive_values(self, values: Iterable[str | None]) -> "ScenarioTranscriptRoot":
        if self.directory is None:
            return self
        return ScenarioTranscriptRoot(
            directory=self.directory,
            sensitive_values=_normalise_sensitive_values((*self.sensitive_values, *values)),
        )

    def for_run(self, review_run_id: str) -> ScenarioTranscriptRun | None:
        if self.directory is None:
            return None
        if _SAFE_RUN_ID.fullmatch(review_run_id) is None:
            raise ScenarioTranscriptError("scenario_transcript_run_id_invalid")
        candidate = self.directory / review_run_id
        try:
            candidate.mkdir(exist_ok=True)
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ScenarioTranscriptError("scenario_transcript_run_unavailable") from exc
        if resolved.parent != self.directory:
            raise ScenarioTranscriptError("scenario_transcript_run_invalid")
        return ScenarioTranscriptRun(directory=resolved, sensitive_values=self.sensitive_values)


__all__ = [
    "ScenarioTranscriptError",
    "ScenarioTranscriptRoot",
    "ScenarioTranscriptRun",
    "TRANSCRIPT_ROOT_ENV",
    "redact_for_transcript",
]
