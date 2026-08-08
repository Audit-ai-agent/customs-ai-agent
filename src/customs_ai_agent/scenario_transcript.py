"""Local-only detailed records for a deliberately run scenario.

The normal service and its structured logs never retain document bodies or
provider output.  Development stays opt-in by setting
``CUSTOMS_SCENARIO_TRANSCRIPT_ROOT`` to the workspace-local
``_local/state/scenario-runs`` directory before starting the Agent;
``functional-uat`` always selects that same fixed directory.  Its standalone
knowledge turns live below the reserved ``scenario-runs/knowledge`` namespace,
never below a Case or Review Run directory.  Earlier explicit development
records in the sibling ``knowledge-runs`` directory remain readable and are
never moved or deleted.  This module deliberately rejects every other
destination so a transcript cannot be accidentally written into a repository
or a shared output path.
"""

from __future__ import annotations

import json
import os
import re
import threading
from uuid import uuid4
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRANSCRIPT_ROOT_ENV = "CUSTOMS_SCENARIO_TRANSCRIPT_ROOT"
KNOWLEDGE_TRANSCRIPT_ROOT_ENV = "CUSTOMS_KNOWLEDGE_TRANSCRIPT_ROOT"
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SAFE_KNOWLEDGE_TURN_ID = re.compile(r"^turn-[a-f0-9]{32}$")
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
# A standalone knowledge turn is intentionally incapable of becoming a
# disguised Case transcript.  The review-specific objects are persisted by
# Backend and, when explicitly recorded, use the separate scenario-runs
# boundary.  These keys are rejected recursively before any local write.
_KNOWLEDGE_FORBIDDEN_FIELDS = frozenset(
    {
        "case",
        "caseid",
        "review",
        "reviewrun",
        "reviewrunid",
        "reviewresult",
        "document",
        "documents",
        "documentid",
        "documentbody",
        "documenttext",
        "rawdocument",
        "rawdocumentbody",
        "rawdocumenttext",
        "rawtext",
        "filename",
        "filecontent",
        "uploadedfile",
        "contenttext",
        "inputmanifestsha256",
        "findingid",
        "correctiondraft",
        "correctionproposal",
    }
)
_KNOWLEDGE_FORBIDDEN_PREFIXES = ("case", "review", "document", "raw", "upload")


class ScenarioTranscriptError(RuntimeError):
    """An opt-in transcript cannot be recorded safely."""


def _is_local_transcript_root(path: Path) -> bool:
    return (
        path.name == "scenario-runs"
        and path.parent.name == "state"
        and path.parent.parent.name == "_local"
    )


def _is_legacy_local_knowledge_transcript_root(path: Path) -> bool:
    return (
        path.name == "knowledge-runs"
        and path.parent.name == "state"
        and path.parent.parent.name == "_local"
    )


def _is_functional_uat_knowledge_transcript_root(path: Path) -> bool:
    """Return whether ``path`` is the non-Case UAT knowledge namespace."""

    return (
        path.name == "knowledge"
        and path.parent.name == "scenario-runs"
        and path.parent.parent.name == "state"
        and path.parent.parent.parent.name == "_local"
    )


def _is_local_knowledge_transcript_root(path: Path) -> bool:
    """Allow only the current UAT namespace or preserved explicit-dev history."""

    return _is_functional_uat_knowledge_transcript_root(
        path
    ) or _is_legacy_local_knowledge_transcript_root(path)


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


def _functional_uat_knowledge_directory() -> Path:
    """Resolve the reserved non-Case UAT destination for knowledge turns."""

    source = Path(__file__).resolve()
    try:
        repository = source.parents[2]
    except IndexError:
        raise ScenarioTranscriptError("knowledge_transcript_workspace_unavailable") from None
    if repository.name != "customs-ai-agent":
        raise ScenarioTranscriptError("knowledge_transcript_workspace_unavailable")
    return repository.parent / "_local" / "state" / "scenario-runs" / "knowledge"


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


def _validated_knowledge_root(
    path: Path, *, sensitive_values: Iterable[str | None]
) -> "KnowledgeTranscriptRoot":
    candidate = path.expanduser()
    if not candidate.is_absolute():
        raise ScenarioTranscriptError("knowledge_transcript_root_not_absolute")
    candidate = candidate.resolve(strict=False)
    if not _is_local_knowledge_transcript_root(candidate):
        raise ScenarioTranscriptError("knowledge_transcript_root_invalid")
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ScenarioTranscriptError("knowledge_transcript_root_unavailable") from exc
    if not _is_local_knowledge_transcript_root(resolved):
        raise ScenarioTranscriptError("knowledge_transcript_root_invalid")
    return KnowledgeTranscriptRoot(
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


def _knowledge_payload_is_safe(value: Any) -> bool:
    """Reject review/document-shaped structures from knowledge transcripts.

    The transcript may retain a user question, official-source excerpts, and
    Claude's bounded answer because it lives below ``_local/state``.  It must
    never become a second copy of a Case, Review Run, or uploaded document.
    Checking keys recursively keeps that boundary true even if a future caller
    accidentally nests an internal object in a Data or provider payload.
    """

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if (
                normalized in _KNOWLEDGE_FORBIDDEN_FIELDS
                or "document" in normalized
                or normalized.startswith(_KNOWLEDGE_FORBIDDEN_PREFIXES)
            ):
                return False
            if not _knowledge_payload_is_safe(child):
                return False
        return True
    if isinstance(value, list | tuple):
        return all(_knowledge_payload_is_safe(item) for item in value)
    return True


@dataclass(frozen=True, slots=True)
class KnowledgeTranscriptTurn(ScenarioTranscriptRun):
    """One isolated standalone knowledge-assistant turn.

    It deliberately keeps the same two Agent-owned streams as a review
    transcript, but its directory is independent and the payload guard blocks
    Case/Review/document objects.  The concrete subclass remains compatible
    with the Claude adapter's narrow transcript sink type.
    """

    def with_sensitive_values(self, values: Iterable[str | None]) -> "KnowledgeTranscriptTurn":
        return KnowledgeTranscriptTurn(
            directory=self.directory,
            sensitive_values=_normalise_sensitive_values((*self.sensitive_values, *values)),
        )

    def write(self, stream: str, event: str, payload: Mapping[str, Any]) -> None:
        if not _knowledge_payload_is_safe(payload):
            raise ScenarioTranscriptError("knowledge_transcript_payload_forbidden")
        try:
            # ``dataclass(slots=True)`` replaces the class object, so use the
            # explicit base method instead of zero-argument ``super()``.
            ScenarioTranscriptRun.write(self, stream, event, payload)
        except ScenarioTranscriptError as exc:
            if str(exc) == "scenario_transcript_write_failed":
                raise ScenarioTranscriptError("knowledge_transcript_write_failed") from exc
            raise


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
        # ``scenario-runs/knowledge`` is a reserved namespace for standalone
        # knowledge turns.  A Case/Review Run can never claim it, even if an
        # upstream caller were to pass a technically path-safe identifier.
        if review_run_id.casefold() == "knowledge":
            raise ScenarioTranscriptError("scenario_transcript_run_reserved")
        candidate = self.directory / review_run_id
        try:
            candidate.mkdir(exist_ok=True)
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ScenarioTranscriptError("scenario_transcript_run_unavailable") from exc
        if resolved.parent != self.directory:
            raise ScenarioTranscriptError("scenario_transcript_run_invalid")
        return ScenarioTranscriptRun(directory=resolved, sensitive_values=self.sensitive_values)


@dataclass(frozen=True, slots=True)
class KnowledgeTranscriptRoot:
    """Disabled unless a standalone knowledge transcript is explicitly selected.

    The Functional-UAT profile always uses the fixed
    ``_local/state/scenario-runs/knowledge`` destination.  It is a sibling
    namespace, not a review-run directory.  ``local-development`` only records
    when its dedicated environment variable names that exact safe directory.
    The historical ``_local/state/knowledge-runs`` root remains an allowed
    explicit-development destination; no existing record is migrated or
    deleted.  The dedicated variable intentionally does not reuse the review
    transcript variable, so routine development cannot begin retaining
    knowledge conversations by accident.
    """

    directory: Path | None = None
    sensitive_values: tuple[str, ...] = field(default=(), repr=False)

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        sensitive_values: Iterable[str | None] = (),
    ) -> "KnowledgeTranscriptRoot":
        source = os.environ if environ is None else environ
        configured = source.get(KNOWLEDGE_TRANSCRIPT_ROOT_ENV, "").strip()
        if not configured:
            return cls()
        return _validated_knowledge_root(Path(configured), sensitive_values=sensitive_values)

    @classmethod
    def for_service_mode(
        cls,
        service_mode: str,
        environ: Mapping[str, str] | None = None,
        *,
        sensitive_values: Iterable[str | None] = (),
    ) -> "KnowledgeTranscriptRoot":
        if service_mode != "functional-uat":
            return cls.from_environment(environ, sensitive_values=sensitive_values)
        expected = _functional_uat_knowledge_directory().resolve(strict=False)
        source = os.environ if environ is None else environ
        configured = source.get(KNOWLEDGE_TRANSCRIPT_ROOT_ENV, "").strip()
        if configured:
            configured_path = Path(configured).expanduser().resolve(strict=False)
            if not _is_same_path(configured_path, expected):
                raise ScenarioTranscriptError("knowledge_transcript_root_invalid")
        return _validated_knowledge_root(expected, sensitive_values=sensitive_values)

    def with_sensitive_values(self, values: Iterable[str | None]) -> "KnowledgeTranscriptRoot":
        if self.directory is None:
            return self
        return KnowledgeTranscriptRoot(
            directory=self.directory,
            sensitive_values=_normalise_sensitive_values((*self.sensitive_values, *values)),
        )

    def start_turn(self, conversation_id: str) -> KnowledgeTranscriptTurn | None:
        """Allocate a random, path-safe local turn ID without retaining input text."""

        if self.directory is None:
            return None
        return self.for_turn(conversation_id, f"turn-{uuid4().hex}")

    def for_turn(self, conversation_id: str, turn_id: str) -> KnowledgeTranscriptTurn | None:
        """Open one explicit safe path; primarily useful to focused local tests."""

        if self.directory is None:
            return None
        if _SAFE_RUN_ID.fullmatch(conversation_id) is None:
            raise ScenarioTranscriptError("knowledge_transcript_conversation_id_invalid")
        if _SAFE_KNOWLEDGE_TURN_ID.fullmatch(turn_id) is None:
            raise ScenarioTranscriptError("knowledge_transcript_turn_id_invalid")
        conversation_directory = self.directory / conversation_id
        try:
            conversation_directory.mkdir(exist_ok=True)
            resolved_conversation = conversation_directory.resolve(strict=True)
        except OSError as exc:
            raise ScenarioTranscriptError("knowledge_transcript_conversation_unavailable") from exc
        if resolved_conversation.parent != self.directory:
            raise ScenarioTranscriptError("knowledge_transcript_conversation_invalid")
        candidate = resolved_conversation / turn_id
        try:
            # A duplicate turn identifier is never appended to.  The generated
            # ID is random, and a collision must stop the transcript rather
            # than merge two user turns.
            candidate.mkdir()
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ScenarioTranscriptError("knowledge_transcript_turn_unavailable") from exc
        if resolved.parent != resolved_conversation:
            raise ScenarioTranscriptError("knowledge_transcript_turn_invalid")
        return KnowledgeTranscriptTurn(directory=resolved, sensitive_values=self.sensitive_values)


__all__ = [
    "KNOWLEDGE_TRANSCRIPT_ROOT_ENV",
    "KnowledgeTranscriptRoot",
    "KnowledgeTranscriptTurn",
    "ScenarioTranscriptError",
    "ScenarioTranscriptRoot",
    "ScenarioTranscriptRun",
    "TRANSCRIPT_ROOT_ENV",
    "redact_for_transcript",
]
