"""Verify the non-circular two-commit OCR release freeze.

C0 is the clean commit authorized for one-time holdout access.  The descriptor
and receipt bind to C0.  C1 must be C0's single-parent child and may change only
the two literals in :mod:`customs_ai_agent.ocr_release_pin`.  C1 is deliberately
not embedded in either the descriptor or source, which avoids a self-referential
commit hash.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import customs_ai_agent.ocr_release_pin as release_pin

PIN_RELATIVE_PATH = "src/customs_ai_agent/ocr_release_pin.py"
DESCRIPTOR_SCHEMA = "ocr-benchmark-descriptor/v3"
RECEIPT_SCHEMA = "release-holdout-authorization-consumption/v2"
DATASET_ID = "trade_document_triplets"
EVALUATION_VERSION = "evaluation-v3"
SHA256 = re.compile(r"^[a-f0-9]{64}$")
COMMIT_SHA = re.compile(r"^[a-f0-9]{40}$")
MAX_DESCRIPTOR_BYTES = 512 * 1024
MAX_RECEIPT_BYTES = 16 * 1024
GitReader = Callable[[Path, Sequence[str]], bytes]

_PIN_HEADER = '"""Literal-only C0 authorization and OCR descriptor release pins."""\n\n'


class OcrFreezeVerificationError(RuntimeError):
    """A content-free two-commit freeze verification failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class OcrFreezeAttestation:
    """Verified source, descriptor, and receipt identities."""

    pin_commit_sha: str
    authorization_commit_sha: str
    descriptor_sha256: str
    authorization_receipt_sha256: str
    authorization_receipt_registry_key: str


def _pin_source(descriptor_sha256: str | None, authorization_commit_sha: str | None) -> bytes:
    descriptor_literal = "None" if descriptor_sha256 is None else repr(descriptor_sha256)
    authorization_literal = (
        "None" if authorization_commit_sha is None else repr(authorization_commit_sha)
    )
    return (
        _PIN_HEADER
        + "CANONICAL_OCR_DESCRIPTOR_SHA256: str | None = "
        + descriptor_literal
        + "\n"
        + "CANONICAL_OCR_AUTHORIZATION_COMMIT_SHA: str | None = "
        + authorization_literal
        + "\n"
    ).encode("utf-8")


def _git_bytes(repo_root: Path, arguments: Sequence[str]) -> bytes:
    candidate = shutil.which("git")
    if candidate is None:
        raise OcrFreezeVerificationError("ocr_release_freeze_git_unavailable")
    try:
        executable = Path(candidate).resolve(strict=True)
    except OSError as exc:
        raise OcrFreezeVerificationError("ocr_release_freeze_git_unavailable") from exc
    if not executable.is_file():
        raise OcrFreezeVerificationError("ocr_release_freeze_git_unavailable")
    try:
        completed = subprocess.run(  # noqa: S603
            [str(executable), "-C", str(repo_root), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OcrFreezeVerificationError("ocr_release_freeze_git_failed") from exc
    if completed.returncode != 0:
        raise OcrFreezeVerificationError("ocr_release_freeze_git_failed")
    return completed.stdout


def _regular_bytes(path: Path, *, max_bytes: int, code: str) -> tuple[Path, bytes]:
    try:
        if path.is_symlink():
            raise OcrFreezeVerificationError(code)
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or resolved.is_symlink():
            raise OcrFreezeVerificationError(code)
        size = resolved.stat().st_size
        if size < 1 or size > max_bytes:
            raise OcrFreezeVerificationError(code)
        payload = resolved.read_bytes()
    except OcrFreezeVerificationError:
        raise
    except OSError as exc:
        raise OcrFreezeVerificationError(code) from exc
    if len(payload) != size:
        raise OcrFreezeVerificationError(code)
    return resolved, payload


def _json_object(payload: bytes, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OcrFreezeVerificationError(code) from exc
    if not isinstance(value, dict):
        raise OcrFreezeVerificationError(code)
    return value


def _line_set(payload: bytes, *, code: str) -> set[str]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise OcrFreezeVerificationError(code) from exc
    return {line for line in text.splitlines() if line}


def verify_ocr_release_freeze(
    *,
    repo_root: Path,
    descriptor_path: Path,
    authorization_receipt_path: Path,
    git_reader: GitReader = _git_bytes,
) -> OcrFreezeAttestation:
    """Verify a clean C1 whose only C0 delta is the two release pin literals."""

    descriptor_pin = release_pin.CANONICAL_OCR_DESCRIPTOR_SHA256
    authorization_pin = release_pin.CANONICAL_OCR_AUTHORIZATION_COMMIT_SHA
    if descriptor_pin is None or authorization_pin is None:
        raise OcrFreezeVerificationError("ocr_release_freeze_not_frozen")
    if SHA256.fullmatch(descriptor_pin) is None or COMMIT_SHA.fullmatch(authorization_pin) is None:
        raise OcrFreezeVerificationError("ocr_release_freeze_pin_invalid")

    try:
        resolved_repo = repo_root.resolve(strict=True)
    except OSError as exc:
        raise OcrFreezeVerificationError("ocr_release_freeze_repo_invalid") from exc
    if not resolved_repo.is_dir() or resolved_repo.is_symlink():
        raise OcrFreezeVerificationError("ocr_release_freeze_repo_invalid")

    try:
        top = (
            git_reader(resolved_repo, ("rev-parse", "--show-toplevel"))
            .decode("utf-8", errors="strict")
            .strip()
        )
        head = (
            git_reader(resolved_repo, ("rev-parse", "HEAD"))
            .decode("ascii", errors="strict")
            .strip()
        )
        status = git_reader(
            resolved_repo,
            ("status", "--porcelain=v1", "--untracked-files=all"),
        )
        parent_line = (
            git_reader(
                resolved_repo,
                ("rev-list", "--parents", "-n", "1", "HEAD"),
            )
            .decode("ascii", errors="strict")
            .strip()
        )
    except (UnicodeDecodeError, OcrFreezeVerificationError):
        raise
    if Path(top).resolve() != resolved_repo:
        raise OcrFreezeVerificationError("ocr_release_freeze_repo_invalid")
    if COMMIT_SHA.fullmatch(head) is None or status:
        raise OcrFreezeVerificationError("ocr_release_freeze_c1_not_clean")
    if parent_line.split() != [head, authorization_pin]:
        raise OcrFreezeVerificationError("ocr_release_freeze_commit_topology_invalid")

    changed_paths = _line_set(
        git_reader(
            resolved_repo,
            (
                "diff",
                "--name-only",
                "--no-renames",
                authorization_pin,
                head,
                "--",
            ),
        ),
        code="ocr_release_freeze_diff_invalid",
    )
    if changed_paths != {PIN_RELATIVE_PATH}:
        raise OcrFreezeVerificationError("ocr_release_freeze_diff_invalid")
    c0_pin_source = git_reader(
        resolved_repo,
        ("show", f"{authorization_pin}:{PIN_RELATIVE_PATH}"),
    )
    c1_pin_source = git_reader(
        resolved_repo,
        ("show", f"{head}:{PIN_RELATIVE_PATH}"),
    )
    if c0_pin_source != _pin_source(None, None) or c1_pin_source != _pin_source(
        descriptor_pin,
        authorization_pin,
    ):
        raise OcrFreezeVerificationError("ocr_release_freeze_literal_delta_invalid")

    _, descriptor_payload = _regular_bytes(
        descriptor_path,
        max_bytes=MAX_DESCRIPTOR_BYTES,
        code="ocr_release_freeze_descriptor_invalid",
    )
    descriptor_sha256 = hashlib.sha256(descriptor_payload).hexdigest()
    if descriptor_sha256 != descriptor_pin:
        raise OcrFreezeVerificationError("ocr_release_freeze_descriptor_hash_mismatch")
    descriptor = _json_object(
        descriptor_payload,
        code="ocr_release_freeze_descriptor_invalid",
    )
    release_context = descriptor.get("release_context")
    if (
        descriptor.get("schema_version") != DESCRIPTOR_SCHEMA
        or descriptor.get("evaluation_version") != EVALUATION_VERSION
        or not isinstance(release_context, dict)
        or set(release_context)
        != {
            "authorization_commit_sha",
            "authorization_receipt_sha256",
            "authorization_receipt_registry_key",
        }
        or release_context.get("authorization_commit_sha") != authorization_pin
        or not isinstance(release_context.get("authorization_receipt_sha256"), str)
        or SHA256.fullmatch(release_context["authorization_receipt_sha256"]) is None
        or not isinstance(release_context.get("authorization_receipt_registry_key"), str)
        or SHA256.fullmatch(release_context["authorization_receipt_registry_key"]) is None
    ):
        raise OcrFreezeVerificationError("ocr_release_freeze_descriptor_context_invalid")
    if head.encode("ascii") in descriptor_payload or head.encode("ascii") in c1_pin_source:
        raise OcrFreezeVerificationError("ocr_release_freeze_circular_identity_invalid")

    resolved_receipt, receipt_payload = _regular_bytes(
        authorization_receipt_path,
        max_bytes=MAX_RECEIPT_BYTES,
        code="ocr_release_freeze_receipt_invalid",
    )
    receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
    receipt = _json_object(receipt_payload, code="ocr_release_freeze_receipt_invalid")
    required_receipt_keys = {
        "schema_version",
        "authorization_id",
        "authorization_sha256",
        "authorized_run_id",
        "repository_commit_sha",
        "sealed_bundle_sha256",
        "registry_key",
        "consumed_at_utc",
        "status",
    }
    registry_key = hashlib.sha256(
        json.dumps(
            {
                "authorization_sha256": receipt.get("authorization_sha256"),
                "repository_commit_sha": authorization_pin,
                "sealed_bundle_sha256": receipt.get("sealed_bundle_sha256"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if (
        set(receipt) != required_receipt_keys
        or receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("status") != "consumed_before_holdout_access"
        or receipt.get("repository_commit_sha") != authorization_pin
        or not isinstance(receipt.get("authorization_sha256"), str)
        or SHA256.fullmatch(receipt["authorization_sha256"]) is None
        or not isinstance(receipt.get("sealed_bundle_sha256"), str)
        or SHA256.fullmatch(receipt["sealed_bundle_sha256"]) is None
        or receipt.get("sealed_bundle_sha256") != descriptor.get("sealed_bundle_sha256")
        or receipt.get("registry_key") != registry_key
        or resolved_receipt.stem != registry_key
        or tuple(resolved_receipt.parent.parts[-3:])
        != ("holdout-consumption-registry", DATASET_ID, EVALUATION_VERSION)
        or release_context["authorization_receipt_sha256"] != receipt_sha256
        or release_context["authorization_receipt_registry_key"] != registry_key
    ):
        raise OcrFreezeVerificationError("ocr_release_freeze_receipt_mismatch")
    return OcrFreezeAttestation(
        pin_commit_sha=head,
        authorization_commit_sha=authorization_pin,
        descriptor_sha256=descriptor_sha256,
        authorization_receipt_sha256=receipt_sha256,
        authorization_receipt_registry_key=registry_key,
    )
