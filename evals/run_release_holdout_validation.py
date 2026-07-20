"""Fail-closed release-holdout entry point.

The repository-visible evaluation-v2 selection is permanently compromised:
its membership is derivable from public inputs.  This module deliberately
cannot read that split or evaluate it.  A future implementation must consume a
human-supplied evaluation-v3 sealed bundle under ``release_holdout_protocol``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from evals.release_holdout_protocol import (
    INVALIDATED_EVALUATION_VERSION,
    RELEASE_INVALIDATION_CODE,
    REPLACEMENT_NOT_PROVISIONED_CODE,
    ReleaseHoldoutProtocolError,
    require_release_eligible_version,
)


REPORT_SCHEMA = "release-holdout-evaluation-report/v2"
AUTHORIZATION_SCHEMA = "release-holdout-authorization/v1"
RECEIPT_SCHEMA = "release-holdout-authorization-consumption/v1"


class ReleaseHoldoutError(ValueError):
    """Release holdout preconditions were not met."""


def _git_state(repo_root: Path) -> tuple[str, bool]:
    """Return exact worktree identity without invoking a shell.

    Kept as a reusable security primitive for the future sealed-bundle runner.
    It is intentionally not called for invalidated evaluation-v2 input.
    """

    candidate = shutil.which("git")
    if candidate is None:
        raise ReleaseHoldoutError("exact Git state cannot be established")
    try:
        git_executable = Path(candidate).resolve(strict=True)
    except OSError as exc:
        raise ReleaseHoldoutError("exact Git state cannot be established") from exc
    if not git_executable.is_file():
        raise ReleaseHoldoutError("exact Git state cannot be established")
    try:
        top = subprocess.run(  # noqa: S603
            [str(git_executable), "-C", str(repo_root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            shell=False,
        ).stdout.strip()
        commit = subprocess.run(  # noqa: S603
            [str(git_executable), "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            shell=False,
        ).stdout.strip()
        status = subprocess.run(  # noqa: S603
            [
                str(git_executable),
                "-C",
                str(repo_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            shell=False,
        ).stdout
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        raise ReleaseHoldoutError("exact Git state cannot be established") from exc
    if Path(top).resolve() != repo_root.resolve():
        raise ReleaseHoldoutError("repo root does not match the exact Git worktree root")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ReleaseHoldoutError("Git HEAD is not a full commit SHA")
    return commit, not bool(status)


def evaluate_release_holdout(
    *,
    repo_root: Path,
    dataset_root: Path,
    splits_path: Path,
    dataset_lock_path: Path,
    authorization_path: Path,
    run_id: str,
    evaluation_version: str = INVALIDATED_EVALUATION_VERSION,
    now: datetime | None = None,
    git_state_provider: Callable[[Path], tuple[str, bool]] = _git_state,
) -> tuple[int, dict[str, Any]]:
    """Reject compromised v2 and block v3 until its sealed runner exists.

    All path-like arguments remain for CLI compatibility, but this function
    must not stat, hash, parse, or enumerate them.  In particular, no locked
    membership can be exposed during this preflight.
    """

    del (
        repo_root,
        dataset_root,
        splits_path,
        dataset_lock_path,
        authorization_path,
        run_id,
        now,
        git_state_provider,
    )
    try:
        require_release_eligible_version(evaluation_version)
    except ReleaseHoldoutProtocolError as exc:
        raise ReleaseHoldoutError(exc.code) from exc
    raise ReleaseHoldoutError(REPLACEMENT_NOT_PROVISIONED_CODE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed release holdout preflight.")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--dataset-lock", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--evaluation-version",
        default=INVALIDATED_EVALUATION_VERSION,
    )
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        exit_code, report = evaluate_release_holdout(
            repo_root=args.repo_root,
            dataset_root=args.dataset_root,
            splits_path=args.splits,
            dataset_lock_path=args.dataset_lock,
            authorization_path=args.authorization,
            run_id=args.run_id,
            evaluation_version=args.evaluation_version,
        )
    except ReleaseHoldoutError as exc:
        print(
            json.dumps(
                {
                    "schema_version": REPORT_SCHEMA,
                    "scope": "ONE_TIME_RELEASE_HOLDOUT",
                    "status": "NOT_EXECUTED",
                    "release_eligible": False,
                    "error_code": str(exc),
                    "invalidated_evaluation_version": (
                        INVALIDATED_EVALUATION_VERSION
                        if str(exc) == RELEASE_INVALIDATION_CODE
                        else None
                    ),
                },
                sort_keys=True,
            )
        )
        return 2
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
