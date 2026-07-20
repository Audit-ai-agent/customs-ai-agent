from __future__ import annotations

from pathlib import Path

import pytest

from evals.release_holdout_protocol import (
    RELEASE_INVALIDATION_CODE,
    REPLACEMENT_EVALUATION_VERSION,
    REPLACEMENT_NOT_PROVISIONED_CODE,
)
from evals.run_release_holdout_validation import (
    ReleaseHoldoutError,
    evaluate_release_holdout,
)


def _run(
    *,
    evaluation_version: str = "evaluation-v2",
    git_state_provider=None,
):
    arguments = {
        "repo_root": Path("must-not-be-read/repo"),
        "dataset_root": Path("must-not-be-read/dataset"),
        "splits_path": Path("must-not-be-read/splits.json"),
        "dataset_lock_path": Path("must-not-be-read/dataset.lock.json"),
        "authorization_path": Path("must-not-be-read/authorization.json"),
        "run_id": "release-holdout-once",
        "evaluation_version": evaluation_version,
    }
    if git_state_provider is not None:
        arguments["git_state_provider"] = git_state_provider
    return evaluate_release_holdout(**arguments)


def test_evaluation_v2_is_permanently_rejected_before_any_access() -> None:
    git_state_called = False

    def git_state(_root: Path) -> tuple[str, bool]:
        nonlocal git_state_called
        git_state_called = True
        raise AssertionError("invalidated v2 must stop before Git or artifact access")

    with pytest.raises(ReleaseHoldoutError, match=RELEASE_INVALIDATION_CODE):
        _run(git_state_provider=git_state)

    assert git_state_called is False


def test_replacement_stays_blocked_until_sealed_runner_is_implemented() -> None:
    with pytest.raises(
        ReleaseHoldoutError,
        match=REPLACEMENT_NOT_PROVISIONED_CODE,
    ):
        _run(evaluation_version=REPLACEMENT_EVALUATION_VERSION)


def test_unknown_release_evaluation_version_is_rejected() -> None:
    with pytest.raises(
        ReleaseHoldoutError,
        match="release_holdout_evaluation_version_unsupported",
    ):
        _run(evaluation_version="evaluation-v999")
