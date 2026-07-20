from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import customs_ai_agent.ocr_release_pin as release_pin
from customs_ai_agent.ocr_freeze import (
    OcrFreezeVerificationError,
    _pin_source,
    verify_ocr_release_freeze,
)


def _write_freeze_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, str, str, dict[tuple[str, ...], bytes]]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    authorization_commit = "a" * 40
    pin_commit = "b" * 40
    sealed_bundle_sha256 = "c" * 64
    authorization_sha256 = "d" * 64
    registry_key = hashlib.sha256(
        json.dumps(
            {
                "authorization_sha256": authorization_sha256,
                "repository_commit_sha": authorization_commit,
                "sealed_bundle_sha256": sealed_bundle_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    receipt = {
        "schema_version": "release-holdout-authorization-consumption/v2",
        "authorization_id": "authorization-1",
        "authorization_sha256": authorization_sha256,
        "authorized_run_id": "release-run-1",
        "repository_commit_sha": authorization_commit,
        "sealed_bundle_sha256": sealed_bundle_sha256,
        "registry_key": registry_key,
        "consumed_at_utc": "2026-07-18T00:00:00Z",
        "status": "consumed_before_holdout_access",
    }
    receipt_path = (
        tmp_path
        / "holdout-consumption-registry"
        / "trade_document_triplets"
        / "evaluation-v3"
        / f"{registry_key}.json"
    )
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    descriptor = {
        "schema_version": "ocr-benchmark-descriptor/v3",
        "evaluation_version": "evaluation-v3",
        "sealed_bundle_sha256": sealed_bundle_sha256,
        "release_context": {
            "authorization_commit_sha": authorization_commit,
            "authorization_receipt_sha256": receipt_sha256,
            "authorization_receipt_registry_key": registry_key,
        },
    }
    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    descriptor_sha256 = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        release_pin,
        "CANONICAL_OCR_DESCRIPTOR_SHA256",
        descriptor_sha256,
    )
    monkeypatch.setattr(
        release_pin,
        "CANONICAL_OCR_AUTHORIZATION_COMMIT_SHA",
        authorization_commit,
    )
    git_results = {
        ("rev-parse", "--show-toplevel"): (str(repo_root.resolve()) + "\n").encode(),
        ("rev-parse", "HEAD"): (pin_commit + "\n").encode(),
        ("status", "--porcelain=v1", "--untracked-files=all"): b"",
        ("rev-list", "--parents", "-n", "1", "HEAD"): (
            f"{pin_commit} {authorization_commit}\n"
        ).encode(),
        (
            "diff",
            "--name-only",
            "--no-renames",
            authorization_commit,
            pin_commit,
            "--",
        ): b"src/customs_ai_agent/ocr_release_pin.py\n",
        (
            "show",
            f"{authorization_commit}:src/customs_ai_agent/ocr_release_pin.py",
        ): _pin_source(None, None),
        (
            "show",
            f"{pin_commit}:src/customs_ai_agent/ocr_release_pin.py",
        ): _pin_source(descriptor_sha256, authorization_commit),
    }
    return (
        repo_root,
        descriptor_path,
        receipt_path,
        authorization_commit,
        pin_commit,
        git_results,
    )


def _reader(results: dict[tuple[str, ...], bytes]):
    def read(_repo_root: Path, arguments: tuple[str, ...]) -> bytes:
        return results[tuple(arguments)]

    return read


def test_two_commit_freeze_accepts_only_exact_c0_to_c1_pin_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, descriptor, receipt, authorization_commit, pin_commit, git_results = _write_freeze_inputs(
        tmp_path, monkeypatch
    )

    attestation = verify_ocr_release_freeze(
        repo_root=repo,
        descriptor_path=descriptor,
        authorization_receipt_path=receipt,
        git_reader=_reader(git_results),
    )

    assert attestation.authorization_commit_sha == authorization_commit
    assert attestation.pin_commit_sha == pin_commit
    assert attestation.descriptor_sha256 == hashlib.sha256(descriptor.read_bytes()).hexdigest()


def test_two_commit_freeze_rejects_c2_behavior_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, descriptor, receipt, _c0, c1, git_results = _write_freeze_inputs(
        tmp_path,
        monkeypatch,
    )
    c2 = "e" * 40
    git_results[("rev-parse", "HEAD")] = (c2 + "\n").encode()
    git_results[("rev-list", "--parents", "-n", "1", "HEAD")] = (f"{c2} {c1}\n").encode()

    with pytest.raises(
        OcrFreezeVerificationError,
        match="ocr_release_freeze_commit_topology_invalid",
    ):
        verify_ocr_release_freeze(
            repo_root=repo,
            descriptor_path=descriptor,
            authorization_receipt_path=receipt,
            git_reader=_reader(git_results),
        )


def test_two_commit_freeze_rejects_extra_file_or_nonliteral_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, descriptor, receipt, c0, c1, git_results = _write_freeze_inputs(
        tmp_path,
        monkeypatch,
    )
    diff_key = ("diff", "--name-only", "--no-renames", c0, c1, "--")
    git_results[diff_key] += b"src/customs_ai_agent/ocr_release.py\n"

    with pytest.raises(
        OcrFreezeVerificationError,
        match="ocr_release_freeze_diff_invalid",
    ):
        verify_ocr_release_freeze(
            repo_root=repo,
            descriptor_path=descriptor,
            authorization_receipt_path=receipt,
            git_reader=_reader(git_results),
        )

    git_results[diff_key] = b"src/customs_ai_agent/ocr_release_pin.py\n"
    show_key = ("show", f"{c1}:src/customs_ai_agent/ocr_release_pin.py")
    git_results[show_key] += b"BEHAVIOR = True\n"
    with pytest.raises(
        OcrFreezeVerificationError,
        match="ocr_release_freeze_literal_delta_invalid",
    ):
        verify_ocr_release_freeze(
            repo_root=repo,
            descriptor_path=descriptor,
            authorization_receipt_path=receipt,
            git_reader=_reader(git_results),
        )


def test_two_commit_freeze_rejects_dirty_c1_and_receipt_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, descriptor, receipt, _c0, _c1, git_results = _write_freeze_inputs(
        tmp_path,
        monkeypatch,
    )
    git_results[("status", "--porcelain=v1", "--untracked-files=all")] = b" M source.py\n"
    with pytest.raises(
        OcrFreezeVerificationError,
        match="ocr_release_freeze_c1_not_clean",
    ):
        verify_ocr_release_freeze(
            repo_root=repo,
            descriptor_path=descriptor,
            authorization_receipt_path=receipt,
            git_reader=_reader(git_results),
        )

    git_results[("status", "--porcelain=v1", "--untracked-files=all")] = b""
    descriptor_value = json.loads(descriptor.read_text(encoding="utf-8"))
    descriptor_value["release_context"]["authorization_receipt_sha256"] = "f" * 64
    descriptor.write_text(
        json.dumps(descriptor_value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        release_pin,
        "CANONICAL_OCR_DESCRIPTOR_SHA256",
        hashlib.sha256(descriptor.read_bytes()).hexdigest(),
    )
    show_key = ("show", f"{'b' * 40}:src/customs_ai_agent/ocr_release_pin.py")
    git_results[show_key] = _pin_source(
        release_pin.CANONICAL_OCR_DESCRIPTOR_SHA256,
        release_pin.CANONICAL_OCR_AUTHORIZATION_COMMIT_SHA,
    )
    with pytest.raises(
        OcrFreezeVerificationError,
        match="ocr_release_freeze_receipt_mismatch",
    ):
        verify_ocr_release_freeze(
            repo_root=repo,
            descriptor_path=descriptor,
            authorization_receipt_path=receipt,
            git_reader=_reader(git_results),
        )


def test_two_commit_freeze_remains_not_frozen_without_both_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(release_pin, "CANONICAL_OCR_DESCRIPTOR_SHA256", None)
    monkeypatch.setattr(
        release_pin,
        "CANONICAL_OCR_AUTHORIZATION_COMMIT_SHA",
        None,
    )

    with pytest.raises(
        OcrFreezeVerificationError,
        match="ocr_release_freeze_not_frozen",
    ):
        verify_ocr_release_freeze(
            repo_root=tmp_path / "must-not-be-read",
            descriptor_path=tmp_path / "must-not-be-read",
            authorization_receipt_path=tmp_path / "must-not-be-read",
        )
