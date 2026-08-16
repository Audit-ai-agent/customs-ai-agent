from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from customs_ai_agent.service.document_preview_runtime import DocumentPreviewRuntimeError
from tools import stage_docling_accurate_tableformer as staging_tool


class _FakeResponse:
    def __init__(
        self, *, payload: bytes, final_url: str, content_length: str | None = None
    ) -> None:
        self._payload = payload
        self._offset = 0
        self._final_url = final_url
        self.headers: dict[str, str] = {}
        if content_length is not None:
            self.headers["Content-Length"] = content_length

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def geturl(self) -> str:
        return self._final_url

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._payload) - self._offset
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace" / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    return root


def _git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def _test_artifacts() -> tuple[tuple[staging_tool._PinnedArtifact, ...], dict[str, bytes]]:
    config = b"{}"
    weight = b"accurate-model-test-bytes"
    artifacts = (
        staging_tool._PinnedArtifact(
            filename="tm_config.json",
            source_uri="https://huggingface.co/ds4sd/docling-models/resolve/test/model_artifacts/tableformer/accurate/tm_config.json",
            byte_size=len(config),
            git_blob_sha1=_git_blob_sha1(config),
        ),
        staging_tool._PinnedArtifact(
            filename="tableformer_accurate.safetensors",
            source_uri="https://huggingface.co/ds4sd/docling-models/resolve/test/model_artifacts/tableformer/accurate/tableformer_accurate.safetensors",
            byte_size=len(weight),
            sha256=hashlib.sha256(weight).hexdigest(),
        ),
    )
    return artifacts, {
        artifact.source_uri: payload for artifact, payload in zip(artifacts, (config, weight))
    }


def _install_test_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, bytes]:
    artifacts, payloads = _test_artifacts()
    monkeypatch.setattr(staging_tool, "_ARTIFACTS", artifacts)
    monkeypatch.setattr(
        staging_tool,
        "_ALLOWED_SOURCE_URIS",
        frozenset(artifact.source_uri for artifact in artifacts),
    )
    monkeypatch.setattr(staging_tool, "_check_free_space", lambda *_args: None)
    return payloads


def test_stages_only_the_verified_pair_without_runtime_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    payloads = _install_test_artifacts(monkeypatch)
    requested: list[str] = []

    def opener(request: Any, *, timeout: int) -> _FakeResponse:
        assert timeout == 30
        uri = request.full_url
        requested.append(uri)
        return _FakeResponse(
            payload=payloads[uri],
            final_url=uri.replace("/ds4sd/", "/docling-project/"),
            content_length=str(len(payloads[uri])),
        )

    staging_tool.stage_docling_accurate_tableformer(
        document_preview_root=root,
        opener=opener,
    )

    target = root.joinpath(*staging_tool._TARGET_RELATIVE_PARTS)
    assert {path.name for path in target.iterdir()} == {
        "tm_config.json",
        "tableformer_accurate.safetensors",
    }
    assert (target / "tm_config.json").read_bytes() == payloads[requested[0]]
    assert (target / "tableformer_accurate.safetensors").read_bytes() == payloads[requested[1]]
    assert requested == list(payloads)
    assert not (root / "runtime-configurations").exists()
    assert not (root / "model-inventories").exists()
    assert not (root / "candidate-outputs").exists()
    assert not (root / "evaluation-results").exists()
    assert not (root / "source-attested-evaluation-results").exists()
    assert not (root / "approvals").exists()


def test_rejects_checksum_mismatch_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    payloads = _install_test_artifacts(monkeypatch)
    weight_uri = next(uri for uri in payloads if uri.endswith(".safetensors"))
    payloads[weight_uri] = b"x" * len(payloads[weight_uri])

    def opener(request: Any, *, timeout: int) -> _FakeResponse:
        uri = request.full_url
        return _FakeResponse(
            payload=payloads[uri], final_url=uri, content_length=str(len(payloads[uri]))
        )

    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_docling_accurate_download_invalid",
    ):
        staging_tool.stage_docling_accurate_tableformer(document_preview_root=root, opener=opener)

    assert not root.joinpath(*staging_tool._TARGET_RELATIVE_PARTS).exists()
    staging = root / "staging"
    assert not any(path.name.startswith(".docling-accurate-") for path in staging.iterdir())


def test_rejects_an_existing_target_without_network_or_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    _install_test_artifacts(monkeypatch)
    target = root.joinpath(*staging_tool._TARGET_RELATIVE_PARTS)
    target.mkdir(parents=True)
    existing = target / "unexpected.bin"
    existing.write_bytes(b"keep")

    def opener(*_args: Any, **_kwargs: Any) -> _FakeResponse:
        pytest.fail("existing target must stop before network access")

    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_docling_accurate_target_exists",
    ):
        staging_tool.stage_docling_accurate_tableformer(document_preview_root=root, opener=opener)
    assert existing.read_bytes() == b"keep"


def test_rejects_unpinned_source_before_opening_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    payloads = _install_test_artifacts(monkeypatch)
    artifacts = list(staging_tool._ARTIFACTS)
    artifacts[0] = replace(artifacts[0], source_uri="https://example.test/tm_config.json")
    monkeypatch.setattr(staging_tool, "_ARTIFACTS", tuple(artifacts))

    def opener(*_args: Any, **_kwargs: Any) -> _FakeResponse:
        pytest.fail("an unpinned URL must stop before network access")

    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_docling_accurate_source_invalid",
    ):
        staging_tool.stage_docling_accurate_tableformer(document_preview_root=root, opener=opener)
    assert not root.joinpath(*staging_tool._TARGET_RELATIVE_PARTS).exists()
    assert payloads


def test_rejects_symlinked_target_parent_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    _install_test_artifacts(monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "models"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("the current Windows test account cannot create directory symlinks")

    def opener(*_args: Any, **_kwargs: Any) -> _FakeResponse:
        pytest.fail("a redirecting target parent must stop before network access")

    with pytest.raises(
        DocumentPreviewRuntimeError, match="document_preview_docling_accurate_target_exists"
    ):
        staging_tool.stage_docling_accurate_tableformer(document_preview_root=root, opener=opener)
    assert not (outside / "docling").exists()


def test_cli_requires_explicit_download_confirmation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "workspace" / "_local" / "data" / "document-preview"

    assert staging_tool.main(["--document-preview-root", str(root)]) == 2

    assert json.loads(capsys.readouterr().out) == {
        "status": "failed",
        "code": "confirmation_required",
    }
    assert not root.exists()


def test_rejects_invalid_free_space_budget(tmp_path: Path) -> None:
    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_docling_accurate_space_budget_invalid",
    ):
        staging_tool._check_free_space(_root(tmp_path), 0)
