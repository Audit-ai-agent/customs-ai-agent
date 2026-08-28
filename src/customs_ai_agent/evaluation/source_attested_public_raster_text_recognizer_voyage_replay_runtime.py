"""Offline, fail-closed orchestration for the fixed Voyage replay release."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import sys

from customs_ai_agent.evaluation import (
    source_attested_public_raster_text_recognizer_voyage_replay as implementation,
)
from customs_ai_agent.evaluation import (
    source_attested_public_raster_text_voyage_replay_corpus as corpus_source,
)
from customs_ai_agent.evaluation import (
    source_attested_raster_text_recognizer_training as source_crop_base,
)
from customs_ai_agent.evaluation.local_process_lock import (
    LocalProcessLockBusy,
    LocalProcessLockInvalid,
    exclusive_local_process_lock,
)
from customs_ai_agent.evaluation.raster_text_direct_paddle_bundle import (
    RasterTextDirectPaddleBundleError,
    load_verified_raster_text_direct_paddle_bundle_v1,
    materialize_verified_raster_text_direct_paddle_bundle_v1,
)
from customs_ai_agent.evaluation.raster_text_mobile_paddle_bundle import (
    RasterTextMobilePaddleBundleError,
    load_verified_raster_text_direct_mobile_paddle_bundle_v1,
    materialize_verified_raster_text_direct_mobile_paddle_bundle_v1,
)
from customs_ai_agent.evaluation.source_attested_public_raster_text_recognizer_finetuning import (
    load_public_mobile_recognizer_dictionary_v1,
)
from customs_ai_agent.evaluation.source_attested_public_raster_text_recognizer_voyage_replay import (
    VoyageReplayError,
    VoyageReplayRunResultV1,
    build_verified_voyage_replay_corpus,
    load_verified_voyage_replay_release_v1,
    train_source_attested_public_recognizer_voyage_replay_v1,
)
from customs_ai_agent.evaluation.source_attested_raster_text_recognizer_training import (
    sha256_file,
)
from customs_ai_agent.evaluation.source_attested_raster_text_recognizer_training_runtime import (
    network_disabled,
    verified_direct_training_imports,
)


_DOCUMENT_PREVIEW_ROOT_SUFFIX = ("_local", "data", "document-preview")
_LOCK_FILE = ".source-attested-public-raster-text-voyage-replay.lock"
_OUTPUT_MODEL_NAME = "source_attested_en_PP-OCRv5_mobile_rec_voyage_replay_head_v3"


def _root(path: Path) -> Path:
    try:
        if path.is_symlink():
            raise OSError
        root = path.resolve(strict=True)
        if (
            root.is_symlink()
            or not root.is_dir()
            or tuple(part.casefold() for part in root.parts[-3:]) != _DOCUMENT_PREVIEW_ROOT_SUFFIX
        ):
            raise OSError
        return root
    except OSError:
        raise VoyageReplayError("voyage_replay_document_preview_root_invalid") from None


def _module_path(module: object) -> Path:
    value = getattr(module, "__file__", None)
    if not isinstance(value, str):
        raise VoyageReplayError("voyage_replay_release_provenance_drift")
    try:
        return Path(value).resolve(strict=True)
    except OSError:
        raise VoyageReplayError("voyage_replay_release_provenance_drift") from None


def _verify_release_provenance(
    root: Path,
    result: VoyageReplayRunResultV1,
) -> VoyageReplayRunResultV1:
    """Rebuild every non-secret commitment after publish or idempotent reuse."""

    receipt = result.release.receipt
    try:
        runtime = load_verified_raster_text_direct_paddle_bundle_v1(
            document_preview_root=root,
            manifest_sha256=receipt.direct_runtime_bundle_manifest_sha256,
        )
        mobile = load_verified_raster_text_direct_mobile_paddle_bundle_v1(
            document_preview_root=root,
            manifest_sha256=receipt.public_mobile_model_bundle_manifest_sha256,
        )
        parent = implementation._verified_parent_release(root)
        release = load_verified_voyage_replay_release_v1(root, result.release.release_sha256)
        corpus, corpus_path = build_verified_voyage_replay_corpus(root)
        inventory_relative = runtime.manifest.sidecar_package_inventory_relative_path
        inventory_path = runtime.bundle_root.joinpath(
            *PurePosixPath(inventory_relative).parts
        ).resolve(strict=True)
        dictionary = load_public_mobile_recognizer_dictionary_v1(mobile)
        config = json.loads(release.file_paths["text-rec/config.json"].read_bytes())
        global_value = config.get("Global") if isinstance(config, dict) else None
        if (
            release.receipt != receipt
            or corpus.train_manifest != receipt.train_corpus_manifest
            or corpus.development_manifest != receipt.development_corpus_manifest
            or receipt.parent_python_revision
            != f"{sys.version_info.major}.{sys.version_info.minor}"
            or sha256_file(_module_path(implementation)) != receipt.training_module_sha256
            or sha256_file(corpus_path) != receipt.corpus_module_sha256
            or corpus_path != _module_path(corpus_source)
            or sha256_file(_module_path(source_crop_base)) != receipt.source_crop_base_module_sha256
            or runtime.file_sha256(inventory_relative)
            != receipt.direct_runtime_package_inventory_sha256
            or sha256_file(inventory_path) != receipt.direct_runtime_package_inventory_sha256
            or dictionary.dictionary_sha256 != receipt.base_dictionary_sha256
            or dictionary.output_class_count != receipt.output_class_count
            or parent.receipt.receipt_sha256 != receipt.parent_receipt_sha256
            or parent.release_sha256 != receipt.parent_release_sha256
            or dict(parent.receipt.model_artifact_sha256s) != receipt.parent_model_artifact_sha256s
            or receipt.model_artifact_sha256s
            != {
                relative: release.file_sha256s[relative]
                for relative in receipt.model_artifact_sha256s
            }
            or not isinstance(global_value, dict)
            or global_value.get("model_name") != _OUTPUT_MODEL_NAME
        ):
            raise ValueError
        return VoyageReplayRunResultV1(release=release, reused=result.reused)
    except VoyageReplayError:
        raise
    except (
        KeyError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        RasterTextDirectPaddleBundleError,
        RasterTextMobilePaddleBundleError,
    ):
        raise VoyageReplayError("voyage_replay_release_provenance_drift") from None


def ensure_source_attested_public_recognizer_voyage_replay_v1(
    document_preview_root: Path,
) -> VoyageReplayRunResultV1:
    """Create or reuse the one fixed two-epoch replay with network denied."""

    root = _root(document_preview_root)
    try:
        with exclusive_local_process_lock(root / _LOCK_FILE):
            with network_disabled():
                runtime = materialize_verified_raster_text_direct_paddle_bundle_v1(
                    document_preview_root=root
                )
                mobile = materialize_verified_raster_text_direct_mobile_paddle_bundle_v1(
                    document_preview_root=root
                )
                # Parent provenance is checked before any Paddle/native import.
                implementation._verified_parent_release(root)
                with verified_direct_training_imports(runtime) as inventory_path:
                    result = train_source_attested_public_recognizer_voyage_replay_v1(
                        root,
                        runtime_bundle=runtime,
                        mobile_bundle=mobile,
                        package_inventory_path=inventory_path,
                    )
                return _verify_release_provenance(root, result)
    except LocalProcessLockBusy:
        raise VoyageReplayError("voyage_replay_locked") from None
    except LocalProcessLockInvalid:
        raise VoyageReplayError("voyage_replay_lock_invalid") from None
    except VoyageReplayError:
        raise
    except (
        RasterTextDirectPaddleBundleError,
        RasterTextMobilePaddleBundleError,
    ):
        raise VoyageReplayError("voyage_replay_bundle_unavailable") from None


__all__ = ["ensure_source_attested_public_recognizer_voyage_replay_v1"]
