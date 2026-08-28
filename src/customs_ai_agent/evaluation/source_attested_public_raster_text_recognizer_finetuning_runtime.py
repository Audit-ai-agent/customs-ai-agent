"""Offline orchestration for the bounded public-recognizer fine-tune.

This module owns the process lock, network denial, exact native import root,
bundle materialization and post-run provenance verification.  It intentionally
does not select or activate a serving profile; a published overfit release is
only evidence for deciding whether a larger pilot is justified.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import sys
from typing import Literal

from customs_ai_agent.evaluation import (
    source_attested_public_raster_text_recognizer_finetuning as implementation,
)
from customs_ai_agent.evaluation import source_attested_public_raster_text_corpus
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
    PublicFineTuneRunResultV1,
    PublicFineTuneReceiptV2,
    SourceAttestedPublicRecognizerFineTuneError,
    build_verified_public_finetune_leakage_safe_corpus_v2,
    load_public_mobile_recognizer_dictionary_v1,
    load_verified_public_finetuned_recognizer_release_v1,
    sha256_file,
    train_source_attested_public_recognizer_leakage_safe_v2,
    train_source_attested_public_recognizer_overfit_v1,
)
from customs_ai_agent.evaluation.source_attested_raster_text_recognizer_training_runtime import (
    network_disabled,
    verified_direct_training_imports,
)


_DOCUMENT_PREVIEW_ROOT_SUFFIX = ("_local", "data", "document-preview")
_LOCK_FILE = ".source-attested-public-raster-text-recognizer-finetuning.lock"


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
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_document_preview_root_invalid"
        ) from None


def _verify_release_provenance(
    root: Path,
    result: PublicFineTuneRunResultV1,
) -> PublicFineTuneRunResultV1:
    """Re-load every content-addressed input and bind it to the receipt."""

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
        release = load_verified_public_finetuned_recognizer_release_v1(
            root, result.release.release_sha256
        )
        module_value = implementation.__file__
        corpus_module_value = source_attested_public_raster_text_corpus.__file__
        source_crop_base_value = source_crop_base.__file__
        if (
            not isinstance(module_value, str)
            or not isinstance(corpus_module_value, str)
            or not isinstance(source_crop_base_value, str)
        ):
            raise ValueError
        module_path = Path(module_value).resolve(strict=True)
        corpus_module_path = Path(corpus_module_value).resolve(strict=True)
        source_crop_base_path = Path(source_crop_base_value).resolve(strict=True)
        inventory_relative = runtime.manifest.sidecar_package_inventory_relative_path
        inventory_path = runtime.bundle_root.joinpath(
            *PurePosixPath(inventory_relative).parts
        ).resolve(strict=True)
        dictionary = load_public_mobile_recognizer_dictionary_v1(mobile)
        if isinstance(receipt, PublicFineTuneReceiptV2):
            corpus = build_verified_public_finetune_leakage_safe_corpus_v2(
                root, dictionary=dictionary
            )
            if (
                corpus.train_manifest != receipt.train_crop_manifest
                or corpus.development_manifest != receipt.development_crop_manifest
            ):
                raise ValueError
        if (
            release.receipt != receipt
            or receipt.parent_python_revision
            != f"{sys.version_info.major}.{sys.version_info.minor}"
            or sha256_file(module_path) != receipt.training_module_sha256
            or sha256_file(corpus_module_path) != receipt.balanced_corpus_module_sha256
            or sha256_file(source_crop_base_path) != receipt.source_crop_base_module_sha256
            or runtime.file_sha256(inventory_relative)
            != receipt.direct_runtime_package_inventory_sha256
            or sha256_file(inventory_path) != receipt.direct_runtime_package_inventory_sha256
            or mobile.manifest.source_model_inventory_sha256
            != receipt.public_mobile_source_inventory_sha256
            or mobile.manifest.source_model_snapshot_sha256
            != receipt.public_mobile_source_snapshot_sha256
            or dictionary.dictionary_sha256 != receipt.base_dictionary_sha256
            or dictionary.output_class_count != receipt.output_class_count
            or dictionary.implicit_space_class_index != receipt.implicit_space_class_index
        ):
            raise ValueError
        for relative, expected in receipt.public_recognizer_artifact_sha256s.items():
            if mobile.file_sha256(relative) != expected:
                raise ValueError
        return PublicFineTuneRunResultV1(release=release, reused=result.reused)
    except SourceAttestedPublicRecognizerFineTuneError:
        raise
    except (
        OSError,
        ValueError,
        RasterTextDirectPaddleBundleError,
        RasterTextMobilePaddleBundleError,
    ):
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_release_provenance_drift"
        ) from None


def ensure_source_attested_public_recognizer_overfit_v1(
    document_preview_root: Path,
    *,
    tuning_scope: Literal["head", "connected_tail"] = "head",
) -> PublicFineTuneRunResultV1:
    """Create or reuse the bounded 1k release with no network or user hashes."""

    root = _root(document_preview_root)
    if tuning_scope not in {"head", "connected_tail"}:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_scope_invalid")
    try:
        with exclusive_local_process_lock(root / _LOCK_FILE):
            with network_disabled():
                runtime = materialize_verified_raster_text_direct_paddle_bundle_v1(
                    document_preview_root=root
                )
                mobile = materialize_verified_raster_text_direct_mobile_paddle_bundle_v1(
                    document_preview_root=root
                )
                with verified_direct_training_imports(runtime) as inventory_path:
                    result = train_source_attested_public_recognizer_overfit_v1(
                        root,
                        runtime_bundle=runtime,
                        mobile_bundle=mobile,
                        package_inventory_path=inventory_path,
                        tuning_scope=tuning_scope,
                    )
                return _verify_release_provenance(root, result)
    except LocalProcessLockBusy:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_locked") from None
    except LocalProcessLockInvalid:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_lock_invalid") from None
    except SourceAttestedPublicRecognizerFineTuneError:
        raise
    except (
        RasterTextDirectPaddleBundleError,
        RasterTextMobilePaddleBundleError,
    ):
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_bundle_unavailable"
        ) from None


def ensure_source_attested_public_recognizer_leakage_safe_v2(
    document_preview_root: Path,
    *,
    tuning_scope: Literal["head"] = "head",
) -> PublicFineTuneRunResultV1:
    """Create/reuse the fixed four-epoch, selection-V2-isolated head release."""

    root = _root(document_preview_root)
    if tuning_scope != "head":
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_scope_invalid")
    try:
        with exclusive_local_process_lock(root / _LOCK_FILE):
            with network_disabled():
                runtime = materialize_verified_raster_text_direct_paddle_bundle_v1(
                    document_preview_root=root
                )
                mobile = materialize_verified_raster_text_direct_mobile_paddle_bundle_v1(
                    document_preview_root=root
                )
                with verified_direct_training_imports(runtime) as inventory_path:
                    result = train_source_attested_public_recognizer_leakage_safe_v2(
                        root,
                        runtime_bundle=runtime,
                        mobile_bundle=mobile,
                        package_inventory_path=inventory_path,
                    )
                return _verify_release_provenance(root, result)
    except LocalProcessLockBusy:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_locked") from None
    except LocalProcessLockInvalid:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_lock_invalid") from None
    except SourceAttestedPublicRecognizerFineTuneError:
        raise
    except (
        RasterTextDirectPaddleBundleError,
        RasterTextMobilePaddleBundleError,
    ):
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_bundle_unavailable"
        ) from None


__all__ = [
    "ensure_source_attested_public_recognizer_leakage_safe_v2",
    "ensure_source_attested_public_recognizer_overfit_v1",
]
