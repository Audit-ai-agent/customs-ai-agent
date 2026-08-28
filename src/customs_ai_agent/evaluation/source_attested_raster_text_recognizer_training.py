"""Offline, source-attested training for a small raster text recognizer.

The module never serializes Gold text into manifests, receipts, or stdout.
Labels are resolved from verified train/development Gold only while a process
is training.  Published releases are immutable and content addressed.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import sys
from typing import Any, Callable, Generic, Literal, Mapping, Sequence, TypeVar
import unicodedata
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator


_SHA256 = r"^[a-f0-9]{64}$"
_RELEASE_COLLECTION = "raster-text-trained-recognizers-v1"
_RELEASE_FILES = (
    "text-rec/config.json",
    "text-rec/inference.json",
    "text-rec/inference.pdiparams",
    "text-rec/inference.yml",
    "training-receipt.json",
)
_TRAINER_REVISION = "source-attested-tiny-crnn-bigru-ctc-v1"
_MANIFEST_REVISION = "source-attested-gold-crop-manifest-v1"
_DEFAULT_SEED = 20260826
_MAX_JSON_BYTES = 128 * 1024 * 1024
_DOCUMENT_PREVIEW_ROOT_SUFFIX = ("_local", "data", "document-preview")


class SourceAttestedRecognizerTrainingError(RuntimeError):
    """Stable, content-free failure raised by the offline trainer."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrainingCropV1(_WireModel):
    sample_id: str = Field(pattern=_SHA256)
    asset_id: str = Field(min_length=1, max_length=160)
    family_id: str = Field(min_length=1, max_length=160)
    split: Literal["train", "development"]
    sample_kind: Literal["span", "merged_line"] = "span"
    asset_relative_path: str = Field(min_length=1, max_length=512)
    gold_relative_path: str = Field(min_length=1, max_length=512)
    source_pdf_sha256: str = Field(pattern=_SHA256)
    raster_sha256: str = Field(pattern=_SHA256)
    gold_sha256: str = Field(pattern=_SHA256)
    span_ids: tuple[str, ...] = Field(min_length=1, max_length=128)
    bbox: tuple[float, float, float, float]
    label_sha256: str = Field(pattern=_SHA256)
    label_length: int = Field(ge=1, le=512)
    page_width: int = Field(ge=1, le=20_000)
    page_height: int = Field(ge=1, le=20_000)

    @model_validator(mode="after")
    def _valid_bbox(self) -> "TrainingCropV1":
        x, y, width, height = self.bbox
        if not all(math.isfinite(value) for value in self.bbox):
            raise ValueError("training_crop_bbox_invalid")
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
            raise ValueError("training_crop_bbox_invalid")
        if len(self.span_ids) != len(set(self.span_ids)):
            raise ValueError("training_crop_span_ids_invalid")
        return self


class CropManifestV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_raster_text_crop_manifest"] = (
        "source_attested_raster_text_crop_manifest"
    )
    manifest_revision: Literal["source-attested-gold-crop-manifest-v1"] = (
        "source-attested-gold-crop-manifest-v1"
    )
    dataset_id: Literal["synthetic-raster-v1"] = "synthetic-raster-v1"
    split: Literal["train", "development"]
    stage: Literal["overfit", "pilot", "full"]
    seed_sha256: str = Field(pattern=_SHA256)
    source_receipt_sha256: str = Field(pattern=_SHA256)
    raster_receipt_sha256: str = Field(pattern=_SHA256)
    family_count: int = Field(ge=1)
    crop_count: int = Field(ge=1)
    character_count: int = Field(ge=1, le=512)
    crop_list_sha256: str = Field(pattern=_SHA256)
    include_merged_lines: bool


class TrainingReceiptV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_raster_text_recognizer_training_receipt"] = (
        "source_attested_raster_text_recognizer_training_receipt"
    )
    trainer_revision: Literal["source-attested-tiny-crnn-bigru-ctc-v1"] = (
        "source-attested-tiny-crnn-bigru-ctc-v1"
    )
    stage: Literal["overfit", "pilot", "full"]
    training_intent_sha256: str = Field(pattern=_SHA256)
    train_manifest_sha256: str = Field(pattern=_SHA256)
    development_manifest_sha256: str = Field(pattern=_SHA256)
    character_dictionary_sha256: str = Field(pattern=_SHA256)
    package_inventory_sha256: str = Field(pattern=_SHA256)
    training_runtime_bundle_manifest_sha256: str = Field(pattern=_SHA256)
    parent_python_revision: str = Field(pattern=r"^3\.[0-9]+$")
    training_module_sha256: str = Field(pattern=_SHA256)
    seed_sha256: str = Field(pattern=_SHA256)
    batch_size: int = Field(ge=1, le=128)
    maximum_epochs: int = Field(ge=1, le=30)
    include_merged_lines: Literal[True] = True
    initialization: Literal["deterministic_random_no_pretrained_checkpoint"] = (
        "deterministic_random_no_pretrained_checkpoint"
    )
    framework_license: Literal["Apache-2.0"] = "Apache-2.0"
    model_revision: Literal["tiny-ds-cnn-bigru-ctc-v1"] = "tiny-ds-cnn-bigru-ctc-v1"
    completed_epochs: int = Field(ge=1, le=100)
    best_epoch: int = Field(ge=1, le=100)
    early_stopped: bool
    development_cer: float = Field(ge=0, le=100)
    development_exact_match: float = Field(ge=0, le=1)
    model_artifact_sha256s: dict[str, str]
    receipt_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _receipt_digest_matches(self) -> "TrainingReceiptV1":
        if self.receipt_sha256 != training_receipt_sha256(self):
            raise ValueError("training_receipt_sha256_mismatch")
        if set(self.model_artifact_sha256s) != set(_RELEASE_FILES[:4]):
            raise ValueError("training_receipt_artifacts_invalid")
        return self


class ReleaseFileV1(_WireModel):
    relative_path: str
    byte_size: int = Field(ge=1)
    sha256: str = Field(pattern=_SHA256)


class RecognizerReleaseManifestV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_raster_text_recognizer_release"] = (
        "source_attested_raster_text_recognizer_release"
    )
    receipt_sha256: str = Field(pattern=_SHA256)
    files: tuple[ReleaseFileV1, ...] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def _exact_files(self) -> "RecognizerReleaseManifestV1":
        if tuple(item.relative_path for item in self.files) != _RELEASE_FILES:
            raise ValueError("recognizer_release_files_invalid")
        return self


@dataclass(frozen=True, slots=True)
class VerifiedRecognizerRelease:
    release_sha256: str
    release_path: Path
    manifest: RecognizerReleaseManifestV1
    receipt: TrainingReceiptV1
    file_paths: Mapping[str, Path]
    file_sha256s: Mapping[str, str]


@dataclass(slots=True)
class VerifiedCropCorpus:
    train_manifest: CropManifestV1
    development_manifest: CropManifestV1
    train_crops: tuple[TrainingCropV1, ...]
    development_crops: tuple[TrainingCropV1, ...]
    labels_by_sample_id: dict[str, str]
    characters: tuple[str, ...]


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def canonical_json_file_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def training_receipt_sha256(receipt: TrainingReceiptV1 | Mapping[str, object]) -> str:
    payload = (
        receipt.model_dump(mode="json", warnings=False)
        if isinstance(receipt, TrainingReceiptV1)
        else dict(receipt)
    )
    payload.pop("receipt_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def _verified_document_preview_root(path: Path) -> Path:
    """Resolve only the fixed local data boundary used by raster evaluation."""

    try:
        if path.is_symlink():
            raise OSError
        root = path.resolve(strict=True)
        if (
            root.is_symlink()
            or not root.is_dir()
            or tuple(part.casefold() for part in root.parts[-3:])
            != _DOCUMENT_PREVIEW_ROOT_SUFFIX
        ):
            raise OSError
        return root
    except OSError:
        raise SourceAttestedRecognizerTrainingError("training_document_preview_root_invalid") from None


def _relative_parts(value: object) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SourceAttestedRecognizerTrainingError("training_material_path_invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise SourceAttestedRecognizerTrainingError("training_material_path_invalid")
    return relative.parts


def _safe_relative_path(root: Path, value: object) -> Path:
    parts = _relative_parts(value)
    candidate = root
    try:
        resolved_root = root.resolve(strict=True)
        for part in parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise OSError
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError):
        raise SourceAttestedRecognizerTrainingError("training_material_path_invalid") from None
    if candidate.is_symlink() or not resolved.is_file():
        raise SourceAttestedRecognizerTrainingError("training_material_path_invalid")
    return resolved


def _safe_relative_directory(root: Path, value: object) -> Path:
    parts = _relative_parts(value)
    candidate = root
    try:
        resolved_root = root.resolve(strict=True)
        for part in parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise OSError
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError):
        raise SourceAttestedRecognizerTrainingError("training_material_path_invalid") from None
    if not resolved.is_dir():
        raise SourceAttestedRecognizerTrainingError("training_material_path_invalid")
    return resolved


def _read_json_object(path: Path, *, limit: int = _MAX_JSON_BYTES) -> tuple[dict[str, Any], bytes]:
    try:
        if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= limit:
            raise ValueError
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        raise SourceAttestedRecognizerTrainingError("training_material_json_invalid") from None
    if not isinstance(value, dict):
        raise SourceAttestedRecognizerTrainingError("training_material_json_invalid")
    return value, raw


def _write_fsynced(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def load_verified_source_attested_raster_text_recognizer_release(
    document_preview_root: Path, release_sha256: str
) -> VerifiedRecognizerRelease:
    if re.fullmatch(_SHA256, release_sha256) is None:
        raise SourceAttestedRecognizerTrainingError("recognizer_release_sha256_invalid")
    root = _verified_document_preview_root(document_preview_root)
    release = root / _RELEASE_COLLECTION / "releases" / release_sha256
    if release.is_symlink() or not release.is_dir():
        raise SourceAttestedRecognizerTrainingError("recognizer_release_unavailable")
    declared_tree = {"manifest.json", *_RELEASE_FILES}
    actual_tree: set[str] = set()
    actual_directories: set[str] = set()
    for candidate in release.rglob("*"):
        if candidate.is_symlink():
            raise SourceAttestedRecognizerTrainingError("recognizer_release_tree_invalid")
        if candidate.is_file():
            actual_tree.add(candidate.relative_to(release).as_posix())
        elif candidate.is_dir():
            actual_directories.add(candidate.relative_to(release).as_posix())
        else:
            raise SourceAttestedRecognizerTrainingError("recognizer_release_tree_invalid")
    if actual_tree != declared_tree or actual_directories != {"text-rec"}:
        raise SourceAttestedRecognizerTrainingError("recognizer_release_tree_invalid")
    manifest_raw, manifest_bytes = _read_json_object(release / "manifest.json", limit=1024 * 1024)
    if manifest_bytes != canonical_json_file_bytes(manifest_raw):
        raise SourceAttestedRecognizerTrainingError("recognizer_release_manifest_not_canonical")
    try:
        manifest = RecognizerReleaseManifestV1.model_validate(manifest_raw)
    except ValueError:
        raise SourceAttestedRecognizerTrainingError("recognizer_release_manifest_invalid") from None
    if manifest.receipt_sha256 != release_sha256:
        raise SourceAttestedRecognizerTrainingError("recognizer_release_binding_invalid")
    file_paths: dict[str, Path] = {}
    file_sha256s: dict[str, str] = {}
    for item in manifest.files:
        path = _safe_relative_path(release, item.relative_path)
        if path.stat().st_size != item.byte_size or sha256_file(path) != item.sha256:
            raise SourceAttestedRecognizerTrainingError("recognizer_release_file_drift")
        file_paths[item.relative_path] = path
        file_sha256s[item.relative_path] = item.sha256
    receipt_raw, receipt_bytes = _read_json_object(file_paths["training-receipt.json"])
    if receipt_bytes != canonical_json_file_bytes(receipt_raw):
        raise SourceAttestedRecognizerTrainingError("training_receipt_not_canonical")
    try:
        receipt = TrainingReceiptV1.model_validate(receipt_raw)
    except ValueError:
        raise SourceAttestedRecognizerTrainingError("training_receipt_invalid") from None
    if receipt.receipt_sha256 != release_sha256:
        raise SourceAttestedRecognizerTrainingError("recognizer_release_binding_invalid")
    for relative_path in _RELEASE_FILES[:4]:
        if receipt.model_artifact_sha256s[relative_path] != file_sha256s[relative_path]:
            raise SourceAttestedRecognizerTrainingError("recognizer_release_binding_invalid")
    return VerifiedRecognizerRelease(
        release_sha256=release_sha256,
        release_path=release,
        manifest=manifest,
        receipt=receipt,
        file_paths=file_paths,
        file_sha256s=file_sha256s,
    )


def _normalized_label(value: object) -> str:
    if not isinstance(value, str):
        raise SourceAttestedRecognizerTrainingError("training_gold_label_invalid")
    label = " ".join(unicodedata.normalize("NFC", value).split())
    if not label or len(label) > 512 or any(ord(character) < 32 for character in label):
        raise SourceAttestedRecognizerTrainingError("training_gold_label_invalid")
    return label


def _strict_sha(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(_SHA256, value) is None:
        raise SourceAttestedRecognizerTrainingError("training_material_sha256_invalid")
    return value


def _strict_identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 160
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", value) is None
    ):
        raise SourceAttestedRecognizerTrainingError("training_material_identifier_invalid")
    return value


def _strict_integer(value: object, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise SourceAttestedRecognizerTrainingError("training_material_integer_invalid")
    return value


def _bbox(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, dict) or set(value) != {"x", "y", "width", "height"}:
        raise SourceAttestedRecognizerTrainingError("training_gold_bbox_invalid")
    try:
        result = tuple(float(value[key]) for key in ("x", "y", "width", "height"))
    except (TypeError, ValueError):
        raise SourceAttestedRecognizerTrainingError("training_gold_bbox_invalid") from None
    x, y, width, height = result
    if (
        not all(math.isfinite(item) for item in result)
        or x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > 1.0000001
        or y + height > 1.0000001
    ):
        raise SourceAttestedRecognizerTrainingError("training_gold_bbox_invalid")
    return result  # type: ignore[return-value]


def _sample_id(
    *, asset_id: str, kind: str, span_ids: Sequence[str], bbox: Sequence[float]
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "asset_id": asset_id,
                "bbox": [round(value, 12) for value in bbox],
                "kind": kind,
                "span_ids": list(span_ids),
            }
        )
    )


def _crop_payload(crop: TrainingCropV1) -> dict[str, object]:
    return crop.model_dump(mode="json", warnings=False)


def _crop_list_sha256(crops: Sequence[TrainingCropV1]) -> str:
    digest = hashlib.sha256()
    digest.update((_MANIFEST_REVISION + "\n").encode("ascii"))
    for crop in crops:
        digest.update(canonical_json_bytes(_crop_payload(crop)))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _GoldSpan:
    span_id: str
    reading_order: int
    bbox: tuple[float, float, float, float]
    label: str


def _vertical_overlap(first: _GoldSpan, second: _GoldSpan) -> float:
    first_bottom = first.bbox[1] + first.bbox[3]
    second_bottom = second.bbox[1] + second.bbox[3]
    overlap = max(0.0, min(first_bottom, second_bottom) - max(first.bbox[1], second.bbox[1]))
    return overlap / min(first.bbox[3], second.bbox[3])


def _merged_line_groups(spans: Sequence[_GoldSpan]) -> tuple[tuple[_GoldSpan, ...], ...]:
    """Deterministically approximate detector line boxes from source geometry."""

    rows: list[list[_GoldSpan]] = []
    ordered = sorted(
        spans,
        key=lambda item: (
            round(item.bbox[1] + item.bbox[3] / 2, 12),
            round(item.bbox[0], 12),
            item.reading_order,
            item.span_id,
        ),
    )
    for span in ordered:
        chosen: list[_GoldSpan] | None = None
        for row in rows:
            anchor = row[0]
            centers = abs(
                (span.bbox[1] + span.bbox[3] / 2)
                - (anchor.bbox[1] + anchor.bbox[3] / 2)
            )
            if _vertical_overlap(span, anchor) >= 0.45 and centers <= max(
                span.bbox[3], anchor.bbox[3]
            ) * 0.65:
                chosen = row
                break
        if chosen is None:
            rows.append([span])
        else:
            chosen.append(span)
    groups: list[tuple[_GoldSpan, ...]] = []
    for row in rows:
        current: list[_GoldSpan] = []
        for span in sorted(row, key=lambda item: (item.bbox[0], item.reading_order, item.span_id)):
            if current:
                previous = current[-1]
                gap = span.bbox[0] - (previous.bbox[0] + previous.bbox[2])
                if gap > 2.75 * max(span.bbox[3], previous.bbox[3]):
                    if len(current) > 1:
                        groups.append(tuple(current))
                    current = []
            current.append(span)
        if len(current) > 1:
            groups.append(tuple(current))
    return tuple(groups)


def _union_bbox(spans: Sequence[_GoldSpan]) -> tuple[float, float, float, float]:
    left = min(span.bbox[0] for span in spans)
    top = min(span.bbox[1] for span in spans)
    right = max(span.bbox[0] + span.bbox[2] for span in spans)
    bottom = max(span.bbox[1] + span.bbox[3] for span in spans)
    return left, top, right - left, bottom - top


def _build_asset_crops(
    *, asset: Mapping[str, object], gold: Mapping[str, object], include_merged_lines: bool
) -> tuple[list[TrainingCropV1], dict[str, str]]:
    required_equal = {
        "asset_id": asset["asset_id"],
        "family_id": asset["family_id"],
        "profile_id": "png-216",
        "source_pdf_sha256": asset["source_pdf_sha256"],
    }
    if any(gold.get(key) != expected for key, expected in required_equal.items()):
        raise SourceAttestedRecognizerTrainingError("training_gold_binding_invalid")
    raw_spans = gold.get("spans")
    if not isinstance(raw_spans, list) or not raw_spans:
        raise SourceAttestedRecognizerTrainingError("training_gold_spans_invalid")
    spans: list[_GoldSpan] = []
    seen: set[str] = set()
    for value in raw_spans:
        if not isinstance(value, dict):
            raise SourceAttestedRecognizerTrainingError("training_gold_spans_invalid")
        span_id = _strict_identifier(value.get("span_id"))
        if span_id in seen:
            raise SourceAttestedRecognizerTrainingError("training_gold_spans_invalid")
        seen.add(span_id)
        reading_order = value.get("reading_order")
        if isinstance(reading_order, bool) or not isinstance(reading_order, int) or reading_order < 0:
            raise SourceAttestedRecognizerTrainingError("training_gold_spans_invalid")
        spans.append(
            _GoldSpan(
                span_id=span_id,
                reading_order=reading_order,
                bbox=_bbox(value.get("bbox")),
                label=_normalized_label(value.get("text")),
            )
        )
    asset_id = _strict_identifier(asset.get("asset_id"))
    family_id = _strict_identifier(asset.get("family_id"))
    split_value = asset.get("split")
    if split_value not in {"train", "development"}:
        raise SourceAttestedRecognizerTrainingError("training_dataset_split_invalid")
    split: Literal["train", "development"] = split_value
    asset_relative_path = str(asset.get("asset_relative_path"))
    gold_relative_path = str(asset.get("gold_relative_path"))
    source_pdf_sha256 = _strict_sha(asset.get("source_pdf_sha256"))
    raster_sha256 = _strict_sha(asset.get("image_sha256"))
    gold_sha256 = _strict_sha(asset.get("gold_sha256"))
    page_width = _strict_integer(asset.get("width"), minimum=1, maximum=20_000)
    page_height = _strict_integer(asset.get("height"), minimum=1, maximum=20_000)
    crops: list[TrainingCropV1] = []
    labels: dict[str, str] = {}
    candidates: list[
        tuple[
            Literal["span", "merged_line"],
            tuple[_GoldSpan, ...],
            tuple[float, float, float, float],
            str,
        ]
    ] = [
        ("span", (span,), span.bbox, span.label) for span in spans
    ]
    if include_merged_lines:
        for group in _merged_line_groups(spans):
            candidates.append(
                ("merged_line", group, _union_bbox(group), " ".join(span.label for span in group))
            )
    for kind, members, bbox, label in candidates:
        span_ids = tuple(member.span_id for member in members)
        sample_id = _sample_id(
            asset_id=asset_id, kind=kind, span_ids=span_ids, bbox=bbox
        )
        crop = TrainingCropV1(
            sample_id=sample_id,
            asset_id=asset_id,
            family_id=family_id,
            split=split,
            sample_kind=kind,
            asset_relative_path=asset_relative_path,
            gold_relative_path=gold_relative_path,
            source_pdf_sha256=source_pdf_sha256,
            raster_sha256=raster_sha256,
            gold_sha256=gold_sha256,
            span_ids=span_ids,
            bbox=bbox,
            label_sha256=sha256_bytes(label.encode("utf-8")),
            label_length=len(label),
            page_width=page_width,
            page_height=page_height,
        )
        crops.append(crop)
        labels[sample_id] = label
    crops.sort(key=lambda item: item.sample_id)
    return crops, labels


def _stage_select(
    crops: Sequence[TrainingCropV1], *, stage: Literal["overfit", "pilot", "full"], seed: int
) -> tuple[TrainingCropV1, ...]:
    if stage == "full":
        return tuple(sorted(crops, key=lambda item: item.sample_id))
    families = sorted({crop.family_id for crop in crops})
    if stage == "pilot":
        count = max(1, math.ceil(len(families) * 0.10))
        chosen = set(
            sorted(
                families,
                key=lambda value: sha256_bytes(f"{seed}:pilot:{value}".encode("utf-8")),
            )[:count]
        )
        return tuple(sorted((crop for crop in crops if crop.family_id in chosen), key=lambda x: x.sample_id))
    return tuple(
        sorted(
            crops,
            key=lambda value: sha256_bytes(f"{seed}:overfit:{value.sample_id}".encode("utf-8")),
        )[:1000]
    )


def _manifest(
    *,
    crops: Sequence[TrainingCropV1],
    split: Literal["train", "development"],
    stage: Literal["overfit", "pilot", "full"],
    seed: int,
    source_receipt_sha256: str,
    raster_receipt_sha256: str,
    character_count: int,
    include_merged_lines: bool,
) -> CropManifestV1:
    return CropManifestV1(
        split=split,
        stage=stage,
        seed_sha256=sha256_bytes(str(seed).encode("ascii")),
        source_receipt_sha256=source_receipt_sha256,
        raster_receipt_sha256=raster_receipt_sha256,
        family_count=len({crop.family_id for crop in crops}),
        crop_count=len(crops),
        character_count=character_count,
        crop_list_sha256=_crop_list_sha256(crops),
        include_merged_lines=include_merged_lines,
    )


def build_verified_gold_crop_corpus(
    document_preview_root: Path,
    *,
    stage: Literal["overfit", "pilot", "full"] = "overfit",
    seed: int = _DEFAULT_SEED,
    include_merged_lines: bool = True,
) -> VerifiedCropCorpus:
    """Verify source/raster/Gold and select only PNG train/development crops."""

    root = _verified_document_preview_root(document_preview_root)
    dataset = root / "synthetic-raster-v1"
    raster_receipt, raster_raw = _read_json_object(dataset / "synthetic-raster-receipt.json")
    source_receipt, source_raw = _read_json_object(dataset / "source-receipt.json")
    source_receipt_sha = sha256_bytes(source_raw)
    raster_receipt_sha = sha256_bytes(raster_raw)
    if raster_receipt.get("dataset_id") not in {None, "synthetic-raster-v1"}:
        raise SourceAttestedRecognizerTrainingError("training_dataset_invalid")
    if raster_receipt.get("source_receipt_sha256") != source_receipt_sha:
        raise SourceAttestedRecognizerTrainingError("training_source_receipt_drift")
    source_root_relative = source_receipt.get("source_root_relative_path")
    if not isinstance(source_root_relative, str):
        raise SourceAttestedRecognizerTrainingError("training_source_receipt_invalid")
    try:
        source_root = _safe_relative_directory(root, source_root_relative)
    except SourceAttestedRecognizerTrainingError:
        raise SourceAttestedRecognizerTrainingError("training_source_receipt_invalid") from None
    raw_sources = source_receipt.get("assets")
    raw_assets = raster_receipt.get("assets")
    if not isinstance(raw_sources, list) or not isinstance(raw_assets, list):
        raise SourceAttestedRecognizerTrainingError("training_dataset_invalid")
    sources: dict[str, Mapping[str, object]] = {}
    for value in raw_sources:
        if not isinstance(value, dict):
            raise SourceAttestedRecognizerTrainingError("training_source_receipt_invalid")
        digest = _strict_sha(value.get("source_pdf_sha256"))
        if digest in sources:
            raise SourceAttestedRecognizerTrainingError("training_source_receipt_invalid")
        sources[digest] = value
    family_splits: dict[str, set[str]] = {}
    selected_assets: list[Mapping[str, object]] = []
    for value in raw_assets:
        if not isinstance(value, dict):
            raise SourceAttestedRecognizerTrainingError("training_dataset_invalid")
        family = _strict_identifier(value.get("family_id"))
        split = value.get("split")
        if split not in {"train", "development", "calibration", "sealed"}:
            raise SourceAttestedRecognizerTrainingError("training_dataset_split_invalid")
        family_splits.setdefault(family, set()).add(str(split))
        if value.get("profile_id") == "png-216" and split in {"train", "development"}:
            if value.get("media_type") != "image/png":
                raise SourceAttestedRecognizerTrainingError("training_raster_profile_invalid")
            selected_assets.append(value)
    if any(len(splits) != 1 for splits in family_splits.values()):
        raise SourceAttestedRecognizerTrainingError("training_family_leakage")
    all_crops: dict[str, list[TrainingCropV1]] = {"train": [], "development": []}
    labels: dict[str, str] = {}
    for asset in sorted(selected_assets, key=lambda item: str(item.get("asset_id"))):
        source_sha = _strict_sha(asset.get("source_pdf_sha256"))
        source = sources.get(source_sha)
        if source is None or source.get("family_id") != asset.get("family_id"):
            raise SourceAttestedRecognizerTrainingError("training_source_binding_invalid")
        source_path = _safe_relative_path(source_root, source.get("source_relative_path"))
        raster_path = _safe_relative_path(root, asset.get("asset_relative_path"))
        gold_path = _safe_relative_path(root, asset.get("gold_relative_path"))
        if (
            source_path.stat().st_size
            != _strict_integer(source.get("byte_size"), minimum=1, maximum=2**63 - 1)
            or sha256_file(source_path) != source_sha
            or raster_path.stat().st_size
            != _strict_integer(asset.get("byte_size"), minimum=1, maximum=2**63 - 1)
            or sha256_file(raster_path) != _strict_sha(asset.get("image_sha256"))
            or sha256_file(gold_path) != _strict_sha(asset.get("gold_sha256"))
        ):
            raise SourceAttestedRecognizerTrainingError("training_material_drift")
        gold, _gold_raw = _read_json_object(gold_path, limit=8 * 1024 * 1024)
        crops, asset_labels = _build_asset_crops(
            asset=asset, gold=gold, include_merged_lines=include_merged_lines
        )
        split = str(asset["split"])
        all_crops[split].extend(crops)
        if set(labels).intersection(asset_labels):
            raise SourceAttestedRecognizerTrainingError("training_sample_collision")
        labels.update(asset_labels)
    train_families = {crop.family_id for crop in all_crops["train"]}
    development_families = {crop.family_id for crop in all_crops["development"]}
    if not train_families or not development_families or train_families & development_families:
        raise SourceAttestedRecognizerTrainingError("training_family_leakage")
    characters = tuple(sorted({character for crop in all_crops["train"] for character in labels[crop.sample_id]}))
    if not characters or len(characters) > 511:
        raise SourceAttestedRecognizerTrainingError("training_character_dictionary_invalid")
    if any(character not in characters for crop in all_crops["development"] for character in labels[crop.sample_id]):
        raise SourceAttestedRecognizerTrainingError("training_development_character_unseen")
    train = _stage_select(all_crops["train"], stage=stage, seed=seed)
    development = _stage_select(all_crops["development"], stage=stage, seed=seed + 1)
    selected_ids = {crop.sample_id for crop in (*train, *development)}
    selected_labels = {sample_id: label for sample_id, label in labels.items() if sample_id in selected_ids}
    return VerifiedCropCorpus(
        train_manifest=_manifest(
            crops=train,
            split="train",
            stage=stage,
            seed=seed,
            source_receipt_sha256=source_receipt_sha,
            raster_receipt_sha256=raster_receipt_sha,
            character_count=len(characters),
            include_merged_lines=include_merged_lines,
        ),
        development_manifest=_manifest(
            crops=development,
            split="development",
            stage=stage,
            seed=seed,
            source_receipt_sha256=source_receipt_sha,
            raster_receipt_sha256=raster_receipt_sha,
            character_count=len(characters),
            include_merged_lines=include_merged_lines,
        ),
        train_crops=train,
        development_crops=development,
        labels_by_sample_id=selected_labels,
        characters=characters,
    )


_T = TypeVar("_T")


class PageLruCache(Generic[_T]):
    """Small deterministic LRU; page pixels never enter receipts."""

    def __init__(self, capacity: int = 8) -> None:
        if not 1 <= capacity <= 128:
            raise ValueError("page_lru_capacity_invalid")
        self.capacity = capacity
        self._values: OrderedDict[object, _T] = OrderedDict()

    def get(self, key: object, loader: Callable[[], _T]) -> _T:
        if key in self._values:
            value = self._values.pop(key)
            self._values[key] = value
            return value
        value = loader()
        self._values[key] = value
        while len(self._values) > self.capacity:
            self._values.popitem(last=False)
        return value

    @property
    def keys(self) -> tuple[object, ...]:
        return tuple(self._values)


def estimated_recognition_width(crop: TrainingCropV1) -> int:
    _x, _y, width, height = crop.bbox
    pixel_width = width * crop.page_width
    pixel_height = height * crop.page_height
    return min(3200, max(8, int(math.ceil(48.0 * pixel_width / pixel_height))))


def deterministic_width_batches(
    crops: Sequence[TrainingCropV1], *, epoch: int, seed: int, batch_size: int
) -> tuple[tuple[TrainingCropV1, ...], ...]:
    if epoch < 0 or not 1 <= batch_size <= 256:
        raise ValueError("training_batch_configuration_invalid")
    boundaries = (128, 256, 384, 512, 768, 1024, 1600, 3200)
    buckets: dict[int, list[TrainingCropV1]] = {value: [] for value in boundaries}
    for crop in crops:
        width = estimated_recognition_width(crop)
        boundary = next(value for value in boundaries if width <= value)
        buckets[boundary].append(crop)
    batches: list[tuple[TrainingCropV1, ...]] = []
    for boundary in boundaries:
        values = sorted(buckets[boundary], key=lambda item: item.sample_id)
        random.Random(f"{seed}:{epoch}:{boundary}").shuffle(values)
        batches.extend(tuple(values[index : index + batch_size]) for index in range(0, len(values), batch_size))
    random.Random(f"{seed}:{epoch}:batch-order").shuffle(batches)
    return tuple(batches)


def build_tiny_paddle_crnn_bigru_ctc(paddle: Any, *, character_count: int) -> Any:
    """Create a random-init, sub-million parameter dynamic-width recognizer."""

    if not 1 <= character_count <= 511:
        raise SourceAttestedRecognizerTrainingError("training_character_dictionary_invalid")
    nn = paddle.nn

    class _DepthwiseSeparable(nn.Layer):  # type: ignore[misc,valid-type,name-defined]
        def __init__(self, input_channels: int, output_channels: int, stride: tuple[int, int]):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2D(
                    input_channels,
                    input_channels,
                    3,
                    stride=stride,
                    padding=1,
                    groups=input_channels,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(input_channels),
                nn.ReLU(),
                nn.Conv2D(input_channels, output_channels, 1, bias_attr=False),
                nn.BatchNorm2D(output_channels),
                nn.ReLU(),
            )

        def forward(self, value: Any) -> Any:
            return self.layers(value)

    class TinyCrnnBiGruCtc(nn.Layer):  # type: ignore[misc,valid-type,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2D(3, 32, 3, stride=(2, 2), padding=1, bias_attr=False),
                nn.BatchNorm2D(32),
                nn.ReLU(),
                _DepthwiseSeparable(32, 64, (2, 2)),
                _DepthwiseSeparable(64, 128, (2, 2)),
                _DepthwiseSeparable(128, 192, (2, 1)),
                nn.Conv2D(192, 192, (3, 1), bias_attr=False),
                nn.BatchNorm2D(192),
                nn.ReLU(),
            )
            self.sequence = nn.GRU(
                input_size=192,
                hidden_size=128,
                num_layers=2,
                direction="bidirectional",
            )
            self.classifier = nn.Linear(256, character_count + 1)

        def forward(self, value: Any) -> Any:
            features = self.features(value)
            features = paddle.squeeze(features, axis=2)
            features = paddle.transpose(features, perm=[0, 2, 1])
            sequence, _state = self.sequence(features)
            return self.classifier(sequence)

    return TinyCrnnBiGruCtc()


def ctc_train_batch_step(
    paddle: Any,
    *,
    model: Any,
    optimizer: Any,
    images: Any,
    labels: Any,
    input_lengths: Any,
    label_lengths: Any,
) -> float:
    """Run one bounded CTC update without exposing labels or predictions."""

    model.train()
    logits = model(images)
    log_probabilities = paddle.nn.functional.log_softmax(logits, axis=-1)
    log_probabilities = paddle.transpose(log_probabilities, perm=[1, 0, 2])
    loss = paddle.nn.functional.ctc_loss(
        log_probabilities,
        labels,
        input_lengths,
        label_lengths,
        blank=0,
        reduction="mean",
    )
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
    numeric = float(loss.numpy().reshape([-1])[0])
    if not math.isfinite(numeric):
        raise SourceAttestedRecognizerTrainingError("training_loss_invalid")
    return numeric


def _native_export_prefix(text_rec_directory: Path) -> Path:
    """Return an ASCII Win32 spelling for Paddle's native serializer.

    Paddle 3's PIR serializer still passes its destination through a narrow
    native filesystem boundary on Windows.  The verified workspace can have
    non-ASCII path components, so keep the canonical path for all attestation
    and use the equivalent existing 8.3 directory only for the native save.
    """

    canonical = text_rec_directory.resolve(strict=True)
    if os.name != "nt":
        return canonical / "inference"
    try:
        import ctypes

        value = str(canonical)
        required = ctypes.windll.kernel32.GetShortPathNameW(value, None, 0)
        if not isinstance(required, int) or required < 1:
            raise OSError
        buffer = ctypes.create_unicode_buffer(required + 1)
        written = ctypes.windll.kernel32.GetShortPathNameW(value, buffer, len(buffer))
        if not isinstance(written, int) or not 0 < written < len(buffer):
            raise OSError
        short = buffer.value
        if not short or not short.isascii():
            raise OSError
        return Path(short) / "inference"
    except Exception:
        raise SourceAttestedRecognizerTrainingError(
            "training_export_path_unsupported"
        ) from None


def export_tiny_paddle_static_recognizer(
    paddle: Any, *, model: Any, text_rec_directory: Path, characters: Sequence[str]
) -> dict[str, str]:
    """Export exactly the four files consumed by the existing static sidecar."""

    if text_rec_directory.exists() or text_rec_directory.is_symlink():
        raise SourceAttestedRecognizerTrainingError("training_export_destination_invalid")
    text_rec_directory.mkdir(parents=True)
    model.eval()
    prefix = _native_export_prefix(text_rec_directory)
    try:
        static_model = paddle.jit.to_static(
            model,
            input_spec=[paddle.static.InputSpec(shape=[None, 3, 48, None], dtype="float32")],
            full_graph=True,
        )
        paddle.jit.save(static_model, str(prefix))
    except Exception:
        raise SourceAttestedRecognizerTrainingError("training_static_export_failed") from None
    inference_json = text_rec_directory / "inference.json"
    inference_parameters = text_rec_directory / "inference.pdiparams"
    if (
        inference_json.is_symlink()
        or inference_parameters.is_symlink()
        or not inference_json.is_file()
        or not inference_parameters.is_file()
        or inference_json.stat().st_size < 1
        or inference_parameters.stat().st_size < 1
    ):
        raise SourceAttestedRecognizerTrainingError("training_static_export_invalid")
    config = {
        "Global": {"model_name": "source_attested_tiny_crnn_bigru_ctc_v1"},
        "PostProcess": {"character_dict": list(characters), "name": "CTCLabelDecode"},
        "PreProcess": {"input_shape": [3, 48, -1], "normalization": "rgb-minus-half-v1"},
    }
    _write_fsynced(text_rec_directory / "config.json", canonical_json_file_bytes(config))
    inference_yml = (
        "Global:\n"
        "  model_name: source_attested_tiny_crnn_bigru_ctc_v1\n"
        "  model_type: rec\n"
        "PostProcess:\n"
        "  name: CTCLabelDecode\n"
    ).encode("utf-8")
    _write_fsynced(text_rec_directory / "inference.yml", inference_yml)
    return {
        relative: sha256_file(text_rec_directory.parent / relative)
        for relative in _RELEASE_FILES[:4]
    }


def publish_source_attested_raster_text_recognizer_release(
    document_preview_root: Path,
    *,
    staged_release: Path,
    receipt_payload: Mapping[str, object],
) -> VerifiedRecognizerRelease:
    """Publish a completed stage by one same-volume directory rename."""

    root = _verified_document_preview_root(document_preview_root)
    collection = root / _RELEASE_COLLECTION
    releases = collection / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    try:
        staged_resolved = staged_release.resolve(strict=True)
        staged_resolved.relative_to(collection.resolve(strict=True))
    except (OSError, ValueError):
        raise SourceAttestedRecognizerTrainingError("training_stage_invalid") from None
    if (
        staged_release.is_symlink()
        or not staged_release.is_dir()
        or not staged_release.name.startswith(".staging-")
    ):
        raise SourceAttestedRecognizerTrainingError("training_stage_invalid")
    staged_files: set[str] = set()
    staged_directories: set[str] = set()
    for candidate in staged_release.rglob("*"):
        if candidate.is_symlink():
            raise SourceAttestedRecognizerTrainingError("training_stage_invalid")
        if candidate.is_file():
            staged_files.add(candidate.relative_to(staged_release).as_posix())
        elif candidate.is_dir():
            staged_directories.add(candidate.relative_to(staged_release).as_posix())
        else:
            raise SourceAttestedRecognizerTrainingError("training_stage_invalid")
    if staged_files != set(_RELEASE_FILES[:4]) or staged_directories != {"text-rec"}:
        raise SourceAttestedRecognizerTrainingError("training_stage_invalid")
    artifact_hashes: dict[str, str] = {}
    for relative in _RELEASE_FILES[:4]:
        path = _safe_relative_path(staged_release, relative)
        artifact_hashes[relative] = sha256_file(path)
    values = dict(receipt_payload)
    values["model_artifact_sha256s"] = artifact_hashes
    values.pop("receipt_sha256", None)
    receipt_sha = training_receipt_sha256(values)
    values["receipt_sha256"] = receipt_sha
    try:
        receipt = TrainingReceiptV1.model_validate(values)
    except ValueError:
        raise SourceAttestedRecognizerTrainingError("training_receipt_invalid") from None
    _write_fsynced(
        staged_release / "training-receipt.json",
        canonical_json_file_bytes(receipt.model_dump(mode="json", warnings=False)),
    )
    files = tuple(
        ReleaseFileV1(
            relative_path=relative,
            byte_size=staged_release.joinpath(*PurePosixPath(relative).parts).stat().st_size,
            sha256=sha256_file(staged_release.joinpath(*PurePosixPath(relative).parts)),
        )
        for relative in _RELEASE_FILES
    )
    manifest = RecognizerReleaseManifestV1(receipt_sha256=receipt_sha, files=files)
    _write_fsynced(
        staged_release / "manifest.json",
        canonical_json_file_bytes(manifest.model_dump(mode="json", warnings=False)),
    )
    final = releases / receipt_sha
    if final.exists() or final.is_symlink():
        existing = load_verified_source_attested_raster_text_recognizer_release(root, receipt_sha)
        expected = {item.relative_path: item.sha256 for item in files}
        if existing.file_sha256s != expected:
            raise SourceAttestedRecognizerTrainingError("recognizer_release_collision")
        shutil.rmtree(staged_release)
        return existing
    try:
        os.rename(staged_release, final)
    except OSError:
        raise SourceAttestedRecognizerTrainingError("recognizer_release_publish_failed") from None
    return load_verified_source_attested_raster_text_recognizer_release(root, receipt_sha)


def create_recognizer_training_stage(document_preview_root: Path) -> Path:
    root = _verified_document_preview_root(document_preview_root)
    collection = root / _RELEASE_COLLECTION
    collection.mkdir(parents=True, exist_ok=True)
    stage = collection / f".staging-{uuid.uuid4().hex}"
    stage.mkdir(mode=0o700)
    return stage


def find_verified_recognizer_release_by_intent(
    document_preview_root: Path, training_intent_sha256: str
) -> VerifiedRecognizerRelease | None:
    if re.fullmatch(_SHA256, training_intent_sha256) is None:
        raise SourceAttestedRecognizerTrainingError("training_intent_invalid")
    root = _verified_document_preview_root(document_preview_root)
    releases = root / _RELEASE_COLLECTION / "releases"
    if not releases.exists():
        return None
    if releases.is_symlink() or not releases.is_dir():
        raise SourceAttestedRecognizerTrainingError("recognizer_release_collection_invalid")
    for candidate in sorted(releases.iterdir(), key=lambda path: path.name):
        if re.fullmatch(_SHA256, candidate.name) is None:
            raise SourceAttestedRecognizerTrainingError("recognizer_release_collection_invalid")
        release = load_verified_source_attested_raster_text_recognizer_release(
            root, candidate.name
        )
        if release.receipt.training_intent_sha256 == training_intent_sha256:
            return release
    return None


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    release: VerifiedRecognizerRelease
    reused: bool


def verify_paddle_training_package_inventory(path: Path) -> str:
    inventory, raw = _read_json_object(path, limit=4 * 1024 * 1024)
    packages = inventory.get("packages")
    if not isinstance(packages, list):
        raise SourceAttestedRecognizerTrainingError("training_package_inventory_invalid")
    expected: dict[str, str] = {}
    for item in packages:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(
            item.get("version"), str
        ):
            raise SourceAttestedRecognizerTrainingError("training_package_inventory_invalid")
        expected[item["name"].casefold()] = item["version"]
    required = {"numpy", "paddlepaddle", "pillow"}
    if not required.issubset(expected):
        raise SourceAttestedRecognizerTrainingError("training_package_inventory_invalid")
    try:
        from importlib import metadata

        for name in required:
            if metadata.version(name) != expected[name]:
                raise SourceAttestedRecognizerTrainingError("training_package_drift")
    except metadata.PackageNotFoundError:
        raise SourceAttestedRecognizerTrainingError("training_package_drift") from None
    return sha256_bytes(raw)


def _load_verified_png_page(root: Path, crop: TrainingCropV1) -> Any:
    try:
        from PIL import Image

        path = _safe_relative_path(root, crop.asset_relative_path)
        if sha256_file(path) != crop.raster_sha256:
            raise SourceAttestedRecognizerTrainingError("training_raster_drift")
        with Image.open(path) as opened:
            if (
                opened.format != "PNG"
                or opened.size != (crop.page_width, crop.page_height)
                or getattr(opened, "n_frames", 1) != 1
            ):
                raise SourceAttestedRecognizerTrainingError("training_raster_decode_invalid")
            opened.load()
            return opened.convert("RGB")
    except SourceAttestedRecognizerTrainingError:
        raise
    except Exception:
        raise SourceAttestedRecognizerTrainingError("training_raster_decode_invalid") from None


def _prepare_paddle_batch(
    paddle: Any,
    *,
    document_preview_root: Path,
    crops: Sequence[TrainingCropV1],
    labels_by_sample_id: Mapping[str, str],
    character_to_index: Mapping[str, int],
    page_cache: PageLruCache[Any],
) -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        raise SourceAttestedRecognizerTrainingError("training_runtime_unavailable") from None
    prepared: list[Any] = []
    widths: list[int] = []
    encoded: list[list[int]] = []
    for crop in crops:
        def load_page(current_crop: TrainingCropV1 = crop) -> Any:
            return _load_verified_png_page(document_preview_root, current_crop)

        page = page_cache.get(
            (crop.asset_id, crop.raster_sha256),
            load_page,
        )
        x, y, width, height = crop.bbox
        pad_x = width * 0.08
        pad_y = height * 0.15
        left = max(0, math.floor((x - pad_x) * crop.page_width))
        top = max(0, math.floor((y - pad_y) * crop.page_height))
        right = min(crop.page_width, math.ceil((x + width + pad_x) * crop.page_width))
        bottom = min(crop.page_height, math.ceil((y + height + pad_y) * crop.page_height))
        if right - left < 2 or bottom - top < 2:
            raise SourceAttestedRecognizerTrainingError("training_crop_invalid")
        image = page.crop((left, top, right, bottom))
        resized_width = min(
            3200, max(8, int(math.ceil(48.0 * image.width / max(1, image.height))))
        )
        image = image.resize((resized_width, 48), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype="float32") / np.float32(255.0)
        # The deployed direct sidecar passes perspective crops to recognition
        # in BGR channel order; preserve that exact inference domain here.
        array = array[:, :, ::-1]
        array = (array - np.float32(0.5)) / np.float32(0.5)
        prepared.append(np.ascontiguousarray(array.transpose((2, 0, 1))))
        widths.append(resized_width)
        label = labels_by_sample_id.get(crop.sample_id)
        if label is None or sha256_bytes(label.encode("utf-8")) != crop.label_sha256:
            raise SourceAttestedRecognizerTrainingError("training_label_binding_invalid")
        try:
            encoded.append([character_to_index[character] for character in label])
        except KeyError:
            raise SourceAttestedRecognizerTrainingError("training_label_dictionary_invalid") from None
    maximum_width = max(widths)
    # Match the existing sidecar's bounded dynamic-width batch contract:
    # unused normalized columns are exactly zero, and width buckets minimize
    # the amount of padding seen by BatchNorm/GRU.
    images = np.zeros((len(crops), 3, 48, maximum_width), dtype="float32")
    for index, array in enumerate(prepared):
        images[index, :, :, : array.shape[2]] = array
    maximum_label = max(len(value) for value in encoded)
    label_array = np.zeros((len(crops), maximum_label), dtype="int32")
    for index, value in enumerate(encoded):
        label_array[index, : len(value)] = value
    # The deployed static sidecar receives no per-item width tensor and
    # decodes the complete padded batch output.  Train/evaluate that same
    # contract so the zero tail is learned as CTC blank instead of becoming
    # an unscored region that could emit suffix characters at runtime.
    output_width = (maximum_width + 7) // 8
    input_lengths = np.full((len(crops),), output_width, dtype="int64")
    label_lengths = np.asarray([len(value) for value in encoded], dtype="int64")
    return (
        paddle.to_tensor(images),
        paddle.to_tensor(label_array),
        paddle.to_tensor(input_lengths),
        paddle.to_tensor(label_lengths),
    )


def _edit_distance(first: Sequence[int], second: Sequence[int]) -> int:
    previous = list(range(len(second) + 1))
    for first_index, first_value in enumerate(first, start=1):
        current = [first_index]
        for second_index, second_value in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[second_index] + 1,
                    previous[second_index - 1] + (first_value != second_value),
                )
            )
        previous = current
    return previous[-1]


def _ctc_decode_indices(values: Sequence[int]) -> list[int]:
    result: list[int] = []
    previous = -1
    for value in values:
        if value != 0 and value != previous:
            result.append(value)
        previous = value
    return result


def _evaluate_crop_model(
    paddle: Any,
    *,
    model: Any,
    document_preview_root: Path,
    crops: Sequence[TrainingCropV1],
    labels_by_sample_id: Mapping[str, str],
    character_to_index: Mapping[str, int],
    seed: int,
    batch_size: int,
) -> tuple[float, float]:
    page_cache: PageLruCache[Any] = PageLruCache(16)
    edits = 0
    characters = 0
    exact = 0
    total = 0
    model.eval()
    with paddle.no_grad():
        for batch in deterministic_width_batches(crops, epoch=0, seed=seed, batch_size=batch_size):
            images, labels, input_lengths, label_lengths = _prepare_paddle_batch(
                paddle,
                document_preview_root=document_preview_root,
                crops=batch,
                labels_by_sample_id=labels_by_sample_id,
                character_to_index=character_to_index,
                page_cache=page_cache,
            )
            logits = model(images)
            predictions = paddle.argmax(logits, axis=-1).numpy().tolist()
            label_values = labels.numpy().tolist()
            input_values = input_lengths.numpy().tolist()
            label_length_values = label_lengths.numpy().tolist()
            for index in range(len(batch)):
                predicted = _ctc_decode_indices(predictions[index][: input_values[index]])
                expected = label_values[index][: label_length_values[index]]
                distance = _edit_distance(expected, predicted)
                edits += distance
                characters += len(expected)
                exact += int(distance == 0)
                total += 1
    if characters < 1 or total < 1:
        raise SourceAttestedRecognizerTrainingError("training_development_empty")
    return edits / characters, exact / total


def _manifest_sha256(manifest: CropManifestV1) -> str:
    return sha256_bytes(canonical_json_bytes(manifest.model_dump(mode="json", warnings=False)))


def train_source_attested_raster_text_recognizer_overfit(
    document_preview_root: Path,
    *,
    package_inventory_path: Path,
    runtime_bundle_manifest_sha256: str,
    max_epochs: int = 8,
    batch_size: int = 32,
    seed: int = _DEFAULT_SEED,
) -> TrainingRunResult:
    """Train the bounded 1,000-crop stage and publish only its best static model."""

    if not 1 <= max_epochs <= 30 or not 1 <= batch_size <= 128:
        raise SourceAttestedRecognizerTrainingError("training_configuration_invalid")
    runtime_bundle_sha = _strict_sha(runtime_bundle_manifest_sha256)
    package_inventory_sha = verify_paddle_training_package_inventory(package_inventory_path)
    corpus = build_verified_gold_crop_corpus(
        document_preview_root, stage="overfit", seed=seed, include_merged_lines=True
    )
    dictionary_sha = sha256_bytes(canonical_json_bytes(list(corpus.characters)))
    training_module_sha = sha256_file(Path(__file__).resolve(strict=True))
    parent_python_revision = f"{sys.version_info.major}.{sys.version_info.minor}"
    intent = {
        "batch_size": batch_size,
        "character_dictionary_sha256": dictionary_sha,
        "development_manifest_sha256": _manifest_sha256(corpus.development_manifest),
        "max_epochs": max_epochs,
        "model_revision": "tiny-ds-cnn-bigru-ctc-v1",
        "package_inventory_sha256": package_inventory_sha,
        "parent_python_revision": parent_python_revision,
        "seed_sha256": sha256_bytes(str(seed).encode("ascii")),
        "stage": "overfit",
        "train_manifest_sha256": _manifest_sha256(corpus.train_manifest),
        "trainer_revision": _TRAINER_REVISION,
        "training_module_sha256": training_module_sha,
        "training_runtime_bundle_manifest_sha256": runtime_bundle_sha,
    }
    intent_sha = sha256_bytes(canonical_json_bytes(intent))
    existing = find_verified_recognizer_release_by_intent(document_preview_root, intent_sha)
    if existing is not None:
        return TrainingRunResult(release=existing, reused=True)
    try:
        import paddle
    except ImportError:
        raise SourceAttestedRecognizerTrainingError("training_runtime_unavailable") from None
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
        paddle.seed(seed)
        paddle.set_device("cpu")
    except Exception:
        raise SourceAttestedRecognizerTrainingError("training_runtime_unavailable") from None
    character_to_index = {character: index + 1 for index, character in enumerate(corpus.characters)}
    model = build_tiny_paddle_crnn_bigru_ctc(paddle, character_count=len(corpus.characters))
    optimizer = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
    stage = create_recognizer_training_stage(document_preview_root)
    best_checkpoint = stage / "best.pdparams"
    best_cer = math.inf
    best_exact = 0.0
    best_epoch = 0
    stale_epochs = 0
    completed_epochs = 0
    try:
        page_cache: PageLruCache[Any] = PageLruCache(16)
        for epoch in range(max_epochs):
            for batch in deterministic_width_batches(
                corpus.train_crops, epoch=epoch, seed=seed, batch_size=batch_size
            ):
                images, labels, input_lengths, label_lengths = _prepare_paddle_batch(
                    paddle,
                    document_preview_root=document_preview_root,
                    crops=batch,
                    labels_by_sample_id=corpus.labels_by_sample_id,
                    character_to_index=character_to_index,
                    page_cache=page_cache,
                )
                ctc_train_batch_step(
                    paddle,
                    model=model,
                    optimizer=optimizer,
                    images=images,
                    labels=labels,
                    input_lengths=input_lengths,
                    label_lengths=label_lengths,
                )
            completed_epochs = epoch + 1
            cer, exact_match = _evaluate_crop_model(
                paddle,
                model=model,
                document_preview_root=document_preview_root,
                crops=corpus.development_crops,
                labels_by_sample_id=corpus.labels_by_sample_id,
                character_to_index=character_to_index,
                seed=seed,
                batch_size=batch_size,
            )
            if cer + 1e-9 < best_cer:
                best_cer = cer
                best_exact = exact_match
                best_epoch = completed_epochs
                paddle.save(model.state_dict(), str(best_checkpoint))
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= 3:
                    break
        if best_epoch < 1 or not best_checkpoint.is_file():
            raise SourceAttestedRecognizerTrainingError("training_no_valid_checkpoint")
        model.set_state_dict(paddle.load(str(best_checkpoint)))
        best_checkpoint.unlink()
        artifact_hashes = export_tiny_paddle_static_recognizer(
            paddle, model=model, text_rec_directory=stage / "text-rec", characters=corpus.characters
        )
        receipt_payload = {
            "schema_version": "1.0",
            "artifact_kind": "source_attested_raster_text_recognizer_training_receipt",
            "trainer_revision": _TRAINER_REVISION,
            "stage": "overfit",
            "training_intent_sha256": intent_sha,
            "train_manifest_sha256": intent["train_manifest_sha256"],
            "development_manifest_sha256": intent["development_manifest_sha256"],
            "character_dictionary_sha256": dictionary_sha,
            "package_inventory_sha256": package_inventory_sha,
            "training_runtime_bundle_manifest_sha256": runtime_bundle_sha,
            "parent_python_revision": parent_python_revision,
            "training_module_sha256": training_module_sha,
            "seed_sha256": intent["seed_sha256"],
            "batch_size": batch_size,
            "maximum_epochs": max_epochs,
            "include_merged_lines": True,
            "initialization": "deterministic_random_no_pretrained_checkpoint",
            "framework_license": "Apache-2.0",
            "model_revision": "tiny-ds-cnn-bigru-ctc-v1",
            "completed_epochs": completed_epochs,
            "best_epoch": best_epoch,
            "early_stopped": completed_epochs < max_epochs,
            "development_cer": round(float(best_cer), 10),
            "development_exact_match": round(float(best_exact), 10),
            "model_artifact_sha256s": artifact_hashes,
        }
        release = publish_source_attested_raster_text_recognizer_release(
            document_preview_root, staged_release=stage, receipt_payload=receipt_payload
        )
        return TrainingRunResult(release=release, reused=False)
    except SourceAttestedRecognizerTrainingError:
        if stage.exists() and stage.name.startswith(".staging-"):
            shutil.rmtree(stage)
        raise
    except Exception:
        if stage.exists() and stage.name.startswith(".staging-"):
            shutil.rmtree(stage)
        raise SourceAttestedRecognizerTrainingError("training_execution_failed") from None


__all__ = [
    "CropManifestV1",
    "PageLruCache",
    "RecognizerReleaseManifestV1",
    "SourceAttestedRecognizerTrainingError",
    "TrainingCropV1",
    "TrainingReceiptV1",
    "TrainingRunResult",
    "VerifiedCropCorpus",
    "VerifiedRecognizerRelease",
    "build_tiny_paddle_crnn_bigru_ctc",
    "build_verified_gold_crop_corpus",
    "canonical_json_bytes",
    "canonical_json_file_bytes",
    "create_recognizer_training_stage",
    "ctc_train_batch_step",
    "deterministic_width_batches",
    "estimated_recognition_width",
    "export_tiny_paddle_static_recognizer",
    "load_verified_source_attested_raster_text_recognizer_release",
    "publish_source_attested_raster_text_recognizer_release",
    "sha256_bytes",
    "sha256_file",
    "train_source_attested_raster_text_recognizer_overfit",
    "training_receipt_sha256",
    "verify_paddle_training_package_inventory",
]
