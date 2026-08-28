"""Balanced, source-attested raster crops for public recognizer adaptation.

Only train/development families are eligible.  The bounded stage pairs each
semantic crop across PNG 216, JPEG 216 q75 and JPEG 150 q75.  The image-only
scan PDF is independently re-read and pixel-compared with JPEG 216 before it
is excluded as a duplicate, keeping the training set useful and small.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence
import unicodedata

from customs_ai_agent.evaluation.source_attested_raster_text_recognizer_training import (
    TrainingCropV1,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)

if TYPE_CHECKING:
    from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
        RasterTextFreshScopeSelectionReceiptV1,
        RasterTextFreshScopeSelectionReceiptV2,
    )


_DOCUMENT_PREVIEW_ROOT_SUFFIX = ("_local", "data", "document-preview")
_TRAINING_PROFILES = ("png-216", "jpeg-216-q75", "jpeg-150-q75")
_ALL_PROFILES = (*_TRAINING_PROFILES, "scan-pdf-216-q75")
_EXPECTED_MEDIA = {
    "png-216": "image/png",
    "jpeg-216-q75": "image/jpeg",
    "jpeg-150-q75": "image/jpeg",
    "scan-pdf-216-q75": "application/pdf",
}
_PROFILE_QUOTAS = {
    "png-216": 333,
    "jpeg-216-q75": 333,
    "jpeg-150-q75": 334,
}
_LEAKAGE_SAFE_SELECTION_REVISION: Literal["source-receipt-train-holdout-v2"] = (
    "source-receipt-train-holdout-v2"
)
_LEAKAGE_SAFE_COMPILER_REVISION = "source-attested-family-first-two-profile-crops-v2"
_LEAKAGE_SAFE_TRAIN_FAMILY_COUNT = 500
_LEAKAGE_SAFE_DEVELOPMENT_FAMILY_COUNT = 100
_LEAKAGE_SAFE_CROPS_PER_FAMILY = 2
_DOCUMENT_TYPES = (
    "bill_of_lading",
    "commercial_invoice",
    "packing_list",
)
_MAX_JSON_BYTES = 128 * 1024 * 1024
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


class SourceAttestedPublicCropCorpusError(RuntimeError):
    """Content-free validation error for the bounded corpus compiler."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class _GoldSpan:
    span_id: str
    reading_order: int
    bbox: tuple[float, float, float, float]
    label: str


@dataclass(frozen=True, slots=True)
class _Candidate:
    kind: Literal["span", "merged_line"]
    span_ids: tuple[str, ...]
    bbox: tuple[float, float, float, float]
    label: str


@dataclass(frozen=True, slots=True)
class _FieldCandidate:
    field_id: str
    candidate: _Candidate


@dataclass(slots=True)
class VerifiedBalancedPublicCropCorpusV1:
    train_crops: tuple[TrainingCropV1, ...]
    development_crops: tuple[TrainingCropV1, ...]
    labels_by_sample_id: dict[str, str]
    profile_by_sample_id: dict[str, str]
    source_receipt_sha256: str
    raster_receipt_sha256: str
    train_profile_crop_counts: Mapping[str, int]
    development_profile_crop_counts: Mapping[str, int]
    train_scan_duplicate_pair_count: int
    development_scan_duplicate_pair_count: int
    train_scan_duplicate_pixel_list_sha256: str
    development_scan_duplicate_pixel_list_sha256: str


@dataclass(slots=True)
class VerifiedLeakageSafePublicCropCorpusV2:
    """Family-first train/development crops isolated from both holdout generations."""

    train_crops: tuple[TrainingCropV1, ...]
    development_crops: tuple[TrainingCropV1, ...]
    labels_by_sample_id: dict[str, str]
    profile_by_sample_id: dict[str, str]
    field_key_by_sample_id: dict[str, str]
    document_type_by_sample_id: dict[str, str]
    source_receipt_sha256: str
    raster_receipt_sha256: str
    selection_receipt_relative_path: str
    selection_receipt_sha256: str
    parent_selection_receipt_sha256: str
    compiler_revision: str
    train_family_ids: tuple[str, ...]
    development_family_ids: tuple[str, ...]
    excluded_family_ids: tuple[str, ...]
    train_family_ids_sha256: str
    development_family_ids_sha256: str
    excluded_family_ids_sha256: str
    train_profile_crop_counts: Mapping[str, int]
    development_profile_crop_counts: Mapping[str, int]
    train_field_crop_counts: Mapping[str, int]
    development_field_crop_counts: Mapping[str, int]
    train_document_type_crop_counts: Mapping[str, int]
    development_document_type_crop_counts: Mapping[str, int]
    crops_per_family: int
    holdout_overlap_count: int


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
        raise SourceAttestedPublicCropCorpusError("public_crop_root_invalid") from None


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SourceAttestedPublicCropCorpusError("public_crop_path_invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SourceAttestedPublicCropCorpusError("public_crop_path_invalid")
    return path.as_posix()


def _safe(root: Path, value: object, *, directory: bool = False) -> Path:
    try:
        path = root.joinpath(*PurePosixPath(_relative(value)).parts)
        if path.is_symlink():
            raise OSError
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
        if directory and not resolved.is_dir():
            raise OSError
        if not directory and not resolved.is_file():
            raise OSError
        return resolved
    except (OSError, ValueError):
        raise SourceAttestedPublicCropCorpusError("public_crop_path_invalid") from None


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or not 1 <= path.stat().st_size <= _MAX_JSON_BYTES
        ):
            raise OSError
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError
        return value, raw
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise SourceAttestedPublicCropCorpusError("public_crop_json_invalid") from None


def _sha(value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SourceAttestedPublicCropCorpusError("public_crop_binding_invalid")
    return value


def _identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 160
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", value) is None
    ):
        raise SourceAttestedPublicCropCorpusError("public_crop_binding_invalid")
    return value


def _integer(value: object, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise SourceAttestedPublicCropCorpusError("public_crop_binding_invalid")
    return value


def _bbox(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, dict) or set(value) != {"x", "y", "width", "height"}:
        raise SourceAttestedPublicCropCorpusError("public_crop_gold_invalid")
    raw = [value[key] for key in ("x", "y", "width", "height")]
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in raw):
        raise SourceAttestedPublicCropCorpusError("public_crop_gold_invalid")
    result = tuple(float(item) for item in raw)
    x, y, width, height = result
    if (
        not all(math.isfinite(item) for item in result)
        or x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > 1.000000001
        or y + height > 1.000000001
    ):
        raise SourceAttestedPublicCropCorpusError("public_crop_gold_invalid")
    return result  # type: ignore[return-value]


def _vertical_overlap(first: _GoldSpan, second: _GoldSpan) -> float:
    first_bottom = first.bbox[1] + first.bbox[3]
    second_bottom = second.bbox[1] + second.bbox[3]
    overlap = max(0.0, min(first_bottom, second_bottom) - max(first.bbox[1], second.bbox[1]))
    return overlap / min(first.bbox[3], second.bbox[3])


def _merged_line_groups(spans: Sequence[_GoldSpan]) -> tuple[tuple[_GoldSpan, ...], ...]:
    rows: list[list[_GoldSpan]] = []
    for span in sorted(
        spans,
        key=lambda item: (
            round(item.bbox[1] + item.bbox[3] / 2, 12),
            round(item.bbox[0], 12),
            item.reading_order,
            item.span_id,
        ),
    ):
        chosen: list[_GoldSpan] | None = None
        for row in rows:
            anchor = row[0]
            center_gap = abs(
                (span.bbox[1] + span.bbox[3] / 2) - (anchor.bbox[1] + anchor.bbox[3] / 2)
            )
            if (
                _vertical_overlap(span, anchor) >= 0.45
                and center_gap <= max(span.bbox[3], anchor.bbox[3]) * 0.65
            ):
                chosen = row
                break
        if chosen is None:
            rows.append([span])
        else:
            chosen.append(span)
    result: list[tuple[_GoldSpan, ...]] = []
    for row in rows:
        current: list[_GoldSpan] = []
        for span in sorted(row, key=lambda item: (item.bbox[0], item.reading_order, item.span_id)):
            if current:
                previous = current[-1]
                gap = span.bbox[0] - (previous.bbox[0] + previous.bbox[2])
                if gap > 2.75 * max(span.bbox[3], previous.bbox[3]):
                    if len(current) > 1:
                        result.append(tuple(current))
                    current = []
            current.append(span)
        if len(current) > 1:
            result.append(tuple(current))
    return tuple(result)


def _union_bbox(spans: Sequence[_GoldSpan]) -> tuple[float, float, float, float]:
    left = min(item.bbox[0] for item in spans)
    top = min(item.bbox[1] for item in spans)
    right = max(item.bbox[0] + item.bbox[2] for item in spans)
    bottom = max(item.bbox[1] + item.bbox[3] for item in spans)
    return left, top, right - left, bottom - top


def _gold_candidates(
    gold: Mapping[str, object],
    *,
    asset_id: str,
    family_id: str,
    profile_id: str,
    source_sha256: str,
) -> tuple[_Candidate, ...]:
    if (
        gold.get("asset_id") != asset_id
        or gold.get("family_id") != family_id
        or gold.get("profile_id") != profile_id
        or gold.get("source_pdf_sha256") != source_sha256
    ):
        raise SourceAttestedPublicCropCorpusError("public_crop_gold_binding_invalid")
    raw_spans = gold.get("spans")
    if not isinstance(raw_spans, list) or not raw_spans:
        raise SourceAttestedPublicCropCorpusError("public_crop_gold_invalid")
    spans: list[_GoldSpan] = []
    seen: set[str] = set()
    for value in raw_spans:
        if not isinstance(value, dict):
            raise SourceAttestedPublicCropCorpusError("public_crop_gold_invalid")
        span_id = _identifier(value.get("span_id"))
        reading_order = _integer(value.get("reading_order"), minimum=0, maximum=100_000)
        text = value.get("text")
        if not isinstance(text, str) or not text or len(text) > 512 or span_id in seen:
            raise SourceAttestedPublicCropCorpusError("public_crop_gold_invalid")
        seen.add(span_id)
        spans.append(
            _GoldSpan(
                span_id=span_id,
                reading_order=reading_order,
                bbox=_bbox(value.get("bbox")),
                label=unicodedata.normalize("NFC", text),
            )
        )
    candidates = [_Candidate("span", (span.span_id,), span.bbox, span.label) for span in spans]
    for group in _merged_line_groups(spans):
        candidates.append(
            _Candidate(
                "merged_line",
                tuple(item.span_id for item in group),
                _union_bbox(group),
                " ".join(item.label for item in group),
            )
        )
    return tuple(candidates)


def _gold_field_candidates_v2(
    gold: Mapping[str, object],
    *,
    asset_id: str,
    family_id: str,
    profile_id: str,
    source_sha256: str,
) -> tuple[_FieldCandidate, ...]:
    """Read only source-Gold fields whose evidence spans reproduce the value."""

    _gold_candidates(
        gold,
        asset_id=asset_id,
        family_id=family_id,
        profile_id=profile_id,
        source_sha256=source_sha256,
    )
    raw_spans = gold.get("spans")
    raw_fields = gold.get("fields")
    if not isinstance(raw_spans, list) or not isinstance(raw_fields, list) or not raw_fields:
        raise SourceAttestedPublicCropCorpusError("public_crop_gold_invalid")
    spans: dict[str, tuple[int, str]] = {}
    for value in raw_spans:
        if not isinstance(value, dict):
            raise SourceAttestedPublicCropCorpusError("public_crop_gold_invalid")
        span_id = _identifier(value.get("span_id"))
        text = value.get("text")
        if not isinstance(text, str) or span_id in spans:
            raise SourceAttestedPublicCropCorpusError("public_crop_gold_invalid")
        spans[span_id] = (
            _integer(value.get("reading_order"), minimum=0, maximum=100_000),
            unicodedata.normalize("NFC", text),
        )
    result: list[_FieldCandidate] = []
    seen_fields: set[str] = set()
    for value in raw_fields:
        if not isinstance(value, dict):
            raise SourceAttestedPublicCropCorpusError("public_crop_gold_invalid")
        field_id = _identifier(value.get("field_id"))
        field_value = value.get("value")
        raw_evidence = value.get("evidence_span_ids")
        if (
            value.get("asset_id") != asset_id
            or value.get("profile_id") != profile_id
            or value.get("source_pdf_sha256") != source_sha256
            or field_id in seen_fields
            or not isinstance(field_value, str)
            or not isinstance(raw_evidence, list)
            or not raw_evidence
        ):
            raise SourceAttestedPublicCropCorpusError("public_crop_gold_invalid")
        evidence = tuple(_identifier(item) for item in raw_evidence)
        if len(evidence) != len(set(evidence)) or any(item not in spans for item in evidence):
            raise SourceAttestedPublicCropCorpusError("public_crop_gold_invalid")
        ordered = tuple(sorted(evidence, key=lambda item: (spans[item][0], item)))
        normalized_value = unicodedata.normalize("NFC", field_value)
        if " ".join(spans[item][1] for item in ordered) != normalized_value:
            continue
        seen_fields.add(field_id)
        result.append(
            _FieldCandidate(
                field_id=field_id,
                candidate=_Candidate(
                    kind="span" if len(ordered) == 1 else "merged_line",
                    span_ids=ordered,
                    bbox=_bbox(value.get("bbox")),
                    label=normalized_value,
                ),
            )
        )
    return tuple(sorted(result, key=lambda item: item.field_id))


def _verify_regular_material(path: Path, record: Mapping[str, object], hash_key: str) -> None:
    expected_size = _integer(record.get("byte_size"), minimum=1, maximum=2**63 - 1)
    if path.stat().st_size != expected_size or sha256_file(path) != _sha(record.get(hash_key)):
        raise SourceAttestedPublicCropCorpusError("public_crop_material_drift")


def _decode_rgb(path: Path, *, profile_id: str, width: int, height: int) -> Any:
    try:
        from PIL import Image

        with Image.open(path) as image:
            expected_format = "PNG" if profile_id == "png-216" else "JPEG"
            if image.format != expected_format or image.size != (width, height):
                raise ValueError
            image.load()
            return image.convert("RGB")
    except Exception:
        raise SourceAttestedPublicCropCorpusError("public_crop_raster_decode_invalid") from None


def _verify_scan_duplicate(
    *,
    jpeg_path: Path,
    scan_path: Path,
    expected_width: int,
    expected_height: int,
) -> str:
    document: Any = None
    try:
        import pypdfium2 as pdfium

        jpeg = _decode_rgb(
            jpeg_path,
            profile_id="jpeg-216-q75",
            width=expected_width,
            height=expected_height,
        )
        document = pdfium.PdfDocument(scan_path)
        if len(document) != 1:
            raise ValueError
        page = document[0]
        text_page = page.get_textpage()
        try:
            extracted = text_page.get_text_bounded()
        except AttributeError:
            extracted = text_page.get_text_range()
        if extracted != "":
            raise ValueError
        rendered = page.render(scale=3.0).to_pil().convert("RGB")
        if rendered.size != (expected_width, expected_height):
            raise ValueError
        jpeg_sha = hashlib.sha256(jpeg.tobytes()).hexdigest()
        scan_sha = hashlib.sha256(rendered.tobytes()).hexdigest()
        if jpeg_sha != scan_sha:
            raise ValueError
        return jpeg_sha
    except SourceAttestedPublicCropCorpusError:
        raise
    except Exception:
        raise SourceAttestedPublicCropCorpusError("public_crop_scan_duplicate_invalid") from None
    finally:
        close = getattr(document, "close", None)
        if callable(close):
            close()


def _crop(
    *,
    asset: Mapping[str, object],
    profile_id: str,
    candidate: _Candidate,
) -> TrainingCropV1:
    asset_id = _identifier(asset.get("asset_id"))
    split_value = asset.get("split")
    if split_value not in {"train", "development"}:
        raise SourceAttestedPublicCropCorpusError("public_crop_split_invalid")
    split: Literal["train", "development"] = split_value
    sample_payload = {
        "asset_id": asset_id,
        "bbox": [round(value, 12) for value in candidate.bbox],
        "kind": candidate.kind,
        "profile_id": profile_id,
        "span_ids": list(candidate.span_ids),
    }
    return TrainingCropV1(
        sample_id=sha256_bytes(canonical_json_bytes(sample_payload)),
        asset_id=asset_id,
        family_id=_identifier(asset.get("family_id")),
        split=split,
        sample_kind=candidate.kind,
        asset_relative_path=_relative(asset.get("asset_relative_path")),
        gold_relative_path=_relative(asset.get("gold_relative_path")),
        source_pdf_sha256=_sha(asset.get("source_pdf_sha256")),
        raster_sha256=_sha(asset.get("image_sha256")),
        gold_sha256=_sha(asset.get("gold_sha256")),
        span_ids=candidate.span_ids,
        bbox=candidate.bbox,
        label_sha256=sha256_bytes(candidate.label.encode("utf-8")),
        label_length=len(candidate.label),
        page_width=_integer(asset.get("width"), minimum=1, maximum=20_000),
        page_height=_integer(asset.get("height"), minimum=1, maximum=20_000),
    )


def _compile_split(
    *,
    root: Path,
    source_root: Path,
    sources: Mapping[str, Mapping[str, object]],
    groups: Mapping[str, Mapping[str, Mapping[str, object]]],
    split: Literal["train", "development"],
    seed: int,
) -> tuple[
    tuple[TrainingCropV1, ...],
    dict[str, str],
    dict[str, str],
    int,
    str,
]:
    quotas = dict(_PROFILE_QUOTAS)
    crops: list[TrainingCropV1] = []
    labels: dict[str, str] = {}
    profiles: dict[str, str] = {}
    duplicate_records: list[dict[str, str]] = []
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: sha256_bytes(f"{seed}:{split}:{item[0]}".encode("utf-8")),
    )
    for _group_key, by_profile in ordered_groups:
        if all(value == 0 for value in quotas.values()):
            break
        if set(by_profile) != set(_ALL_PROFILES):
            raise SourceAttestedPublicCropCorpusError("public_crop_profile_set_invalid")
        exemplar = by_profile["png-216"]
        asset_id = _identifier(exemplar.get("asset_id"))
        family_id = _identifier(exemplar.get("family_id"))
        source_sha = _sha(exemplar.get("source_pdf_sha256"))
        if exemplar.get("split") != split:
            raise SourceAttestedPublicCropCorpusError("public_crop_split_invalid")
        source = sources.get(source_sha)
        if (
            source is None
            or source.get("asset_id") != asset_id
            or source.get("family_id") != family_id
        ):
            raise SourceAttestedPublicCropCorpusError("public_crop_source_binding_invalid")
        source_path = _safe(source_root, source.get("source_relative_path"))
        _verify_regular_material(source_path, source, "source_pdf_sha256")
        candidate_sets: dict[str, tuple[_Candidate, ...]] = {}
        paths: dict[str, Path] = {}
        for profile_id in _ALL_PROFILES:
            asset = by_profile[profile_id]
            if (
                asset.get("asset_id") != asset_id
                or asset.get("family_id") != family_id
                or asset.get("source_pdf_sha256") != source_sha
                or asset.get("split") != split
                or asset.get("profile_id") != profile_id
                or asset.get("media_type") != _EXPECTED_MEDIA[profile_id]
            ):
                raise SourceAttestedPublicCropCorpusError("public_crop_profile_binding_invalid")
            raster_path = _safe(root, asset.get("asset_relative_path"))
            gold_path = _safe(root, asset.get("gold_relative_path"))
            _verify_regular_material(raster_path, asset, "image_sha256")
            if sha256_file(gold_path) != _sha(asset.get("gold_sha256")):
                raise SourceAttestedPublicCropCorpusError("public_crop_material_drift")
            gold, _raw = _read_json(gold_path)
            candidate_sets[profile_id] = _gold_candidates(
                gold,
                asset_id=asset_id,
                family_id=family_id,
                profile_id=profile_id,
                source_sha256=source_sha,
            )
            paths[profile_id] = raster_path
            if profile_id in _TRAINING_PROFILES and profile_id != "jpeg-216-q75":
                _decode_rgb(
                    raster_path,
                    profile_id=profile_id,
                    width=_integer(asset.get("width"), minimum=1, maximum=20_000),
                    height=_integer(asset.get("height"), minimum=1, maximum=20_000),
                )
        canonical_candidates = candidate_sets["png-216"]
        if any(candidate_sets[profile] != canonical_candidates for profile in _ALL_PROFILES[1:]):
            raise SourceAttestedPublicCropCorpusError("public_crop_profile_gold_mismatch")
        width = _integer(exemplar.get("width"), minimum=1, maximum=20_000)
        height = _integer(exemplar.get("height"), minimum=1, maximum=20_000)
        if any(
            _integer(by_profile[profile].get("width"), minimum=1, maximum=20_000) != width
            or _integer(by_profile[profile].get("height"), minimum=1, maximum=20_000) != height
            for profile in ("jpeg-216-q75", "scan-pdf-216-q75")
        ):
            raise SourceAttestedPublicCropCorpusError("public_crop_profile_geometry_invalid")
        pixel_sha = _verify_scan_duplicate(
            jpeg_path=paths["jpeg-216-q75"],
            scan_path=paths["scan-pdf-216-q75"],
            expected_width=width,
            expected_height=height,
        )
        duplicate_records.append(
            {
                "asset_id_sha256": sha256_bytes(asset_id.encode("utf-8")),
                "pixel_sha256": pixel_sha,
            }
        )
        for candidate in sorted(
            canonical_candidates,
            key=lambda item: sha256_bytes(
                canonical_json_bytes(
                    {
                        "asset_id": asset_id,
                        "bbox": item.bbox,
                        "kind": item.kind,
                        "seed": seed,
                        "span_ids": item.span_ids,
                    }
                )
            ),
        ):
            for profile_id in _TRAINING_PROFILES:
                if quotas[profile_id] == 0:
                    continue
                crop = _crop(
                    asset=by_profile[profile_id], profile_id=profile_id, candidate=candidate
                )
                if crop.sample_id in labels:
                    raise SourceAttestedPublicCropCorpusError("public_crop_sample_collision")
                crops.append(crop)
                labels[crop.sample_id] = candidate.label
                profiles[crop.sample_id] = profile_id
                quotas[profile_id] -= 1
            if all(value == 0 for value in quotas.values()):
                break
    if any(quotas.values()) or len(crops) != 1_000:
        raise SourceAttestedPublicCropCorpusError("public_crop_bounded_set_unavailable")
    return (
        tuple(sorted(crops, key=lambda item: item.sample_id)),
        labels,
        profiles,
        len(duplicate_records),
        sha256_bytes(canonical_json_bytes(duplicate_records)),
    )


def build_verified_balanced_public_crop_corpus_v1(
    document_preview_root: Path,
    *,
    seed: int,
) -> VerifiedBalancedPublicCropCorpusV1:
    """Build exactly 1k balanced train and 1k balanced development crops."""

    root = _root(document_preview_root)
    dataset = _safe(root, "synthetic-raster-v1", directory=True)
    raster_receipt, raster_raw = _read_json(dataset / "synthetic-raster-receipt.json")
    source_receipt, source_raw = _read_json(dataset / "source-receipt.json")
    source_sha = sha256_bytes(source_raw)
    raster_sha = sha256_bytes(raster_raw)
    if (
        raster_receipt.get("dataset_id") not in {None, "synthetic-raster-v1"}
        or raster_receipt.get("source_receipt_sha256") != source_sha
    ):
        raise SourceAttestedPublicCropCorpusError("public_crop_receipt_binding_invalid")
    source_root = _safe(root, source_receipt.get("source_root_relative_path"), directory=True)
    raw_sources = source_receipt.get("assets")
    raw_assets = raster_receipt.get("assets")
    if not isinstance(raw_sources, list) or not isinstance(raw_assets, list):
        raise SourceAttestedPublicCropCorpusError("public_crop_receipt_invalid")
    sources: dict[str, Mapping[str, object]] = {}
    for value in raw_sources:
        if not isinstance(value, dict):
            raise SourceAttestedPublicCropCorpusError("public_crop_receipt_invalid")
        digest = _sha(value.get("source_pdf_sha256"))
        if digest in sources:
            raise SourceAttestedPublicCropCorpusError("public_crop_receipt_invalid")
        sources[digest] = value
    family_splits: dict[str, set[str]] = {}
    grouped: dict[str, dict[str, dict[str, Mapping[str, object]]]] = {
        "train": {},
        "development": {},
    }
    for value in raw_assets:
        if not isinstance(value, dict):
            raise SourceAttestedPublicCropCorpusError("public_crop_receipt_invalid")
        family = _identifier(value.get("family_id"))
        split = value.get("split")
        if split not in {"train", "development", "calibration", "sealed"}:
            raise SourceAttestedPublicCropCorpusError("public_crop_split_invalid")
        family_splits.setdefault(family, set()).add(str(split))
        if split not in {"train", "development"}:
            continue
        profile = value.get("profile_id")
        if profile not in _ALL_PROFILES:
            raise SourceAttestedPublicCropCorpusError("public_crop_profile_set_invalid")
        asset_id = _identifier(value.get("asset_id"))
        profile_map = grouped[str(split)].setdefault(asset_id, {})
        if str(profile) in profile_map:
            raise SourceAttestedPublicCropCorpusError("public_crop_profile_set_invalid")
        profile_map[str(profile)] = value
    if any(len(values) != 1 for values in family_splits.values()):
        raise SourceAttestedPublicCropCorpusError("public_crop_family_leakage")
    train, train_labels, train_profiles, train_pairs, train_pair_sha = _compile_split(
        root=root,
        source_root=source_root,
        sources=sources,
        groups=grouped["train"],
        split="train",
        seed=seed,
    )
    development, dev_labels, dev_profiles, dev_pairs, dev_pair_sha = _compile_split(
        root=root,
        source_root=source_root,
        sources=sources,
        groups=grouped["development"],
        split="development",
        seed=seed + 1,
    )
    if {item.family_id for item in train} & {item.family_id for item in development}:
        raise SourceAttestedPublicCropCorpusError("public_crop_family_leakage")
    labels = {**train_labels, **dev_labels}
    profiles = {**train_profiles, **dev_profiles}
    if len(labels) != 2_000 or len(profiles) != 2_000:
        raise SourceAttestedPublicCropCorpusError("public_crop_sample_collision")
    return VerifiedBalancedPublicCropCorpusV1(
        train_crops=train,
        development_crops=development,
        labels_by_sample_id=labels,
        profile_by_sample_id=profiles,
        source_receipt_sha256=source_sha,
        raster_receipt_sha256=raster_sha,
        train_profile_crop_counts=dict(_PROFILE_QUOTAS),
        development_profile_crop_counts=dict(_PROFILE_QUOTAS),
        train_scan_duplicate_pair_count=train_pairs,
        development_scan_duplicate_pair_count=dev_pairs,
        train_scan_duplicate_pixel_list_sha256=train_pair_sha,
        development_scan_duplicate_pixel_list_sha256=dev_pair_sha,
    )


def _family_ids_sha256(values: Sequence[str]) -> str:
    canonical = tuple(sorted(values))
    if len(canonical) != len(set(canonical)):
        raise SourceAttestedPublicCropCorpusError("public_crop_family_leakage")
    return sha256_bytes(canonical_json_bytes(list(canonical)))


def _canonical_selection_bytes(value: object) -> bytes:
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        raise SourceAttestedPublicCropCorpusError("public_crop_selection_receipt_invalid")
    return canonical_json_bytes(model_dump(mode="json", warnings=False))


def _load_leakage_safe_selection_v2(
    *,
    root: Path,
    raster_receipt_sha256: str,
    source_receipt_sha256: str,
) -> tuple[
    RasterTextFreshScopeSelectionReceiptV2,
    RasterTextFreshScopeSelectionReceiptV1,
    str,
    str,
]:
    """Create/re-read the result-blind V2 receipt and its immutable V1 parent."""

    try:
        from pydantic import ValidationError

        from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
            RasterTextFreshScopeSelectionReceiptV1,
            RasterTextFreshScopeSelectionReceiptV2,
            SyntheticRasterTextActivationError,
            write_fresh_raster_text_scope_selection_receipt,
        )

        relative_path, receipt_sha256 = write_fresh_raster_text_scope_selection_receipt(
            document_preview_root=root,
            selection_revision=_LEAKAGE_SAFE_SELECTION_REVISION,
        )
        path = _safe(root, relative_path)
        value, raw = _read_json(path)
        selection = RasterTextFreshScopeSelectionReceiptV2.model_validate(value)
        if (
            raw != _canonical_selection_bytes(selection)
            or sha256_bytes(raw) != receipt_sha256
            or selection.synthetic_raster_receipt_sha256 != raster_receipt_sha256
            or selection.source_receipt_sha256 != source_receipt_sha256
        ):
            raise ValueError
        parent_path = _safe(root, selection.parent_selection_receipt_relative_path)
        parent_value, parent_raw = _read_json(parent_path)
        parent = RasterTextFreshScopeSelectionReceiptV1.model_validate(parent_value)
        if (
            parent_raw != _canonical_selection_bytes(parent)
            or sha256_bytes(parent_raw) != selection.parent_selection_receipt_sha256
            or parent.synthetic_raster_receipt_sha256 != raster_receipt_sha256
            or parent.source_receipt_sha256 != source_receipt_sha256
        ):
            raise ValueError
        return selection, parent, relative_path, receipt_sha256
    except SourceAttestedPublicCropCorpusError:
        raise
    except (OSError, ValueError, ValidationError, SyntheticRasterTextActivationError):
        raise SourceAttestedPublicCropCorpusError("public_crop_selection_receipt_invalid") from None


def _trainable_source_candidate(candidate: _Candidate) -> bool:
    label = candidate.label
    return (
        2 <= len(label) <= 64
        and label.isascii()
        and all(character.isprintable() for character in label)
        and any(character.isalnum() for character in label)
    )


def _ranked_profiles_v2(*, family_id: str, split: str, seed: int) -> tuple[str, str]:
    ranked = sorted(
        _TRAINING_PROFILES,
        key=lambda profile_id: (
            sha256_bytes(
                f"{_LEAKAGE_SAFE_COMPILER_REVISION}:{seed}:{split}:{family_id}:{profile_id}".encode(
                    "ascii"
                )
            ),
            profile_id,
        ),
    )
    return ranked[0], ranked[1]


def _verified_candidates_for_asset_v2(
    *,
    root: Path,
    source_root: Path,
    sources: Mapping[str, Mapping[str, object]],
    by_profile: Mapping[str, Mapping[str, object]],
    family_id: str,
    split: Literal["train", "development"],
    selected_profiles: tuple[str, str],
) -> tuple[tuple[_FieldCandidate, ...], dict[str, Mapping[str, object]]]:
    if set(by_profile) != set(_ALL_PROFILES):
        raise SourceAttestedPublicCropCorpusError("public_crop_profile_set_invalid")
    exemplar = by_profile["png-216"]
    asset_id = _identifier(exemplar.get("asset_id"))
    source_sha = _sha(exemplar.get("source_pdf_sha256"))
    source = sources.get(source_sha)
    if (
        exemplar.get("family_id") != family_id
        or exemplar.get("split") != split
        or source is None
        or source.get("asset_id") != asset_id
        or source.get("family_id") != family_id
    ):
        raise SourceAttestedPublicCropCorpusError("public_crop_source_binding_invalid")
    source_path = _safe(source_root, source.get("source_relative_path"))
    _verify_regular_material(source_path, source, "source_pdf_sha256")
    required_profiles = tuple(dict.fromkeys(("png-216", *selected_profiles)))
    candidate_sets: dict[str, tuple[_FieldCandidate, ...]] = {}
    for profile_id in required_profiles:
        asset = by_profile[profile_id]
        if (
            asset.get("asset_id") != asset_id
            or asset.get("family_id") != family_id
            or asset.get("source_pdf_sha256") != source_sha
            or asset.get("split") != split
            or asset.get("profile_id") != profile_id
            or asset.get("media_type") != _EXPECTED_MEDIA[profile_id]
        ):
            raise SourceAttestedPublicCropCorpusError("public_crop_profile_binding_invalid")
        raster_path = _safe(root, asset.get("asset_relative_path"))
        gold_path = _safe(root, asset.get("gold_relative_path"))
        _verify_regular_material(raster_path, asset, "image_sha256")
        if sha256_file(gold_path) != _sha(asset.get("gold_sha256")):
            raise SourceAttestedPublicCropCorpusError("public_crop_material_drift")
        gold, _raw = _read_json(gold_path)
        candidate_sets[profile_id] = _gold_field_candidates_v2(
            gold,
            asset_id=asset_id,
            family_id=family_id,
            profile_id=profile_id,
            source_sha256=source_sha,
        )
        _decode_rgb(
            raster_path,
            profile_id=profile_id,
            width=_integer(asset.get("width"), minimum=1, maximum=20_000),
            height=_integer(asset.get("height"), minimum=1, maximum=20_000),
        )
    canonical = candidate_sets["png-216"]
    if any(candidate_sets[profile_id] != canonical for profile_id in selected_profiles):
        raise SourceAttestedPublicCropCorpusError("public_crop_profile_gold_mismatch")
    return canonical, {profile: by_profile[profile] for profile in selected_profiles}


def _compile_family_first_split_v2(
    *,
    root: Path,
    source_root: Path,
    sources: Mapping[str, Mapping[str, object]],
    families: Mapping[str, Mapping[str, Mapping[str, Mapping[str, object]]]],
    selected_family_ids: Sequence[str],
    split: Literal["train", "development"],
    seed: int,
) -> tuple[
    tuple[TrainingCropV1, ...],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, int],
    dict[str, int],
    dict[str, int],
]:
    expected_family_count = (
        _LEAKAGE_SAFE_TRAIN_FAMILY_COUNT
        if split == "train"
        else _LEAKAGE_SAFE_DEVELOPMENT_FAMILY_COUNT
    )
    canonical_families = tuple(sorted(selected_family_ids))
    if (
        len(canonical_families) != expected_family_count
        or len(canonical_families) != len(set(canonical_families))
        or set(canonical_families) != set(families)
    ):
        raise SourceAttestedPublicCropCorpusError("public_crop_family_first_coverage_invalid")
    crops: list[TrainingCropV1] = []
    labels: dict[str, str] = {}
    profiles: dict[str, str] = {}
    field_keys: dict[str, str] = {}
    document_types: dict[str, str] = {}
    profile_counts = {profile: 0 for profile in _TRAINING_PROFILES}
    field_counts: dict[str, int] = {}
    document_type_counts = {document_type: 0 for document_type in _DOCUMENT_TYPES}
    ranked_families = sorted(
        canonical_families,
        key=lambda family_id: (
            sha256_bytes(
                f"{_LEAKAGE_SAFE_COMPILER_REVISION}:{seed}:{split}:family-lane:{family_id}".encode(
                    "ascii"
                )
            ),
            family_id,
        ),
    )
    for family_ordinal, family_id in enumerate(ranked_families):
        asset_groups = families[family_id]
        if len(asset_groups) != 6 or any(
            set(by_profile) != set(_ALL_PROFILES) for by_profile in asset_groups.values()
        ):
            raise SourceAttestedPublicCropCorpusError("public_crop_family_first_coverage_invalid")
        selected_profiles = _ranked_profiles_v2(family_id=family_id, split=split, seed=seed)
        target_document_type = _DOCUMENT_TYPES[family_ordinal % len(_DOCUMENT_TYPES)]
        document_assets = {
            asset_id: by_profile
            for asset_id, by_profile in asset_groups.items()
            if by_profile["png-216"].get("document_type") == target_document_type
        }
        if len(document_assets) != 2:
            raise SourceAttestedPublicCropCorpusError("public_crop_family_first_coverage_invalid")
        ranked_assets = sorted(
            document_assets.items(),
            key=lambda item: (
                sha256_bytes(
                    f"{_LEAKAGE_SAFE_COMPILER_REVISION}:{seed}:{split}:"
                    f"{family_id}:{item[0]}".encode("ascii")
                ),
                item[0],
            ),
        )
        selected: tuple[_FieldCandidate, Mapping[str, Mapping[str, object]]] | None = None
        for asset_id, by_profile in ranked_assets:
            candidates, selected_assets = _verified_candidates_for_asset_v2(
                root=root,
                source_root=source_root,
                sources=sources,
                by_profile=by_profile,
                family_id=family_id,
                split=split,
                selected_profiles=selected_profiles,
            )
            usable = [item for item in candidates if _trainable_source_candidate(item.candidate)]
            if not usable:
                continue
            ordered_fields = sorted(usable, key=lambda item: item.field_id)
            document_lane = family_ordinal // len(_DOCUMENT_TYPES)
            field_candidate = ordered_fields[document_lane % len(ordered_fields)]
            selected = field_candidate, selected_assets
            break
        if selected is None:
            raise SourceAttestedPublicCropCorpusError(
                "public_crop_family_first_candidate_unavailable"
            )
        field_candidate, chosen_assets = selected
        candidate = field_candidate.candidate
        field_key = f"{target_document_type}:{field_candidate.field_id}"
        family_crops: list[TrainingCropV1] = []
        for profile_id in selected_profiles:
            crop = _crop(
                asset=chosen_assets[profile_id],
                profile_id=profile_id,
                candidate=candidate,
            )
            if crop.sample_id in labels:
                raise SourceAttestedPublicCropCorpusError("public_crop_sample_collision")
            family_crops.append(crop)
            crops.append(crop)
            labels[crop.sample_id] = candidate.label
            profiles[crop.sample_id] = profile_id
            field_keys[crop.sample_id] = field_key
            document_types[crop.sample_id] = target_document_type
            profile_counts[profile_id] += 1
            field_counts[field_key] = field_counts.get(field_key, 0) + 1
            document_type_counts[target_document_type] += 1
        if len(family_crops) != _LEAKAGE_SAFE_CROPS_PER_FAMILY:
            raise SourceAttestedPublicCropCorpusError("public_crop_family_first_coverage_invalid")
    expected_crop_count = expected_family_count * _LEAKAGE_SAFE_CROPS_PER_FAMILY
    if (
        len(crops) != expected_crop_count
        or len(labels) != expected_crop_count
        or len(profiles) != expected_crop_count
        or len(field_keys) != expected_crop_count
        or len(document_types) != expected_crop_count
        or sum(profile_counts.values()) != expected_crop_count
        or sum(field_counts.values()) != expected_crop_count
        or sum(document_type_counts.values()) != expected_crop_count
        or any(value < 1 for value in profile_counts.values())
        or any(value < 1 for value in field_counts.values())
        or max(document_type_counts.values()) - min(document_type_counts.values()) > 2
        or any(
            sum(crop.family_id == family_id for crop in crops) != _LEAKAGE_SAFE_CROPS_PER_FAMILY
            for family_id in canonical_families
        )
    ):
        raise SourceAttestedPublicCropCorpusError("public_crop_family_first_coverage_invalid")
    return (
        tuple(sorted(crops, key=lambda item: item.sample_id)),
        labels,
        profiles,
        field_keys,
        document_types,
        profile_counts,
        field_counts,
        document_type_counts,
    )


def build_verified_leakage_safe_public_crop_corpus_v2(
    document_preview_root: Path,
    *,
    seed: int,
) -> VerifiedLeakageSafePublicCropCorpusV2:
    """Build 500x2 train and 100x2 development crops with zero holdout overlap."""

    root = _root(document_preview_root)
    dataset = _safe(root, "synthetic-raster-v1", directory=True)
    raster_receipt, raster_raw = _read_json(dataset / "synthetic-raster-receipt.json")
    source_receipt, source_raw = _read_json(dataset / "source-receipt.json")
    source_sha = sha256_bytes(source_raw)
    raster_sha = sha256_bytes(raster_raw)
    if (
        raster_receipt.get("dataset_id") not in {None, "synthetic-raster-v1"}
        or raster_receipt.get("source_receipt_sha256") != source_sha
    ):
        raise SourceAttestedPublicCropCorpusError("public_crop_receipt_binding_invalid")
    selection, parent, selection_relative, selection_sha = _load_leakage_safe_selection_v2(
        root=root,
        raster_receipt_sha256=raster_sha,
        source_receipt_sha256=source_sha,
    )
    source_root = _safe(root, source_receipt.get("source_root_relative_path"), directory=True)
    raw_sources = source_receipt.get("assets")
    raw_assets = raster_receipt.get("assets")
    if not isinstance(raw_sources, list) or not isinstance(raw_assets, list):
        raise SourceAttestedPublicCropCorpusError("public_crop_receipt_invalid")
    sources: dict[str, Mapping[str, object]] = {}
    for value in raw_sources:
        if not isinstance(value, dict):
            raise SourceAttestedPublicCropCorpusError("public_crop_receipt_invalid")
        digest = _sha(value.get("source_pdf_sha256"))
        if digest in sources:
            raise SourceAttestedPublicCropCorpusError("public_crop_receipt_invalid")
        sources[digest] = value
    family_splits: dict[str, set[str]] = {}
    grouped: dict[str, dict[str, dict[str, dict[str, Mapping[str, object]]]]] = {
        "train": {},
        "development": {},
    }
    for value in raw_assets:
        if not isinstance(value, dict):
            raise SourceAttestedPublicCropCorpusError("public_crop_receipt_invalid")
        family_id = _identifier(value.get("family_id"))
        split_value = value.get("split")
        if split_value not in {"train", "development", "calibration", "sealed"}:
            raise SourceAttestedPublicCropCorpusError("public_crop_split_invalid")
        split = str(split_value)
        family_splits.setdefault(family_id, set()).add(split)
        if split not in grouped:
            continue
        profile = value.get("profile_id")
        if profile not in _ALL_PROFILES:
            raise SourceAttestedPublicCropCorpusError("public_crop_profile_set_invalid")
        asset_id = _identifier(value.get("asset_id"))
        profile_map = grouped[split].setdefault(family_id, {}).setdefault(asset_id, {})
        if str(profile) in profile_map:
            raise SourceAttestedPublicCropCorpusError("public_crop_profile_set_invalid")
        profile_map[str(profile)] = value
    if any(len(splits) != 1 for splits in family_splits.values()):
        raise SourceAttestedPublicCropCorpusError("public_crop_family_leakage")
    split_families = {
        split: {family_id for family_id, values in family_splits.items() if values == {split}}
        for split in ("train", "development", "calibration", "sealed")
    }
    if {split: len(values) for split, values in split_families.items()} != {
        "train": 750,
        "development": 100,
        "calibration": 50,
        "sealed": 100,
    }:
        raise SourceAttestedPublicCropCorpusError("public_crop_family_first_coverage_invalid")
    parent_holdouts = set(parent.fresh_calibration_family_ids) | set(parent.fresh_sealed_family_ids)
    new_sealed = set(selection.fresh_sealed_family_ids)
    non_train = set().union(
        split_families["development"],
        split_families["calibration"],
        split_families["sealed"],
    )
    v2_eligible_before_sealed = split_families["train"] - parent_holdouts
    train_families = v2_eligible_before_sealed - new_sealed
    excluded = non_train | parent_holdouts | new_sealed
    if (
        len(parent_holdouts) != 150
        or len(new_sealed) != 100
        or len(non_train) != 250
        or len(v2_eligible_before_sealed) != 600
        or len(train_families) != _LEAKAGE_SAFE_TRAIN_FAMILY_COUNT
        or len(excluded) != 500
        or set(selection.fresh_calibration_family_ids) != set(parent.fresh_calibration_family_ids)
        or _family_ids_sha256(sorted(non_train | parent_holdouts))
        != selection.excluded_family_ids_sha256
        or _family_ids_sha256(sorted(v2_eligible_before_sealed))
        != selection.eligible_family_ids_sha256
        or train_families & excluded
    ):
        raise SourceAttestedPublicCropCorpusError("public_crop_family_leakage")
    train_groups = {family: grouped["train"][family] for family in train_families}
    development_groups = grouped["development"]
    (
        train,
        train_labels,
        train_profiles,
        train_fields,
        train_documents,
        train_counts,
        train_field_counts,
        train_document_counts,
    ) = _compile_family_first_split_v2(
        root=root,
        source_root=source_root,
        sources=sources,
        families=train_groups,
        selected_family_ids=sorted(train_families),
        split="train",
        seed=seed,
    )
    (
        development,
        dev_labels,
        dev_profiles,
        dev_fields,
        dev_documents,
        dev_counts,
        dev_field_counts,
        dev_document_counts,
    ) = _compile_family_first_split_v2(
        root=root,
        source_root=source_root,
        sources=sources,
        families=development_groups,
        selected_family_ids=sorted(split_families["development"]),
        split="development",
        seed=seed + 1,
    )
    train_ids = tuple(sorted({crop.family_id for crop in train}))
    development_ids = tuple(sorted({crop.family_id for crop in development}))
    excluded_ids = tuple(sorted(excluded))
    if (
        len(train) != 1_000
        or len(development) != 200
        or len(train_ids) != 500
        or len(development_ids) != 100
        or set(train_ids) & set(excluded_ids)
        or set(train_ids) & set(development_ids)
    ):
        raise SourceAttestedPublicCropCorpusError("public_crop_family_leakage")
    labels = {**train_labels, **dev_labels}
    profiles = {**train_profiles, **dev_profiles}
    fields = {**train_fields, **dev_fields}
    documents = {**train_documents, **dev_documents}
    if (
        len(labels) != 1_200
        or len(profiles) != 1_200
        or len(fields) != 1_200
        or len(documents) != 1_200
    ):
        raise SourceAttestedPublicCropCorpusError("public_crop_sample_collision")
    return VerifiedLeakageSafePublicCropCorpusV2(
        train_crops=train,
        development_crops=development,
        labels_by_sample_id=labels,
        profile_by_sample_id=profiles,
        field_key_by_sample_id=fields,
        document_type_by_sample_id=documents,
        source_receipt_sha256=source_sha,
        raster_receipt_sha256=raster_sha,
        selection_receipt_relative_path=selection_relative,
        selection_receipt_sha256=selection_sha,
        parent_selection_receipt_sha256=selection.parent_selection_receipt_sha256,
        compiler_revision=_LEAKAGE_SAFE_COMPILER_REVISION,
        train_family_ids=train_ids,
        development_family_ids=development_ids,
        excluded_family_ids=excluded_ids,
        train_family_ids_sha256=_family_ids_sha256(train_ids),
        development_family_ids_sha256=_family_ids_sha256(development_ids),
        excluded_family_ids_sha256=_family_ids_sha256(excluded_ids),
        train_profile_crop_counts=train_counts,
        development_profile_crop_counts=dev_counts,
        train_field_crop_counts=train_field_counts,
        development_field_crop_counts=dev_field_counts,
        train_document_type_crop_counts=train_document_counts,
        development_document_type_crop_counts=dev_document_counts,
        crops_per_family=_LEAKAGE_SAFE_CROPS_PER_FAMILY,
        holdout_overlap_count=0,
    )


__all__ = [
    "SourceAttestedPublicCropCorpusError",
    "VerifiedBalancedPublicCropCorpusV1",
    "VerifiedLeakageSafePublicCropCorpusV2",
    "build_verified_balanced_public_crop_corpus_v1",
    "build_verified_leakage_safe_public_crop_corpus_v2",
]
