"""Offline, fail-closed training for native-PDF supervised template selectors.

This module is intentionally a training-only boundary.  It consumes a
family-safe split, compact value-hash labels, the canonical dataset receipt,
development/sealed *metadata* packets, and only the PDFs selected by the
re-derived training split.  It never opens a development or sealed PDF.

The learner does not receive a source layout profile, source run ordinal,
coordinate, or source-parser transform.  For every field it discovers a
candidate-native span/table selector and one bounded transform by matching
only normalized value hashes in memory.  The resulting model is a supervised
template extractor for this born-digital native-PDF family; it is not an
independent or general optical-OCR system.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Mapping, Sequence, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from customs_ai_agent.evaluation.document_gold import DatasetReceiptV1
from customs_ai_agent.evaluation.document_ocr_quality import (
    EvaluationPacketV2,
    FieldNormalizer,
    SyntheticDocumentType,
    canonical_json_bytes,
    canonical_json_file_bytes,
    normalize_field_value,
    packet_receipt_sha256,
    receipt_sha256,
    sha256_bytes,
)
from customs_ai_agent.evaluation.native_pdf_semantic_adapter import NativePdfSemanticAdapter
from customs_ai_agent.evaluation.native_pdf_semantic_model import (
    HashBoundNativePdfSemanticModelV1,
    NativePdfSemanticDocumentModelV1,
    NativePdfSemanticFieldModelV1,
    NativePdfSemanticModelArtifactV1,
    NativePdfSemanticSelectorV1,
    NativePdfSemanticStructuralSignatureV1,
    bind_native_pdf_semantic_model_artifact,
)
from customs_ai_agent.evaluation.native_pdf_vector_grid import (
    NativePdfVectorGridExtraction,
    extract_native_pdf_vector_grid_from_pdf_bytes,
)
from customs_ai_agent.evaluation.supervised_template_training import (
    SourceAttestedSupervisedTemplateModelTrainingReceiptV1,
    SourceAttestedSupervisedTemplateTrainingLabelsV1,
    SourceAttestedSupervisedTemplateTrainingSplitV1,
    SupervisedTemplateTrainingAssetV1,
    SupervisedTemplateTrainingFieldLabelV1,
    SupervisedTemplateTrainingLabelRecordV1,
    SupervisedTemplateTrainingError,
    build_source_attested_supervised_template_model_training_receipt,
    build_source_attested_supervised_template_training_split,
    write_source_attested_supervised_template_model_training_receipt,
)


_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_SHA256 = r"^[a-f0-9]{64}$"
# The canonical r1 compact label receipt contains 73,760 value digests and is
# about 17 MiB.  Bound it above that known local receipt while still refusing
# arbitrarily large metadata input before any PDF is opened.
_MAX_INPUT_JSON_BYTES: Final = 32 * 1024 * 1024
_MAX_TRAINING_PDF_BYTES: Final = 32 * 1024 * 1024
_ANCHOR_DECIMALS: Final = 6
_ANCHOR_TOLERANCE: Final = 0.00001

SUPERVISED_TEMPLATE_DIRECT_SELECTOR_TRAINER_REVISION: Literal[
    "source-attested-supervised-template-direct-selector-trainer-v1"
] = "source-attested-supervised-template-direct-selector-trainer-v1"
SUPERVISED_TEMPLATE_DIRECT_SELECTOR_RECIPE_KIND: Literal[
    "source_attested_supervised_template_direct_selector_trainer_recipe"
] = "source_attested_supervised_template_direct_selector_trainer_recipe"
SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_KIND: Literal["native_pdf_semantic_model"] = (
    "native_pdf_semantic_model"
)
SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_ID: Literal[
    "synthetic-customs-pdf-2000-r1-native-template"
] = "synthetic-customs-pdf-2000-r1-native-template"
SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_REVISION: Literal["supervised-native-selector-r1-v1"] = (
    "supervised-native-selector-r1-v1"
)

_CANDIDATE_EVIDENCE_REVISION: Final = "native-pdf-vector-grid-v1"
_CANONICAL_SELECTOR_ORDER_REVISION: Final = "candidate-native-selector-order-v1"
_REQUIRED_DOCUMENT_TYPES: Final[tuple[SyntheticDocumentType, ...]] = (
    "bill_of_lading",
    "invoice",
    "packing_list",
)
# This is an explicit semantic output schema, not a source-parser layout
# profile.  It has no source run, coordinate, transform, label text, or
# template identity; the trainer independently learns all selectors from
# candidate-native evidence.
_REQUIRED_FIELDS: Final[Mapping[SyntheticDocumentType, tuple[tuple[str, FieldNormalizer], ...]]] = {
    "bill_of_lading": (
        ("bl_no", "identifier-v1"),
        ("consignee", "text-v1"),
        ("container_no", "identifier-v1"),
        ("containers", "numeric-v1"),
        ("disch_port", "text-v1"),
        ("exporter_name", "text-v1"),
        ("final_dest", "text-v1"),
        ("good", "text-v1"),
        ("gross_weight", "numeric-v1"),
        ("load_port", "text-v1"),
        ("seal_no", "identifier-v1"),
        ("vessel", "text-v1"),
        ("voyage", "identifier-v1"),
    ),
    "invoice": (
        ("amount", "numeric-v1"),
        ("consignee", "text-v1"),
        ("disch_port", "text-v1"),
        ("exporter_name", "text-v1"),
        ("good", "text-v1"),
        ("importer_name", "text-v1"),
        ("inv_date", "identifier-v1"),
        ("inv_no", "identifier-v1"),
        ("lc_no", "identifier-v1"),
        ("load_port", "text-v1"),
        ("qty", "numeric-v1"),
        ("total", "numeric-v1"),
        ("unit", "text-v1"),
        ("unit_price", "numeric-v1"),
        ("vessel", "text-v1"),
    ),
    "packing_list": (
        ("cartons", "numeric-v1"),
        ("cbm", "numeric-v1"),
        ("consignee", "text-v1"),
        ("exporter_name", "text-v1"),
        ("good", "text-v1"),
        ("gross_weight", "numeric-v1"),
        ("inv_no", "identifier-v1"),
        ("lc_no", "identifier-v1"),
        ("net_gross_weight", "text-v1"),
        ("net_weight", "numeric-v1"),
        ("qty", "numeric-v1"),
        ("unit", "text-v1"),
    ),
}
_EXPECTED_R1_TRAINING_ASSET_COUNT: Final = 5_532
_EXPECTED_R1_HELD_OUT_FAMILY_COUNT: Final = 78
_EXPECTED_R1_EXCLUDED_FAMILY_ASSET_COUNT: Final = 468
_EXPECTED_R1_TRAINING_STRATA: Final[tuple[tuple[str, SyntheticDocumentType, int], ...]] = (
    ("clean", "bill_of_lading", 922),
    ("clean", "invoice", 922),
    ("clean", "packing_list", 922),
    ("error", "bill_of_lading", 922),
    ("error", "invoice", 922),
    ("error", "packing_list", 922),
)
_DOCUMENT_SUFFIX: Final[Mapping[SyntheticDocumentType, str]] = {
    "bill_of_lading": "BL",
    "invoice": "INV",
    "packing_list": "PL",
}
_FIXTURE_DIRECTORY: Final[Mapping[str, str]] = {
    "clean": "clean_sets",
    "error": "error_sets",
}
_FIXTURE_PREFIX: Final[Mapping[str, str]] = {"clean": "C", "error": "E"}
_VALUE_TRANSFORMS: Final[
    tuple[
        Literal[
            "raw-v1",
            "single-numeric-token-v1",
            "single-alpha-token-v1",
            "join-slash-v1",
        ],
        ...,
    ]
] = (
    "raw-v1",
    "single-numeric-token-v1",
    "single-alpha-token-v1",
    "join-slash-v1",
)
_TRANSFORM_RANK: Final[Mapping[str, int]] = {
    transform: index for index, transform in enumerate(_VALUE_TRANSFORMS)
}
# These patterns intentionally match the runtime semantic adapter's bounded
# transform contract.  The trainer has no source-truth import; final adapter
# replay below proves there was no implementation drift before any artifact is
# returned for writing.
_NUMERIC_TOKEN = re.compile(r"(?<![A-Za-z0-9])(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?![A-Za-z0-9])")
_ALPHA_TOKEN = re.compile(r"[A-Za-z]+")


class SupervisedTemplateDirectSelectorTrainerError(ValueError):
    """Content-free failure safe for a local training summary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SourceAttestedSupervisedTemplateDirectSelectorTrainerRecipeV1(_WireModel):
    """Content-free recipe for the deterministic candidate-native learner."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_supervised_template_direct_selector_trainer_recipe"] = (
        SUPERVISED_TEMPLATE_DIRECT_SELECTOR_RECIPE_KIND
    )
    trainer_revision: Literal["source-attested-supervised-template-direct-selector-trainer-v1"] = (
        SUPERVISED_TEMPLATE_DIRECT_SELECTOR_TRAINER_REVISION
    )
    model_kind: Literal["native_pdf_semantic_model"] = (
        SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_KIND
    )
    model_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    model_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    candidate_evidence_revision: Literal["native-pdf-vector-grid-v1"] = _CANDIDATE_EVIDENCE_REVISION
    canonical_selector_order_revision: Literal["candidate-native-selector-order-v1"] = (
        _CANONICAL_SELECTOR_ORDER_REVISION
    )
    anchor_tolerance: float = Field(gt=0.0, le=0.05)
    document_types: list[SyntheticDocumentType] = Field(min_length=3, max_length=3)
    trainer_recipe_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _is_bound_and_scoped(
        self,
    ) -> "SourceAttestedSupervisedTemplateDirectSelectorTrainerRecipeV1":
        if tuple(self.document_types) != _REQUIRED_DOCUMENT_TYPES:
            raise ValueError("trainer recipe document types are invalid")
        if not math.isclose(self.anchor_tolerance, _ANCHOR_TOLERANCE, abs_tol=1e-12):
            raise ValueError("trainer recipe anchor tolerance is invalid")
        if self.trainer_recipe_sha256 != direct_selector_trainer_recipe_sha256(self):
            raise ValueError("trainer recipe SHA-256 mismatch")
        return self


@dataclass(frozen=True, slots=True)
class SourceAttestedSupervisedTemplateDirectSelectorTrainingResultV1:
    """In-memory, content-free result ready for immutable local writes."""

    recipe: SourceAttestedSupervisedTemplateDirectSelectorTrainerRecipeV1
    model: NativePdfSemanticModelArtifactV1
    model_artifact_bytes: bytes
    model_artifact_sha256: str
    model_training_receipt: SourceAttestedSupervisedTemplateModelTrainingReceiptV1
    training_asset_count: int
    field_label_count: int


@dataclass(frozen=True, slots=True)
class SourceAttestedSupervisedTemplateDirectSelectorTrainingPathsV1:
    """Hash-named local paths written only after successful full replay."""

    recipe_path: Path
    model_path: Path
    model_training_receipt_path: Path


@dataclass(frozen=True, slots=True)
class _EvidenceValue:
    selector_key: str
    selector: NativePdfSemanticSelectorV1
    selector_sort_key: tuple[int, int, int, int, int]
    source_key: tuple[str, int]
    text: str


@dataclass(frozen=True, slots=True)
class _AnchorObservation:
    document_type: SyntheticDocumentType
    anchors: tuple[tuple[int, float, float], ...]


_OptionKey = tuple[str, tuple[str, ...]]


def _recipe_payload(
    recipe: SourceAttestedSupervisedTemplateDirectSelectorTrainerRecipeV1 | Mapping[str, object],
) -> dict[str, object]:
    payload = (
        recipe.model_dump(mode="json", warnings=False)
        if isinstance(recipe, SourceAttestedSupervisedTemplateDirectSelectorTrainerRecipeV1)
        else dict(recipe)
    )
    payload.pop("trainer_recipe_sha256", None)
    return payload


def direct_selector_trainer_recipe_sha256(
    recipe: SourceAttestedSupervisedTemplateDirectSelectorTrainerRecipeV1 | Mapping[str, object],
) -> str:
    """Hash the complete content-free deterministic trainer recipe."""

    return sha256_bytes(canonical_json_bytes(_recipe_payload(recipe)))


def build_source_attested_supervised_template_direct_selector_trainer_recipe(
    *, model_id: str, model_revision: str
) -> SourceAttestedSupervisedTemplateDirectSelectorTrainerRecipeV1:
    """Build the fixed recipe that a later training receipt will bind."""

    payload: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": SUPERVISED_TEMPLATE_DIRECT_SELECTOR_RECIPE_KIND,
        "trainer_revision": SUPERVISED_TEMPLATE_DIRECT_SELECTOR_TRAINER_REVISION,
        "model_kind": SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_KIND,
        "model_id": model_id,
        "model_revision": model_revision,
        "candidate_evidence_revision": _CANDIDATE_EVIDENCE_REVISION,
        "canonical_selector_order_revision": _CANONICAL_SELECTOR_ORDER_REVISION,
        "anchor_tolerance": _ANCHOR_TOLERANCE,
        "document_types": list(_REQUIRED_DOCUMENT_TYPES),
    }
    payload["trainer_recipe_sha256"] = direct_selector_trainer_recipe_sha256(payload)
    try:
        return SourceAttestedSupervisedTemplateDirectSelectorTrainerRecipeV1.model_validate(payload)
    except ValidationError as error:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_recipe_invalid"
        ) from error


def _is_local_data_path(path: Path) -> bool:
    parts = [part.casefold() for part in path.parts]
    return any(parts[index : index + 2] == ["_local", "data"] for index in range(len(parts) - 1))


def _native_filesystem_path(path: Path) -> str:
    """Preserve safe canonical paths while supporting Windows long paths."""

    value = str(path)
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        return "\\\\?\\" + value
    return value


def _reject_symlink_path_components(path: Path, *, code: str) -> None:
    current = path
    while True:
        try:
            mode = os.lstat(_native_filesystem_path(current)).st_mode
        except OSError as error:
            raise SupervisedTemplateDirectSelectorTrainerError(code) from error
        if stat.S_ISLNK(mode):
            raise SupervisedTemplateDirectSelectorTrainerError(code)
        parent = current.parent
        if parent == current:
            return
        current = parent


def _local_input_directory(path: Path, *, code: str) -> Path:
    try:
        _reject_symlink_path_components(path, code=code)
        resolved = path.resolve(strict=True)
        if not _is_local_data_path(resolved) or not resolved.is_dir() or resolved.is_symlink():
            raise SupervisedTemplateDirectSelectorTrainerError(code)
    except SupervisedTemplateDirectSelectorTrainerError:
        raise
    except OSError as error:
        raise SupervisedTemplateDirectSelectorTrainerError(code) from error
    return resolved


def _local_output_directory(path: Path, *, code: str) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
        _reject_symlink_path_components(path, code=code)
        resolved = path.resolve(strict=True)
        if not _is_local_data_path(resolved) or not resolved.is_dir() or resolved.is_symlink():
            raise SupervisedTemplateDirectSelectorTrainerError(code)
    except SupervisedTemplateDirectSelectorTrainerError:
        raise
    except OSError as error:
        raise SupervisedTemplateDirectSelectorTrainerError(code) from error
    return resolved


def _read_regular_local_file(path: Path, *, maximum_bytes: int, code: str) -> bytes:
    try:
        _reject_symlink_path_components(path, code=code)
        resolved = path.resolve(strict=True)
        if not _is_local_data_path(resolved):
            raise SupervisedTemplateDirectSelectorTrainerError(code)
        mode = resolved.lstat().st_mode
        if not stat.S_ISREG(mode) or resolved.is_symlink():
            raise SupervisedTemplateDirectSelectorTrainerError(code)
        with resolved.open("rb") as stream:
            payload = stream.read(maximum_bytes + 1)
    except SupervisedTemplateDirectSelectorTrainerError:
        raise
    except OSError as error:
        raise SupervisedTemplateDirectSelectorTrainerError(code) from error
    if not payload or len(payload) > maximum_bytes:
        raise SupervisedTemplateDirectSelectorTrainerError(code)
    return payload


def _write_hash_named_immutable_bytes(
    *, directory: Path, digest: str, payload: bytes, code: str
) -> Path:
    destination = directory / f"{digest}.json"
    try:
        destination_name = _native_filesystem_path(destination)
        if os.path.lexists(destination_name):
            _reject_symlink_path_components(destination, code=code)
            if os.path.islink(destination_name):
                raise SupervisedTemplateDirectSelectorTrainerError(code)
            with open(destination_name, "rb") as stream:
                existing_payload = stream.read(len(payload) + 1)
            if existing_payload != payload:
                raise SupervisedTemplateDirectSelectorTrainerError(code)
            return destination
        # The required per-model SHA directory plus SHA filename can exceed
        # legacy Windows MAX_PATH under a test/worktree root.  Use the native
        # extended path form for the final create while retaining Path checks
        # and canonical naming above.
        with open(destination_name, "xb") as stream:
            stream.write(payload)
            stream.flush()
    except SupervisedTemplateDirectSelectorTrainerError:
        raise
    except OSError as error:
        raise SupervisedTemplateDirectSelectorTrainerError(code) from error
    return destination


def read_canonical_source_attested_supervised_template_training_split_for_selector_training(
    path: Path,
) -> SourceAttestedSupervisedTemplateTrainingSplitV1:
    """Load one local canonical split before it can authorize PDF reads."""

    raw = _read_regular_local_file(
        path,
        maximum_bytes=_MAX_INPUT_JSON_BYTES,
        code="supervised_template_selector_training_split_invalid",
    )
    try:
        split = SourceAttestedSupervisedTemplateTrainingSplitV1.model_validate_json(raw)
    except ValidationError as error:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_training_split_invalid"
        ) from error
    if raw != canonical_json_file_bytes(
        split.model_dump(mode="json", warnings=False)
    ) or path.name != (f"{split.training_split_receipt_sha256}.json"):
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_training_split_invalid"
        )
    return split


def read_canonical_source_attested_supervised_template_training_labels_for_selector_training(
    path: Path,
) -> SourceAttestedSupervisedTemplateTrainingLabelsV1:
    """Load compact hash labels without opening or accepting CSV/Gold input."""

    raw = _read_regular_local_file(
        path,
        maximum_bytes=_MAX_INPUT_JSON_BYTES,
        code="supervised_template_selector_training_labels_invalid",
    )
    try:
        labels = SourceAttestedSupervisedTemplateTrainingLabelsV1.model_validate_json(raw)
    except ValidationError as error:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_training_labels_invalid"
        ) from error
    if raw != canonical_json_file_bytes(
        labels.model_dump(mode="json", warnings=False)
    ) or path.name != (f"{labels.training_labels_receipt_sha256}.json"):
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_training_labels_invalid"
        )
    return labels


def read_canonical_source_attested_dataset_receipt_for_selector_training(
    path: Path,
) -> DatasetReceiptV1:
    """Read a canonical metadata receipt; this does not open any PDF."""

    raw = _read_regular_local_file(
        path,
        maximum_bytes=_MAX_INPUT_JSON_BYTES,
        code="supervised_template_selector_dataset_receipt_invalid",
    )
    try:
        receipt = DatasetReceiptV1.model_validate_json(raw)
    except ValidationError as error:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_dataset_receipt_invalid"
        ) from error
    if raw != canonical_json_file_bytes(
        receipt.model_dump(mode="json", warnings=False)
    ) or path.name != (f"{receipt_sha256(receipt)}.json"):
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_dataset_receipt_invalid"
        )
    return receipt


def read_canonical_source_attested_packet_for_selector_training(path: Path) -> EvaluationPacketV2:
    """Read a canonical development/sealed metadata packet only."""

    raw = _read_regular_local_file(
        path,
        maximum_bytes=_MAX_INPUT_JSON_BYTES,
        code="supervised_template_selector_packet_invalid",
    )
    try:
        packet = EvaluationPacketV2.model_validate_json(raw)
    except ValidationError as error:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_packet_invalid"
        ) from error
    if raw != canonical_json_file_bytes(
        packet.model_dump(mode="json", warnings=False)
    ) or path.name != (f"{packet_receipt_sha256(packet)}.json"):
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_packet_invalid"
        )
    return packet


def _validate_expected_r1_split(split: SourceAttestedSupervisedTemplateTrainingSplitV1) -> None:
    actual_strata = tuple(
        (item.fixture_kind, item.document_type, item.asset_count) for item in split.training_strata
    )
    if (
        split.training_asset_count != _EXPECTED_R1_TRAINING_ASSET_COUNT
        or split.held_out_family_count != _EXPECTED_R1_HELD_OUT_FAMILY_COUNT
        or split.excluded_family_asset_count != _EXPECTED_R1_EXCLUDED_FAMILY_ASSET_COUNT
        or actual_strata != _EXPECTED_R1_TRAINING_STRATA
    ):
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_r1_split_counts_invalid"
        )


def _expected_field_schema(
    document_type: SyntheticDocumentType,
) -> tuple[tuple[str, FieldNormalizer], ...]:
    expected = _REQUIRED_FIELDS.get(document_type)
    if expected is None:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_document_type_invalid"
        )
    return expected


def _validate_split_and_labels(
    *,
    receipt: DatasetReceiptV1,
    development_packet: EvaluationPacketV2,
    sealed_packet: EvaluationPacketV2,
    training_split: SourceAttestedSupervisedTemplateTrainingSplitV1,
    training_labels: SourceAttestedSupervisedTemplateTrainingLabelsV1,
) -> dict[str, SupervisedTemplateTrainingLabelRecordV1]:
    """Re-derive the partition before any training source asset is opened."""

    try:
        rebuilt_split = build_source_attested_supervised_template_training_split(
            receipt=receipt,
            development_packet=development_packet,
            sealed_packet=sealed_packet,
        )
    except SupervisedTemplateTrainingError as error:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_split_rederivation_invalid"
        ) from error
    if rebuilt_split != training_split:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_split_rederivation_invalid"
        )
    _validate_expected_r1_split(training_split)
    if (
        training_labels.dataset_id != training_split.dataset_id
        or training_labels.dataset_revision != training_split.dataset_revision
        or training_labels.dataset_receipt_sha256 != training_split.dataset_receipt_sha256
        or training_labels.training_split_receipt_sha256
        != training_split.training_split_receipt_sha256
        or training_labels.record_count != training_split.training_asset_count
    ):
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_label_split_binding_invalid"
        )
    labels_by_asset_id = {record.asset_id: record for record in training_labels.records}
    if len(labels_by_asset_id) != len(training_labels.records):
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_label_split_binding_invalid"
        )
    if set(labels_by_asset_id) != {asset.asset_id for asset in training_split.training_assets}:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_label_split_binding_invalid"
        )
    for asset in training_split.training_assets:
        record = labels_by_asset_id.get(asset.asset_id)
        if record is None or (
            record.source_sha256 != asset.source_sha256
            or record.fixture_kind != asset.fixture_kind
            or record.document_type != asset.document_type
            or record.family_set_number != asset.family_set_number
        ):
            raise SupervisedTemplateDirectSelectorTrainerError(
                "supervised_template_selector_label_split_binding_invalid"
            )
        expected_fields = _expected_field_schema(asset.document_type)
        actual_fields = tuple((label.field_name, label.normalizer) for label in record.field_labels)
        if actual_fields != expected_fields:
            raise SupervisedTemplateDirectSelectorTrainerError(
                "supervised_template_selector_label_schema_invalid"
            )
    return labels_by_asset_id


def _source_pdf_path(*, source_directory: Path, asset: SupervisedTemplateTrainingAssetV1) -> Path:
    try:
        fixture_directory = _FIXTURE_DIRECTORY[asset.fixture_kind]
        prefix = _FIXTURE_PREFIX[asset.fixture_kind]
        suffix = _DOCUMENT_SUFFIX[asset.document_type]
    except KeyError as error:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_training_asset_invalid"
        ) from error
    return (
        source_directory
        / fixture_directory
        / f"set_{asset.family_set_number}"
        / f"{prefix}{asset.family_set_number}_{suffix}.pdf"
    )


def _read_verified_training_pdf(
    *, source_directory: Path, asset: SupervisedTemplateTrainingAssetV1
) -> bytes:
    payload = _read_regular_local_file(
        _source_pdf_path(source_directory=source_directory, asset=asset),
        maximum_bytes=_MAX_TRAINING_PDF_BYTES,
        code="supervised_template_selector_training_pdf_invalid",
    )
    if hashlib.sha256(payload).hexdigest() != asset.source_sha256:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_training_pdf_sha256_mismatch"
        )
    return payload


def _selector_key(selector: NativePdfSemanticSelectorV1) -> str:
    return json.dumps(
        selector.model_dump(mode="json", warnings=False, exclude_none=True),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _selector_sort_key(selector: NativePdfSemanticSelectorV1) -> tuple[int, int, int, int, int]:
    if selector.source_kind == "span_reading_order":
        assert selector.reading_order is not None
        return (0, selector.page_number, selector.reading_order, 0, 0)
    if selector.source_kind == "table_cell_coordinate":
        assert (
            selector.table_index is not None
            and selector.row_index is not None
            and selector.column_index is not None
        )
        return (
            1,
            selector.page_number,
            selector.table_index,
            selector.row_index,
            selector.column_index,
        )
    # Learned span anchors are structural signatures, never field options.
    return (2, selector.page_number, 0, 0, 0)


def _candidate_evidence(extraction: NativePdfVectorGridExtraction) -> tuple[_EvidenceValue, ...]:
    evidence: list[_EvidenceValue] = []
    for index, span in enumerate(extraction.spans):
        selector = NativePdfSemanticSelectorV1(
            source_kind="span_reading_order",
            page_number=span.page_number,
            reading_order=span.reading_order,
        )
        evidence.append(
            _EvidenceValue(
                selector_key=_selector_key(selector),
                selector=selector,
                selector_sort_key=_selector_sort_key(selector),
                source_key=("span", index),
                text=span.text,
            )
        )
    for index, cell in enumerate(extraction.table_cells):
        selector = NativePdfSemanticSelectorV1(
            source_kind="table_cell_coordinate",
            page_number=cell.page_number,
            table_index=cell.table_index,
            row_index=cell.row_index,
            column_index=cell.column_index,
        )
        evidence.append(
            _EvidenceValue(
                selector_key=_selector_key(selector),
                selector=selector,
                selector_sort_key=_selector_sort_key(selector),
                source_key=("table_cell", index),
                text=cell.text,
            )
        )
    if not evidence:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_candidate_evidence_invalid"
        )
    return tuple(evidence)


def _single_token(*, text: str, pattern: re.Pattern[str]) -> str | None:
    tokens = pattern.findall(text)
    if len(tokens) != 1 or not tokens[0].strip():
        return None
    return tokens[0].strip()


def _normalized_value_sha256(*, value: str, normalizer: FieldNormalizer) -> str:
    normalized = normalize_field_value(value, normalizer)
    if not normalized:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_candidate_value_invalid"
        )
    return sha256_bytes(normalized.encode("utf-8"))


def _candidate_unary_options_for_label(
    *, evidence: Sequence[_EvidenceValue], label: SupervisedTemplateTrainingFieldLabelV1
) -> set[_OptionKey]:
    """Discover matching single-selector transforms from candidate evidence."""

    options: set[_OptionKey] = set()
    for item in evidence:
        if _normalized_value_sha256(value=item.text, normalizer=label.normalizer) == (
            label.normalized_value_sha256
        ):
            options.add(("raw-v1", (item.selector_key,)))
        numeric = _single_token(text=item.text, pattern=_NUMERIC_TOKEN)
        if (
            numeric is not None
            and _normalized_value_sha256(value=numeric, normalizer=label.normalizer)
            == label.normalized_value_sha256
        ):
            options.add(("single-numeric-token-v1", (item.selector_key,)))
        alpha = _single_token(text=item.text, pattern=_ALPHA_TOKEN)
        if (
            alpha is not None
            and _normalized_value_sha256(value=alpha, normalizer=label.normalizer)
            == label.normalized_value_sha256
        ):
            options.add(("single-alpha-token-v1", (item.selector_key,)))
    return options


def _candidate_join_options_for_label(
    *, evidence: Sequence[_EvidenceValue], label: SupervisedTemplateTrainingFieldLabelV1
) -> set[_OptionKey]:
    """Discover candidate-native two-selector joins only when unary learning failed."""

    options: set[_OptionKey] = set()
    for left in evidence:
        if not left.text.strip():
            continue
        for right in evidence:
            if left.source_key == right.source_key or not right.text.strip():
                continue
            joined = f"{left.text.strip()} / {right.text.strip()}"
            if _normalized_value_sha256(value=joined, normalizer=label.normalizer) == (
                label.normalized_value_sha256
            ):
                options.add(("join-slash-v1", (left.selector_key, right.selector_key)))
    return options


def _anchor_points(
    extraction: NativePdfVectorGridExtraction,
) -> tuple[tuple[int, float, float], ...]:
    # Keep duplicate positions.  A structural selector must resolve exactly
    # one span, so collapsing co-located candidate runs would create a false
    # positive during training that runtime rightly rejects.
    points: list[tuple[int, float, float]] = []
    for span in extraction.spans:
        if span.anchor_x is None or span.anchor_y is None:
            raise SupervisedTemplateDirectSelectorTrainerError(
                "supervised_template_selector_anchor_evidence_invalid"
            )
        points.append(
            (
                span.page_number,
                round(span.anchor_x, _ANCHOR_DECIMALS),
                round(span.anchor_y, _ANCHOR_DECIMALS),
            )
        )
    if not points:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_anchor_evidence_invalid"
        )
    return tuple(points)


def _anchor_selector(anchor: tuple[int, float, float]) -> NativePdfSemanticSelectorV1:
    return NativePdfSemanticSelectorV1(
        source_kind="span_anchor",
        page_number=anchor[0],
        anchor_x=anchor[1],
        anchor_y=anchor[2],
        anchor_tolerance=_ANCHOR_TOLERANCE,
    )


def _anchor_match_count(
    *, anchor: NativePdfSemanticSelectorV1, points: Sequence[tuple[int, float, float]]
) -> int:
    assert anchor.anchor_x is not None and anchor.anchor_y is not None
    return sum(
        1
        for page_number, anchor_x, anchor_y in points
        if page_number == anchor.page_number
        and abs(anchor_x - anchor.anchor_x) <= _ANCHOR_TOLERANCE
        and abs(anchor_y - anchor.anchor_y) <= _ANCHOR_TOLERANCE
    )


def _learn_unique_structural_anchor(
    *,
    document_type: SyntheticDocumentType,
    common_anchors: frozenset[tuple[int, float, float]],
    observations: Sequence[_AnchorObservation],
) -> NativePdfSemanticSelectorV1:
    for candidate in sorted(common_anchors):
        selector = _anchor_selector(candidate)
        if all(
            _anchor_match_count(anchor=selector, points=observation.anchors) == 1
            for observation in observations
            if observation.document_type == document_type
        ) and all(
            _anchor_match_count(anchor=selector, points=observation.anchors) == 0
            for observation in observations
            if observation.document_type != document_type
        ):
            return selector
    raise SupervisedTemplateDirectSelectorTrainerError(
        "supervised_template_selector_structural_anchor_unavailable"
    )


def _option_sort_key(
    *,
    option: _OptionKey,
    selector_sort_keys: Mapping[str, tuple[int, int, int, int, int]],
) -> tuple[object, ...]:
    transform, selector_keys = option
    try:
        transform_rank = _TRANSFORM_RANK[transform]
        selector_ranks = tuple(selector_sort_keys[key] for key in selector_keys)
    except KeyError as error:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_option_invalid"
        ) from error
    return (transform_rank, selector_ranks, selector_keys)


def _choose_stable_option(
    *,
    options: set[_OptionKey],
    selector_by_key: Mapping[str, NativePdfSemanticSelectorV1],
    selector_sort_keys: Mapping[str, tuple[int, int, int, int, int]],
) -> tuple[
    Literal["raw-v1", "single-numeric-token-v1", "single-alpha-token-v1", "join-slash-v1"],
    list[NativePdfSemanticSelectorV1],
]:
    if not options:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_field_selector_unstable"
        )
    selected = min(
        options,
        key=lambda option: _option_sort_key(option=option, selector_sort_keys=selector_sort_keys),
    )
    transform, selector_keys = selected
    if transform not in _VALUE_TRANSFORMS:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_option_invalid"
        )
    try:
        selectors = [selector_by_key[key] for key in selector_keys]
    except KeyError as error:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_option_invalid"
        ) from error
    return transform, selectors  # type: ignore[return-value]


def _validate_model_scope(
    *,
    model: NativePdfSemanticModelArtifactV1,
    learned_structural_shapes: Mapping[SyntheticDocumentType, tuple[int, int, int]],
) -> None:
    if (
        model.artifact_kind != SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_KIND
        or model.model_id != SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_ID
        or model.model_revision != SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_REVISION
        or model.inference_revision != "native-template-selector-v1"
        or tuple(document.document_type for document in model.documents) != _REQUIRED_DOCUMENT_TYPES
    ):
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_model_scope_invalid"
        )
    for document in model.documents:
        if document.document_type not in _REQUIRED_DOCUMENT_TYPES:
            raise SupervisedTemplateDirectSelectorTrainerError(
                "supervised_template_selector_model_scope_invalid"
            )
        document_type = cast(SyntheticDocumentType, document.document_type)
        expected_shape = learned_structural_shapes.get(document_type)
        if expected_shape is None:
            raise SupervisedTemplateDirectSelectorTrainerError(
                "supervised_template_selector_model_scope_invalid"
            )
        expected_page_count, expected_spans, expected_cells = expected_shape
        signature = document.structural_signature
        if (
            signature.page_count != expected_page_count
            or signature.minimum_span_count != expected_spans
            or signature.minimum_table_cell_count != expected_cells
            or len(signature.required_selectors) != 1
            or signature.required_selectors[0].source_kind != "span_anchor"
        ):
            raise SupervisedTemplateDirectSelectorTrainerError(
                "supervised_template_selector_model_scope_invalid"
            )
        actual_fields = tuple((field.field_name, field.normalizer) for field in document.fields)
        if actual_fields != _expected_field_schema(document_type):
            raise SupervisedTemplateDirectSelectorTrainerError(
                "supervised_template_selector_model_scope_invalid"
            )


def _assert_model_has_no_training_identifiers(
    *,
    model_bytes: bytes,
    training_split: SourceAttestedSupervisedTemplateTrainingSplitV1,
    training_labels: SourceAttestedSupervisedTemplateTrainingLabelsV1,
) -> None:
    forbidden = {asset.asset_id.encode("utf-8") for asset in training_split.training_assets} | {
        asset.source_sha256.encode("ascii") for asset in training_split.training_assets
    }
    forbidden |= {record.asset_id.encode("utf-8") for record in training_labels.records} | {
        record.source_sha256.encode("ascii") for record in training_labels.records
    }
    forbidden |= {
        label.normalized_value_sha256.encode("ascii")
        for record in training_labels.records
        for label in record.field_labels
    }
    if any(item in model_bytes for item in forbidden):
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_model_content_invalid"
        )


def _verify_model_against_all_training_inputs(
    *,
    model: HashBoundNativePdfSemanticModelV1,
    source_directory: Path,
    training_split: SourceAttestedSupervisedTemplateTrainingSplitV1,
    labels_by_asset_id: Mapping[str, SupervisedTemplateTrainingLabelRecordV1],
) -> None:
    """Replay byte-only runtime extraction over every permitted training PDF."""

    adapter = NativePdfSemanticAdapter(model=model)
    for asset in training_split.training_assets:
        record = labels_by_asset_id.get(asset.asset_id)
        if record is None:
            raise SupervisedTemplateDirectSelectorTrainerError(
                "supervised_template_selector_label_split_binding_invalid"
            )
        output = adapter.extract_pdf_bytes(
            _read_verified_training_pdf(source_directory=source_directory, asset=asset)
        )
        if output.document_type != asset.document_type:
            raise SupervisedTemplateDirectSelectorTrainerError(
                "supervised_template_selector_structural_coverage_invalid"
            )
        expected = {label.field_name: label for label in record.field_labels}
        actual = {field.field_name: field for field in output.fields}
        if set(actual) != set(expected) or len(actual) != len(expected):
            raise SupervisedTemplateDirectSelectorTrainerError(
                "supervised_template_selector_field_coverage_invalid"
            )
        for field_name, field in actual.items():
            label = expected.get(field_name)
            if (
                label is None
                or _normalized_value_sha256(value=field.value, normalizer=label.normalizer)
                != label.normalized_value_sha256
            ):
                raise SupervisedTemplateDirectSelectorTrainerError(
                    "supervised_template_selector_field_coverage_invalid"
                )


def train_source_attested_supervised_template_direct_selector_model(
    *,
    receipt: DatasetReceiptV1,
    development_packet: EvaluationPacketV2,
    sealed_packet: EvaluationPacketV2,
    training_split: SourceAttestedSupervisedTemplateTrainingSplitV1,
    training_labels: SourceAttestedSupervisedTemplateTrainingLabelsV1,
    source_directory: Path,
    model_id: str,
    model_revision: str,
    training_environment_inventory_sha256: str,
) -> SourceAttestedSupervisedTemplateDirectSelectorTrainingResultV1:
    """Learn and fully replay a direct-selector model before any file is written.

    Only after immutable split re-derivation, 100% per-field candidate-hash
    coverage, unique structural profile selection, and a second byte-only
    runtime replay succeeds does this return an in-memory model and receipt.
    """

    source_directory = _local_input_directory(
        source_directory, code="supervised_template_selector_source_directory_invalid"
    )
    if (
        model_id != SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_ID
        or model_revision != SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_REVISION
    ):
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_model_identity_invalid"
        )
    labels_by_asset_id = _validate_split_and_labels(
        receipt=receipt,
        development_packet=development_packet,
        sealed_packet=sealed_packet,
        training_split=training_split,
        training_labels=training_labels,
    )
    recipe = build_source_attested_supervised_template_direct_selector_trainer_recipe(
        model_id=model_id, model_revision=model_revision
    )
    unary_field_options: dict[tuple[SyntheticDocumentType, str], set[_OptionKey]] = {}
    selector_by_key: dict[str, NativePdfSemanticSelectorV1] = {}
    selector_sort_keys: dict[str, tuple[int, int, int, int, int]] = {}
    common_anchors: dict[SyntheticDocumentType, frozenset[tuple[int, float, float]]] = {}
    learned_structural_shapes: dict[SyntheticDocumentType, tuple[int, int, int]] = {}
    observations: list[_AnchorObservation] = []

    for asset in training_split.training_assets:
        record = labels_by_asset_id.get(asset.asset_id)
        if record is None:
            raise SupervisedTemplateDirectSelectorTrainerError(
                "supervised_template_selector_label_split_binding_invalid"
            )
        try:
            extraction = extract_native_pdf_vector_grid_from_pdf_bytes(
                pdf_bytes=_read_verified_training_pdf(
                    source_directory=source_directory, asset=asset
                )
            )
        except SupervisedTemplateDirectSelectorTrainerError:
            raise
        except Exception as error:
            raise SupervisedTemplateDirectSelectorTrainerError(
                "supervised_template_selector_candidate_evidence_invalid"
            ) from error
        observed_shape = (
            len(extraction.pages),
            len(extraction.spans),
            len(extraction.table_cells),
        )
        if any(value < 1 for value in observed_shape):
            raise SupervisedTemplateDirectSelectorTrainerError(
                "supervised_template_selector_structural_evidence_invalid"
            )
        prior_shape = learned_structural_shapes.setdefault(asset.document_type, observed_shape)
        if prior_shape != observed_shape:
            raise SupervisedTemplateDirectSelectorTrainerError(
                "supervised_template_selector_structural_evidence_not_invariant"
            )
        evidence = _candidate_evidence(extraction)
        for item in evidence:
            existing = selector_by_key.setdefault(item.selector_key, item.selector)
            if existing != item.selector:
                raise SupervisedTemplateDirectSelectorTrainerError(
                    "supervised_template_selector_candidate_evidence_invalid"
                )
            existing_sort_key = selector_sort_keys.setdefault(
                item.selector_key, item.selector_sort_key
            )
            if existing_sort_key != item.selector_sort_key:
                raise SupervisedTemplateDirectSelectorTrainerError(
                    "supervised_template_selector_candidate_evidence_invalid"
                )
        for label in record.field_labels:
            field_key = (asset.document_type, label.field_name)
            candidates = _candidate_unary_options_for_label(evidence=evidence, label=label)
            if field_key not in unary_field_options:
                unary_field_options[field_key] = candidates
            else:
                unary_field_options[field_key] &= candidates
        anchors = _anchor_points(extraction)
        observations.append(_AnchorObservation(document_type=asset.document_type, anchors=anchors))
        anchor_set = frozenset(anchors)
        common_anchors[asset.document_type] = (
            anchor_set
            if asset.document_type not in common_anchors
            else common_anchors[asset.document_type] & anchor_set
        )

    field_options = dict(unary_field_options)
    join_field_keys = {
        (document_type, field_name)
        for document_type in _REQUIRED_DOCUMENT_TYPES
        for field_name, _normalizer in _expected_field_schema(document_type)
        if not field_options.get((document_type, field_name), set())
    }
    if join_field_keys:
        join_field_options: dict[tuple[SyntheticDocumentType, str], set[_OptionKey]] = {}
        for asset in training_split.training_assets:
            record = labels_by_asset_id.get(asset.asset_id)
            if record is None:
                raise SupervisedTemplateDirectSelectorTrainerError(
                    "supervised_template_selector_label_split_binding_invalid"
                )
            required_labels = [
                label
                for label in record.field_labels
                if (asset.document_type, label.field_name) in join_field_keys
            ]
            if not required_labels:
                continue
            try:
                extraction = extract_native_pdf_vector_grid_from_pdf_bytes(
                    pdf_bytes=_read_verified_training_pdf(
                        source_directory=source_directory, asset=asset
                    )
                )
            except SupervisedTemplateDirectSelectorTrainerError:
                raise
            except Exception as error:
                raise SupervisedTemplateDirectSelectorTrainerError(
                    "supervised_template_selector_candidate_evidence_invalid"
                ) from error
            evidence = _candidate_evidence(extraction)
            for label in required_labels:
                field_key = (asset.document_type, label.field_name)
                candidates = _candidate_join_options_for_label(evidence=evidence, label=label)
                if not candidates:
                    raise SupervisedTemplateDirectSelectorTrainerError(
                        "supervised_template_selector_field_coverage_invalid"
                    )
                if field_key not in join_field_options:
                    join_field_options[field_key] = candidates
                else:
                    join_field_options[field_key] &= candidates
        for field_key in join_field_keys:
            candidates = join_field_options.get(field_key, set())
            if not candidates:
                raise SupervisedTemplateDirectSelectorTrainerError(
                    "supervised_template_selector_field_selector_unstable"
                )
            field_options[field_key] = candidates

    if set(learned_structural_shapes) != set(_REQUIRED_DOCUMENT_TYPES):
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_structural_evidence_invalid"
        )

    learned_anchors = {
        document_type: _learn_unique_structural_anchor(
            document_type=document_type,
            common_anchors=common_anchors.get(document_type, frozenset()),
            observations=observations,
        )
        for document_type in _REQUIRED_DOCUMENT_TYPES
    }
    documents: list[NativePdfSemanticDocumentModelV1] = []
    for document_type in _REQUIRED_DOCUMENT_TYPES:
        fields: list[NativePdfSemanticFieldModelV1] = []
        for field_name, normalizer in _expected_field_schema(document_type):
            transform, selectors = _choose_stable_option(
                options=field_options.get((document_type, field_name), set()),
                selector_by_key=selector_by_key,
                selector_sort_keys=selector_sort_keys,
            )
            fields.append(
                NativePdfSemanticFieldModelV1(
                    field_name=field_name,
                    normalizer=normalizer,
                    selectors=selectors,
                    value_transform=transform,
                )
            )
        page_count, minimum_spans, minimum_cells = learned_structural_shapes[document_type]
        documents.append(
            NativePdfSemanticDocumentModelV1(
                document_type=document_type,
                structural_signature=NativePdfSemanticStructuralSignatureV1(
                    page_count=page_count,
                    minimum_span_count=minimum_spans,
                    minimum_table_cell_count=minimum_cells,
                    required_selectors=[learned_anchors[document_type]],
                ),
                fields=fields,
            )
        )
    try:
        model = NativePdfSemanticModelArtifactV1(
            model_id=model_id,
            model_revision=model_revision,
            documents=documents,
        )
    except ValidationError as error:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_model_invalid"
        ) from error
    _validate_model_scope(model=model, learned_structural_shapes=learned_structural_shapes)
    model_bytes = canonical_json_file_bytes(
        model.model_dump(mode="json", warnings=False, exclude_none=True)
    )
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    try:
        bound_model = bind_native_pdf_semantic_model_artifact(
            artifact_bytes=model_bytes, expected_sha256=model_sha256
        )
    except Exception as error:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_model_binding_invalid"
        ) from error
    _assert_model_has_no_training_identifiers(
        model_bytes=model_bytes,
        training_split=training_split,
        training_labels=training_labels,
    )
    _verify_model_against_all_training_inputs(
        model=bound_model,
        source_directory=source_directory,
        training_split=training_split,
        labels_by_asset_id=labels_by_asset_id,
    )
    try:
        model_training_receipt = build_source_attested_supervised_template_model_training_receipt(
            training_split=training_split,
            training_labels=training_labels,
            model_kind=SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_KIND,
            model_id=SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_ID,
            model_revision=model_revision,
            model_artifact_sha256=model_sha256,
            training_recipe_sha256=recipe.trainer_recipe_sha256,
            training_environment_inventory_sha256=training_environment_inventory_sha256,
        )
    except (SupervisedTemplateTrainingError, ValidationError) as error:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_training_receipt_invalid"
        ) from error
    return SourceAttestedSupervisedTemplateDirectSelectorTrainingResultV1(
        recipe=recipe,
        model=model,
        model_artifact_bytes=model_bytes,
        model_artifact_sha256=model_sha256,
        model_training_receipt=model_training_receipt,
        training_asset_count=training_split.training_asset_count,
        field_label_count=training_labels.field_label_count,
    )


def write_source_attested_supervised_template_direct_selector_training_result(
    *,
    result: SourceAttestedSupervisedTemplateDirectSelectorTrainingResultV1,
    recipe_output_directory: Path,
    model_root_directory: Path,
    model_training_receipt_output_directory: Path,
) -> SourceAttestedSupervisedTemplateDirectSelectorTrainingPathsV1:
    """Write local hash-named outputs only after all full-coverage checks pass.

    Models are intentionally isolated at
    ``models/native-pdf-semantic/<model-sha>/<model-sha>.json``.  A later
    runtime inventory can therefore require an exact one-file directory for a
    configured model rather than accepting a mutable shared model collection.
    """

    recipe_directory = _local_output_directory(
        recipe_output_directory, code="supervised_template_selector_recipe_directory_invalid"
    )
    model_root = _local_output_directory(
        model_root_directory, code="supervised_template_selector_model_directory_invalid"
    )
    model_directory = _local_output_directory(
        model_root / result.model_artifact_sha256,
        code="supervised_template_selector_model_directory_invalid",
    )
    receipt_directory = _local_output_directory(
        model_training_receipt_output_directory,
        code="supervised_template_selector_training_receipt_directory_invalid",
    )
    recipe_bytes = canonical_json_file_bytes(result.recipe.model_dump(mode="json", warnings=False))
    recipe_path = _write_hash_named_immutable_bytes(
        directory=recipe_directory,
        digest=result.recipe.trainer_recipe_sha256,
        payload=recipe_bytes,
        code="supervised_template_selector_recipe_immutable_conflict",
    )
    model_path = _write_hash_named_immutable_bytes(
        directory=model_directory,
        digest=result.model_artifact_sha256,
        payload=result.model_artifact_bytes,
        code="supervised_template_selector_model_immutable_conflict",
    )
    try:
        model_receipt_path = write_source_attested_supervised_template_model_training_receipt(
            receipt=result.model_training_receipt,
            output_directory=receipt_directory,
        )
    except SupervisedTemplateTrainingError as error:
        raise SupervisedTemplateDirectSelectorTrainerError(
            "supervised_template_selector_training_receipt_immutable_conflict"
        ) from error
    return SourceAttestedSupervisedTemplateDirectSelectorTrainingPathsV1(
        recipe_path=recipe_path,
        model_path=model_path,
        model_training_receipt_path=model_receipt_path,
    )


__all__ = [
    "SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_KIND",
    "SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_ID",
    "SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_REVISION",
    "SUPERVISED_TEMPLATE_DIRECT_SELECTOR_RECIPE_KIND",
    "SUPERVISED_TEMPLATE_DIRECT_SELECTOR_TRAINER_REVISION",
    "SupervisedTemplateDirectSelectorTrainerError",
    "SourceAttestedSupervisedTemplateDirectSelectorTrainerRecipeV1",
    "SourceAttestedSupervisedTemplateDirectSelectorTrainingPathsV1",
    "SourceAttestedSupervisedTemplateDirectSelectorTrainingResultV1",
    "build_source_attested_supervised_template_direct_selector_trainer_recipe",
    "direct_selector_trainer_recipe_sha256",
    "read_canonical_source_attested_dataset_receipt_for_selector_training",
    "read_canonical_source_attested_packet_for_selector_training",
    "read_canonical_source_attested_supervised_template_training_labels_for_selector_training",
    "read_canonical_source_attested_supervised_template_training_split_for_selector_training",
    "train_source_attested_supervised_template_direct_selector_model",
    "write_source_attested_supervised_template_direct_selector_training_result",
]
