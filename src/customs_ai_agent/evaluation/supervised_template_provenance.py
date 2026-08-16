"""Fail-closed runtime provenance for the supervised synthetic-PDF r1 model.

This module is deliberately independent from the source-layout parser and
training compiler.  It opens only compact, canonical metadata artifacts and a
single learned semantic-model artifact; it never opens a PDF, CSV, Gold record,
or source-layout profile.  Its job is to make a runtime model binding prove all
of the following before the model can inspect selected PDF bytes:

* the fixed r1 dataset receipt and fixed development/sealed packets were used;
* the persisted training split is exactly the family-safe split re-derived from
  those metadata artifacts, rather than merely self-consistent;
* compact labels cover exactly that split with the public r1 field schema; and
* the opaque training receipt and semantic model are mutually hash-bound to
  the one allowed supervised-template profile.

The resulting compatibility claim remains narrow: it concerns the fixed,
born-digital synthetic r1 template.  It is not a general OCR or scanned-document
claim.
"""

from __future__ import annotations

import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Mapping, Sequence, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from customs_ai_agent.evaluation.document_gold import (
    DatasetAssetReceiptV1,
    DatasetReceiptV1,
    receipt_sha256,
)
from customs_ai_agent.evaluation.document_ocr_quality import (
    EvaluationPacketV2,
    FieldNormalizer,
    SyntheticDocumentType,
    canonical_json_bytes,
    canonical_json_file_bytes,
    sha256_bytes,
)
from customs_ai_agent.evaluation.native_pdf_semantic_model import (
    HashBoundNativePdfSemanticModelV1,
    NativePdfSemanticModelError,
    load_hash_bound_native_pdf_semantic_model,
)
from customs_ai_agent.evaluation.supervised_template_contract import (
    R1_SUPERVISED_TEMPLATE_REQUIRED_FIELDS,
    SUPERVISED_TEMPLATE_MODEL_ID,
    SUPERVISED_TEMPLATE_MODEL_KIND,
    SUPERVISED_TEMPLATE_MODEL_REVISION,
    SupervisedTemplateContractError,
    validate_r1_supervised_template_model,
)


_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_SHA256 = r"^[a-f0-9]{64}$"
_FAMILY_SET_NUMBER = r"^[0-9]{4}$"
_SYNTHETIC_PDF_RECEIPT_PATH = re.compile(
    r"^(?P<fixture>clean_sets|error_sets)/set_(?P<set_number>\d{4})/"
    r"(?P<record>[CE]\d{4})_(?P<document>BL|INV|PL)\.pdf$"
)
_MAX_DATASET_RECEIPT_BYTES: Final = 4 * 1024 * 1024
_MAX_PACKET_BYTES: Final = 128 * 1024
_MAX_TRAINING_SPLIT_BYTES: Final = 4 * 1024 * 1024
_MAX_TRAINING_LABELS_BYTES: Final = 32 * 1024 * 1024
_MAX_TRAINING_RECEIPT_BYTES: Final = 128 * 1024


R1_SUPERVISED_TEMPLATE_DATASET_RECEIPT_SHA256: Final = (
    "aa3c828f4fdf4afc6a02171b271c71f314eefb363e6676c65625bae803e1b39d"
)
R1_SUPERVISED_TEMPLATE_DEVELOPMENT_PACKET_RECEIPT_SHA256: Final = (
    "5b8a804b022e97df0e58a60c4fadfc9d3637c900f368273b8c1b6ff347c3f7e2"
)
R1_SUPERVISED_TEMPLATE_SEALED_PACKET_RECEIPT_SHA256: Final = (
    "174d7fad0040e89ee93d367f31694183049ea649a28515cde7a59458261aa271"
)
R1_SUPERVISED_TEMPLATE_TRAINING_SPLIT_RECEIPT_SHA256: Final = (
    "4ff18546cd1de20596e40e6620f52f2487ebae35bbf54fef02092e69406db9b2"
)
R1_SUPERVISED_TEMPLATE_TRAINING_LABELS_RECEIPT_SHA256: Final = (
    "7110d6c418083840b6c2fe4efc03744fd1e9228b4111b4cca5821f17f72eb7e7"
)


class SupervisedTemplateProvenanceError(ValueError):
    """A content-free failure that is safe for an operator summary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _TrainingAssetV1(_WireModel):
    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    source_sha256: str = Field(pattern=_SHA256)
    fixture_kind: Literal["clean", "error"]
    document_type: SyntheticDocumentType
    family_set_number: str = Field(pattern=_FAMILY_SET_NUMBER)


class _TrainingStratumV1(_WireModel):
    fixture_kind: Literal["clean", "error"]
    document_type: SyntheticDocumentType
    asset_count: int = Field(ge=0, le=100_000)


class _TrainingSplitV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_supervised_template_training_split"] = (
        "source_attested_supervised_template_training_split"
    )
    split_revision: Literal["source-attested-family-safe-training-split-v1"] = (
        "source-attested-family-safe-training-split-v1"
    )
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_receipt_sha256: str = Field(pattern=_SHA256)
    development_packet_receipt_sha256: str = Field(pattern=_SHA256)
    sealed_packet_receipt_sha256: str = Field(pattern=_SHA256)
    held_out_family_set_numbers: list[str] = Field(min_length=1, max_length=100_000)
    held_out_family_count: int = Field(ge=1, le=100_000)
    excluded_family_asset_count: int = Field(ge=1, le=100_000)
    training_assets: list[_TrainingAssetV1] = Field(min_length=1, max_length=100_000)
    training_asset_count: int = Field(ge=1, le=100_000)
    training_asset_list_sha256: str = Field(pattern=_SHA256)
    training_strata: list[_TrainingStratumV1] = Field(min_length=1, max_length=16)
    training_split_receipt_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _is_canonical_and_complete(self) -> "_TrainingSplitV1":
        held_out = self.held_out_family_set_numbers
        if held_out != sorted(set(held_out)) or len(held_out) != self.held_out_family_count:
            raise ValueError("training split family set numbers must be sorted and unique")
        assets = self.training_assets
        if [asset.asset_id for asset in assets] != sorted(asset.asset_id for asset in assets):
            raise ValueError("training split assets must be sorted")
        if len({asset.asset_id for asset in assets}) != len(assets):
            raise ValueError("training split asset IDs must be unique")
        if any(asset.family_set_number in held_out for asset in assets):
            raise ValueError("training split contains a held-out family")
        if len(assets) != self.training_asset_count:
            raise ValueError("training split asset count mismatch")
        if self.training_asset_list_sha256 != _training_asset_list_sha256(assets):
            raise ValueError("training split asset list hash mismatch")
        strata = self.training_strata
        ordered_strata = sorted(strata, key=lambda item: (item.fixture_kind, item.document_type))
        if strata != ordered_strata or len(
            {(item.fixture_kind, item.document_type) for item in strata}
        ) != len(strata):
            raise ValueError("training split strata must be sorted and unique")
        expected_counts: dict[tuple[str, str], int] = {}
        for asset in assets:
            key = (asset.fixture_kind, asset.document_type)
            expected_counts[key] = expected_counts.get(key, 0) + 1
        observed_counts = {
            (item.fixture_kind, item.document_type): item.asset_count for item in strata
        }
        if observed_counts != expected_counts:
            raise ValueError("training split stratum counts mismatch")
        if self.training_split_receipt_sha256 != _training_split_sha256(self):
            raise ValueError("training split receipt hash mismatch")
        return self


class _TrainingFieldLabelV1(_WireModel):
    field_name: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    normalizer: FieldNormalizer
    normalized_value_sha256: str = Field(pattern=_SHA256)


class _TrainingLabelRecordV1(_WireModel):
    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    source_sha256: str = Field(pattern=_SHA256)
    fixture_kind: Literal["clean", "error"]
    document_type: SyntheticDocumentType
    family_set_number: str = Field(pattern=_FAMILY_SET_NUMBER)
    content_stream_sha256: str = Field(pattern=_SHA256)
    field_labels: list[_TrainingFieldLabelV1] = Field(min_length=1, max_length=1_000)
    training_label_record_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _is_canonical_and_bound(self) -> "_TrainingLabelRecordV1":
        names = [label.field_name for label in self.field_labels]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("training label field names must be sorted and unique")
        if self.training_label_record_sha256 != _training_label_record_sha256(self):
            raise ValueError("training label record hash mismatch")
        return self


class _TrainingLabelsV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_supervised_template_training_labels"] = (
        "source_attested_supervised_template_training_labels"
    )
    label_kind: Literal["pdf-content-stream-semantic-slot-label-v1"] = (
        "pdf-content-stream-semantic-slot-label-v1"
    )
    label_compiler_revision: Literal["source-attested-family-safe-training-label-compiler-v1"] = (
        "source-attested-family-safe-training-label-compiler-v1"
    )
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_receipt_sha256: str = Field(pattern=_SHA256)
    training_split_receipt_sha256: str = Field(pattern=_SHA256)
    record_count: int = Field(ge=1, le=100_000)
    field_label_count: int = Field(ge=1, le=1_000_000)
    records: list[_TrainingLabelRecordV1] = Field(min_length=1, max_length=100_000)
    training_labels_receipt_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _is_canonical_and_bound(self) -> "_TrainingLabelsV1":
        records = self.records
        if [record.asset_id for record in records] != sorted(record.asset_id for record in records):
            raise ValueError("training label records must be sorted")
        if len({record.asset_id for record in records}) != len(records):
            raise ValueError("training label record asset IDs must be unique")
        if self.record_count != len(records):
            raise ValueError("training label record count mismatch")
        if self.field_label_count != sum(len(record.field_labels) for record in records):
            raise ValueError("training label field count mismatch")
        if self.training_labels_receipt_sha256 != _training_labels_sha256(self):
            raise ValueError("training labels receipt hash mismatch")
        return self


class _ModelTrainingReceiptV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_supervised_template_model_training_receipt"] = (
        "source_attested_supervised_template_model_training_receipt"
    )
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_receipt_sha256: str = Field(pattern=_SHA256)
    training_split_receipt_sha256: str = Field(pattern=_SHA256)
    training_labels_receipt_sha256: str = Field(pattern=_SHA256)
    model_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    model_kind: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    model_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    model_artifact_sha256: str = Field(pattern=_SHA256)
    training_recipe_sha256: str = Field(pattern=_SHA256)
    training_environment_inventory_sha256: str = Field(pattern=_SHA256)
    model_training_receipt_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _is_canonical_and_bound(self) -> "_ModelTrainingReceiptV1":
        if self.model_training_receipt_sha256 != _model_training_receipt_sha256(self):
            raise ValueError("model training receipt hash mismatch")
        return self


def _without_digest(value: BaseModel | Mapping[str, object], digest_key: str) -> dict[str, object]:
    payload = (
        value.model_dump(mode="json", warnings=False)
        if isinstance(value, BaseModel)
        else dict(value)
    )
    payload.pop(digest_key, None)
    return payload


def _training_asset_list_sha256(assets: Sequence[_TrainingAssetV1]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            [
                {"asset_id": asset.asset_id, "source_sha256": asset.source_sha256}
                for asset in assets
            ]
        )
    )


def _training_split_sha256(value: _TrainingSplitV1 | Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(_without_digest(value, "training_split_receipt_sha256")))


def _training_label_record_sha256(
    value: _TrainingLabelRecordV1 | Mapping[str, object],
) -> str:
    return sha256_bytes(canonical_json_bytes(_without_digest(value, "training_label_record_sha256")))


def _training_labels_sha256(value: _TrainingLabelsV1 | Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(_without_digest(value, "training_labels_receipt_sha256")))


def _model_training_receipt_sha256(
    value: _ModelTrainingReceiptV1 | Mapping[str, object],
) -> str:
    return sha256_bytes(canonical_json_bytes(_without_digest(value, "model_training_receipt_sha256")))


@dataclass(frozen=True, slots=True)
class _R1ProvenancePolicy:
    """Internal immutable policy; only the fixed shipped instance is public."""

    dataset_id: str
    dataset_revision: str
    dataset_receipt_sha256: str
    development_packet_receipt_sha256: str
    sealed_packet_receipt_sha256: str
    training_split_receipt_sha256: str
    training_labels_receipt_sha256: str
    total_pdf_asset_count: int
    clean_pdf_asset_count: int
    error_pdf_asset_count: int
    sets_per_fixture_kind: int
    development_item_count: int
    sealed_item_count: int
    held_out_family_count: int
    excluded_family_asset_count: int
    training_asset_count: int
    training_stratum_count: int
    training_assets_per_stratum: int
    field_label_count: int


_EXPECTED_R1_POLICY: Final = _R1ProvenancePolicy(
    dataset_id="synthetic-customs-pdf-2000",
    dataset_revision="20260811-r1",
    dataset_receipt_sha256=R1_SUPERVISED_TEMPLATE_DATASET_RECEIPT_SHA256,
    development_packet_receipt_sha256=(
        R1_SUPERVISED_TEMPLATE_DEVELOPMENT_PACKET_RECEIPT_SHA256
    ),
    sealed_packet_receipt_sha256=R1_SUPERVISED_TEMPLATE_SEALED_PACKET_RECEIPT_SHA256,
    training_split_receipt_sha256=R1_SUPERVISED_TEMPLATE_TRAINING_SPLIT_RECEIPT_SHA256,
    training_labels_receipt_sha256=R1_SUPERVISED_TEMPLATE_TRAINING_LABELS_RECEIPT_SHA256,
    total_pdf_asset_count=6_000,
    clean_pdf_asset_count=3_000,
    error_pdf_asset_count=3_000,
    sets_per_fixture_kind=1_000,
    development_item_count=60,
    sealed_item_count=20,
    held_out_family_count=78,
    excluded_family_asset_count=468,
    training_asset_count=5_532,
    training_stratum_count=6,
    training_assets_per_stratum=922,
    field_label_count=73_760,
)


@dataclass(frozen=True, slots=True)
class _ReceiptAssetMetadata:
    asset: DatasetAssetReceiptV1
    fixture_kind: Literal["clean", "error"]
    document_type: SyntheticDocumentType
    family_set_number: str


@dataclass(frozen=True, slots=True)
class R1SupervisedTemplateProvenanceBinding:
    """Validated, content-free provenance plus the byte-only model binding."""

    model: HashBoundNativePdfSemanticModelV1
    dataset_receipt_sha256: str
    development_packet_receipt_sha256: str
    sealed_packet_receipt_sha256: str
    training_split_receipt_sha256: str
    training_labels_receipt_sha256: str
    model_training_receipt_sha256: str
    document_preview_root: Path
    model_artifact_path: Path
    expected_model_artifact_sha256: str

    def verified_model(self) -> HashBoundNativePdfSemanticModelV1:
        """Recheck the in-memory artifact before it is supplied to inference."""

        try:
            artifact = self.model.verified_artifact()
            validate_r1_supervised_template_model(artifact)
        except (NativePdfSemanticModelError, SupervisedTemplateContractError) as error:
            raise SupervisedTemplateProvenanceError(
                "supervised_template_runtime_model_binding_invalid"
            ) from error
        return self.model

    def verify_runtime_state(self) -> None:
        """Re-open and fully revalidate the hash-bound local provenance chain."""

        refreshed = _load_from_document_preview_root(
            document_preview_root=self.document_preview_root,
            model_artifact_path=self.model_artifact_path,
            expected_model_training_receipt_sha256=self.model_training_receipt_sha256,
            expected_model_artifact_sha256=self.expected_model_artifact_sha256,
        )
        if (
            refreshed.dataset_receipt_sha256 != self.dataset_receipt_sha256
            or refreshed.development_packet_receipt_sha256
            != self.development_packet_receipt_sha256
            or refreshed.sealed_packet_receipt_sha256 != self.sealed_packet_receipt_sha256
            or refreshed.training_split_receipt_sha256 != self.training_split_receipt_sha256
            or refreshed.training_labels_receipt_sha256 != self.training_labels_receipt_sha256
            or refreshed.model_training_receipt_sha256 != self.model_training_receipt_sha256
            or refreshed.model.artifact_sha256 != self.model.artifact_sha256
        ):
            raise SupervisedTemplateProvenanceError(
                "supervised_template_runtime_state_binding_invalid"
            )


def _require_sha256(value: str, *, code: str) -> str:
    if not isinstance(value, str) or re.fullmatch(_SHA256, value) is None:
        raise SupervisedTemplateProvenanceError(code)
    return value


def _require_absolute_path(path: Path, *, code: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise SupervisedTemplateProvenanceError(code)
    return path


def _reject_symlink_path_components(path: Path, *, code: str) -> None:
    """Reject an artifact reached through any symlink or non-existent parent."""

    current = path
    while True:
        try:
            mode = current.lstat().st_mode
        except OSError as error:
            raise SupervisedTemplateProvenanceError(code) from error
        if stat.S_ISLNK(mode):
            raise SupervisedTemplateProvenanceError(code)
        parent = current.parent
        if parent == current:
            return
        current = parent


def _read_regular_artifact_bytes(*, path: Path, max_bytes: int, code: str) -> bytes:
    path = _require_absolute_path(path, code=code)
    _reject_symlink_path_components(path, code=code)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise SupervisedTemplateProvenanceError(code) from error
    if not stat.S_ISREG(mode):
        raise SupervisedTemplateProvenanceError(code)
    try:
        with path.open("rb") as stream:
            value = stream.read(max_bytes + 1)
    except OSError as error:
        raise SupervisedTemplateProvenanceError(code) from error
    if not 1 <= len(value) <= max_bytes:
        raise SupervisedTemplateProvenanceError(code)
    return value


def _require_hash_named_path(*, path: Path, digest: str, code: str) -> None:
    _require_sha256(digest, code=code)
    if path.name != f"{digest}.json":
        raise SupervisedTemplateProvenanceError(code)


def _read_canonical_dataset_receipt(
    *, path: Path, expected_sha256: str
) -> DatasetReceiptV1:
    expected_sha256 = _require_sha256(expected_sha256, code="supervised_template_dataset_hash_invalid")
    raw = _read_regular_artifact_bytes(
        path=path,
        max_bytes=_MAX_DATASET_RECEIPT_BYTES,
        code="supervised_template_dataset_artifact_invalid",
    )
    try:
        receipt = DatasetReceiptV1.model_validate_json(raw)
    except ValidationError as error:
        raise SupervisedTemplateProvenanceError(
            "supervised_template_dataset_artifact_invalid"
        ) from error
    canonical = canonical_json_file_bytes(receipt.model_dump(mode="json", warnings=False))
    if raw != canonical or receipt_sha256(receipt) != expected_sha256:
        raise SupervisedTemplateProvenanceError("supervised_template_dataset_artifact_invalid")
    _require_hash_named_path(
        path=path,
        digest=expected_sha256,
        code="supervised_template_dataset_filename_invalid",
    )
    return receipt


def _read_canonical_packet(*, path: Path, expected_sha256: str) -> EvaluationPacketV2:
    expected_sha256 = _require_sha256(expected_sha256, code="supervised_template_packet_hash_invalid")
    raw = _read_regular_artifact_bytes(
        path=path,
        max_bytes=_MAX_PACKET_BYTES,
        code="supervised_template_packet_artifact_invalid",
    )
    try:
        packet = EvaluationPacketV2.model_validate_json(raw)
    except ValidationError as error:
        raise SupervisedTemplateProvenanceError(
            "supervised_template_packet_artifact_invalid"
        ) from error
    canonical = canonical_json_file_bytes(packet.model_dump(mode="json", warnings=False))
    if raw != canonical or packet.packet_receipt_sha256 != expected_sha256:
        raise SupervisedTemplateProvenanceError("supervised_template_packet_artifact_invalid")
    _require_hash_named_path(
        path=path,
        digest=expected_sha256,
        code="supervised_template_packet_filename_invalid",
    )
    return packet


def _read_canonical_training_split(*, path: Path, expected_sha256: str) -> _TrainingSplitV1:
    expected_sha256 = _require_sha256(expected_sha256, code="supervised_template_split_hash_invalid")
    raw = _read_regular_artifact_bytes(
        path=path,
        max_bytes=_MAX_TRAINING_SPLIT_BYTES,
        code="supervised_template_split_artifact_invalid",
    )
    try:
        split = _TrainingSplitV1.model_validate_json(raw)
    except ValidationError as error:
        raise SupervisedTemplateProvenanceError(
            "supervised_template_split_artifact_invalid"
        ) from error
    canonical = canonical_json_file_bytes(split.model_dump(mode="json", warnings=False))
    if raw != canonical or split.training_split_receipt_sha256 != expected_sha256:
        raise SupervisedTemplateProvenanceError("supervised_template_split_artifact_invalid")
    _require_hash_named_path(
        path=path,
        digest=expected_sha256,
        code="supervised_template_split_filename_invalid",
    )
    return split


def _read_canonical_training_labels(*, path: Path, expected_sha256: str) -> _TrainingLabelsV1:
    expected_sha256 = _require_sha256(expected_sha256, code="supervised_template_labels_hash_invalid")
    raw = _read_regular_artifact_bytes(
        path=path,
        max_bytes=_MAX_TRAINING_LABELS_BYTES,
        code="supervised_template_labels_artifact_invalid",
    )
    try:
        labels = _TrainingLabelsV1.model_validate_json(raw)
    except ValidationError as error:
        raise SupervisedTemplateProvenanceError(
            "supervised_template_labels_artifact_invalid"
        ) from error
    canonical = canonical_json_file_bytes(labels.model_dump(mode="json", warnings=False))
    if raw != canonical or labels.training_labels_receipt_sha256 != expected_sha256:
        raise SupervisedTemplateProvenanceError("supervised_template_labels_artifact_invalid")
    _require_hash_named_path(
        path=path,
        digest=expected_sha256,
        code="supervised_template_labels_filename_invalid",
    )
    return labels


def _read_canonical_model_training_receipt(
    *, path: Path, expected_sha256: str
) -> _ModelTrainingReceiptV1:
    expected_sha256 = _require_sha256(
        expected_sha256, code="supervised_template_model_receipt_hash_invalid"
    )
    raw = _read_regular_artifact_bytes(
        path=path,
        max_bytes=_MAX_TRAINING_RECEIPT_BYTES,
        code="supervised_template_model_receipt_artifact_invalid",
    )
    try:
        receipt = _ModelTrainingReceiptV1.model_validate_json(raw)
    except ValidationError as error:
        raise SupervisedTemplateProvenanceError(
            "supervised_template_model_receipt_artifact_invalid"
        ) from error
    canonical = canonical_json_file_bytes(receipt.model_dump(mode="json", warnings=False))
    if raw != canonical or receipt.model_training_receipt_sha256 != expected_sha256:
        raise SupervisedTemplateProvenanceError(
            "supervised_template_model_receipt_artifact_invalid"
        )
    _require_hash_named_path(
        path=path,
        digest=expected_sha256,
        code="supervised_template_model_receipt_filename_invalid",
    )
    return receipt


def _metadata_for_asset(asset: DatasetAssetReceiptV1) -> _ReceiptAssetMetadata:
    match = _SYNTHETIC_PDF_RECEIPT_PATH.fullmatch(asset.relative_path)
    if (
        match is None
        or asset.media_type != "application/pdf"
        or asset.source_role != "preview_candidate"
    ):
        raise SupervisedTemplateProvenanceError("supervised_template_dataset_inventory_invalid")
    fixture_directory = match.group("fixture")
    record = match.group("record")
    set_number = match.group("set_number")
    fixture_kind: Literal["clean", "error"] = (
        "clean" if fixture_directory == "clean_sets" else "error"
    )
    expected_record_prefix = "C" if fixture_kind == "clean" else "E"
    if record != f"{expected_record_prefix}{set_number}":
        raise SupervisedTemplateProvenanceError("supervised_template_dataset_inventory_invalid")
    document_type = {
        "BL": "bill_of_lading",
        "INV": "invoice",
        "PL": "packing_list",
    }[match.group("document")]
    return _ReceiptAssetMetadata(
        asset=asset,
        fixture_kind=fixture_kind,
        document_type=cast(SyntheticDocumentType, document_type),
        family_set_number=set_number,
    )


def _validate_r1_dataset_inventory(
    *, receipt: DatasetReceiptV1, policy: _R1ProvenancePolicy
) -> dict[str, _ReceiptAssetMetadata]:
    if (
        receipt.dataset_id != policy.dataset_id
        or receipt.dataset_revision != policy.dataset_revision
        or receipt_sha256(receipt) != policy.dataset_receipt_sha256
        or len(receipt.assets) != policy.total_pdf_asset_count
    ):
        raise SupervisedTemplateProvenanceError("supervised_template_dataset_scope_invalid")
    metadata_by_asset_id: dict[str, _ReceiptAssetMetadata] = {}
    inventory_keys: set[tuple[str, str, str]] = set()
    fixture_counts: dict[str, int] = {"clean": 0, "error": 0}
    for asset in receipt.assets:
        metadata = _metadata_for_asset(asset)
        if asset.asset_id in metadata_by_asset_id:
            raise SupervisedTemplateProvenanceError("supervised_template_dataset_inventory_invalid")
        key = (metadata.fixture_kind, metadata.family_set_number, metadata.document_type)
        if key in inventory_keys:
            raise SupervisedTemplateProvenanceError("supervised_template_dataset_inventory_invalid")
        metadata_by_asset_id[asset.asset_id] = metadata
        inventory_keys.add(key)
        fixture_counts[metadata.fixture_kind] += 1
    if (
        fixture_counts["clean"] != policy.clean_pdf_asset_count
        or fixture_counts["error"] != policy.error_pdf_asset_count
    ):
        raise SupervisedTemplateProvenanceError("supervised_template_dataset_scope_invalid")
    expected_document_types: tuple[SyntheticDocumentType, ...] = (
        "bill_of_lading",
        "invoice",
        "packing_list",
    )
    for fixture_kind in ("clean", "error"):
        for number in range(1, policy.sets_per_fixture_kind + 1):
            family_set_number = f"{number:04d}"
            for document_type in expected_document_types:
                if (fixture_kind, family_set_number, document_type) not in inventory_keys:
                    raise SupervisedTemplateProvenanceError(
                        "supervised_template_dataset_inventory_invalid"
                    )
    return metadata_by_asset_id


def _validate_packet_metadata(
    *,
    packet: EvaluationPacketV2,
    split: Literal["development", "sealed"],
    expected_packet_sha256: str,
    expected_item_count: int,
    receipt: DatasetReceiptV1,
    metadata_by_asset_id: Mapping[str, _ReceiptAssetMetadata],
) -> tuple[_ReceiptAssetMetadata, ...]:
    if (
        packet.split != split
        or packet.dataset_id != receipt.dataset_id
        or packet.dataset_revision != receipt.dataset_revision
        or packet.receipt_sha256 != receipt_sha256(receipt)
        or packet.packet_receipt_sha256 != expected_packet_sha256
        or len(packet.items) != expected_item_count
    ):
        raise SupervisedTemplateProvenanceError("supervised_template_packet_scope_invalid")
    values: list[_ReceiptAssetMetadata] = []
    for item in packet.items:
        metadata = metadata_by_asset_id.get(item.asset_id)
        expected_set_id = (
            ("C" if metadata is not None and metadata.fixture_kind == "clean" else "E")
            + metadata.family_set_number
            if metadata is not None
            else ""
        )
        if (
            metadata is None
            or item.source_sha256 != metadata.asset.sha256
            or item.media_type != "application/pdf"
            or item.fixture_kind != metadata.fixture_kind
            or item.document_type != metadata.document_type
            or item.set_id != expected_set_id
        ):
            raise SupervisedTemplateProvenanceError("supervised_template_packet_scope_invalid")
        values.append(metadata)
    if len({metadata.asset.asset_id for metadata in values}) != len(values):
        raise SupervisedTemplateProvenanceError("supervised_template_packet_scope_invalid")
    return tuple(values)


def _build_rederived_training_split(
    *,
    receipt: DatasetReceiptV1,
    development_packet: EvaluationPacketV2,
    sealed_packet: EvaluationPacketV2,
    metadata_by_asset_id: Mapping[str, _ReceiptAssetMetadata],
    policy: _R1ProvenancePolicy,
) -> _TrainingSplitV1:
    development_metadata = _validate_packet_metadata(
        packet=development_packet,
        split="development",
        expected_packet_sha256=policy.development_packet_receipt_sha256,
        expected_item_count=policy.development_item_count,
        receipt=receipt,
        metadata_by_asset_id=metadata_by_asset_id,
    )
    sealed_metadata = _validate_packet_metadata(
        packet=sealed_packet,
        split="sealed",
        expected_packet_sha256=policy.sealed_packet_receipt_sha256,
        expected_item_count=policy.sealed_item_count,
        receipt=receipt,
        metadata_by_asset_id=metadata_by_asset_id,
    )
    development_asset_ids = {metadata.asset.asset_id for metadata in development_metadata}
    sealed_asset_ids = {metadata.asset.asset_id for metadata in sealed_metadata}
    if development_asset_ids & sealed_asset_ids:
        raise SupervisedTemplateProvenanceError("supervised_template_packet_overlap")
    held_out_families = {
        metadata.family_set_number for metadata in (*development_metadata, *sealed_metadata)
    }
    if len(held_out_families) != policy.held_out_family_count:
        raise SupervisedTemplateProvenanceError("supervised_template_family_exclusion_invalid")
    training_assets: list[_TrainingAssetV1] = []
    excluded_count = 0
    for metadata in metadata_by_asset_id.values():
        if metadata.family_set_number in held_out_families:
            excluded_count += 1
            continue
        training_assets.append(
            _TrainingAssetV1(
                asset_id=metadata.asset.asset_id,
                source_sha256=metadata.asset.sha256,
                fixture_kind=metadata.fixture_kind,
                document_type=metadata.document_type,
                family_set_number=metadata.family_set_number,
            )
        )
    training_assets.sort(key=lambda asset: asset.asset_id)
    if (
        excluded_count != policy.excluded_family_asset_count
        or len(training_assets) != policy.training_asset_count
        or any(asset.asset_id in development_asset_ids | sealed_asset_ids for asset in training_assets)
    ):
        raise SupervisedTemplateProvenanceError("supervised_template_family_exclusion_invalid")
    counts: dict[tuple[str, str], int] = {}
    for asset in training_assets:
        key = (asset.fixture_kind, asset.document_type)
        counts[key] = counts.get(key, 0) + 1
    if (
        len(counts) != policy.training_stratum_count
        or set(counts.values()) != {policy.training_assets_per_stratum}
    ):
        raise SupervisedTemplateProvenanceError("supervised_template_training_strata_invalid")
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "source_attested_supervised_template_training_split",
        "split_revision": "source-attested-family-safe-training-split-v1",
        "dataset_id": receipt.dataset_id,
        "dataset_revision": receipt.dataset_revision,
        "dataset_receipt_sha256": receipt_sha256(receipt),
        "development_packet_receipt_sha256": development_packet.packet_receipt_sha256,
        "sealed_packet_receipt_sha256": sealed_packet.packet_receipt_sha256,
        "held_out_family_set_numbers": sorted(held_out_families),
        "held_out_family_count": len(held_out_families),
        "excluded_family_asset_count": excluded_count,
        "training_assets": [asset.model_dump(mode="json") for asset in training_assets],
        "training_asset_count": len(training_assets),
        "training_asset_list_sha256": _training_asset_list_sha256(training_assets),
        "training_strata": [
            {
                "fixture_kind": fixture_kind,
                "document_type": document_type,
                "asset_count": count,
            }
            for (fixture_kind, document_type), count in sorted(counts.items())
        ],
    }
    payload["training_split_receipt_sha256"] = _training_split_sha256(payload)
    try:
        return _TrainingSplitV1.model_validate(payload)
    except ValidationError as error:
        raise SupervisedTemplateProvenanceError(
            "supervised_template_rederived_split_invalid"
        ) from error


def _validate_persisted_split(
    *, persisted: _TrainingSplitV1, rederived: _TrainingSplitV1, policy: _R1ProvenancePolicy
) -> None:
    if (
        persisted.training_split_receipt_sha256 != policy.training_split_receipt_sha256
        or rederived.training_split_receipt_sha256 != policy.training_split_receipt_sha256
        or persisted != rederived
    ):
        raise SupervisedTemplateProvenanceError("supervised_template_split_binding_invalid")


def _validate_training_label_schema(record: _TrainingLabelRecordV1) -> None:
    expected = R1_SUPERVISED_TEMPLATE_REQUIRED_FIELDS.get(record.document_type)
    observed = tuple((field.field_name, field.normalizer) for field in record.field_labels)
    if expected is None or observed != expected:
        raise SupervisedTemplateProvenanceError("supervised_template_labels_schema_invalid")


def _validate_persisted_labels(
    *, labels: _TrainingLabelsV1, split: _TrainingSplitV1, policy: _R1ProvenancePolicy
) -> None:
    if (
        labels.dataset_id != split.dataset_id
        or labels.dataset_revision != split.dataset_revision
        or labels.dataset_receipt_sha256 != split.dataset_receipt_sha256
        or labels.training_split_receipt_sha256 != split.training_split_receipt_sha256
        or labels.training_labels_receipt_sha256 != policy.training_labels_receipt_sha256
        or labels.record_count != policy.training_asset_count
        or labels.field_label_count != policy.field_label_count
    ):
        raise SupervisedTemplateProvenanceError("supervised_template_labels_binding_invalid")
    assets_by_id = {asset.asset_id: asset for asset in split.training_assets}
    if [record.asset_id for record in labels.records] != list(sorted(assets_by_id)):
        raise SupervisedTemplateProvenanceError("supervised_template_labels_binding_invalid")
    for record in labels.records:
        asset = assets_by_id.get(record.asset_id)
        if (
            asset is None
            or record.source_sha256 != asset.source_sha256
            or record.fixture_kind != asset.fixture_kind
            or record.document_type != asset.document_type
            or record.family_set_number != asset.family_set_number
        ):
            raise SupervisedTemplateProvenanceError("supervised_template_labels_binding_invalid")
        _validate_training_label_schema(record)


def _validate_model_training_receipt(
    *,
    receipt: _ModelTrainingReceiptV1,
    split: _TrainingSplitV1,
    labels: _TrainingLabelsV1,
    expected_model_artifact_sha256: str,
) -> None:
    if (
        receipt.dataset_id != split.dataset_id
        or receipt.dataset_revision != split.dataset_revision
        or receipt.dataset_receipt_sha256 != split.dataset_receipt_sha256
        or receipt.training_split_receipt_sha256 != split.training_split_receipt_sha256
        or receipt.training_labels_receipt_sha256 != labels.training_labels_receipt_sha256
        or receipt.model_id != SUPERVISED_TEMPLATE_MODEL_ID
        or receipt.model_kind != SUPERVISED_TEMPLATE_MODEL_KIND
        or receipt.model_revision != SUPERVISED_TEMPLATE_MODEL_REVISION
        or receipt.model_artifact_sha256 != expected_model_artifact_sha256
    ):
        raise SupervisedTemplateProvenanceError("supervised_template_model_receipt_binding_invalid")


def _load_with_policy(
    *,
    dataset_receipt_path: Path,
    development_packet_path: Path,
    sealed_packet_path: Path,
    training_split_path: Path,
    training_labels_path: Path,
    model_training_receipt_path: Path,
    model_artifact_path: Path,
    expected_model_training_receipt_sha256: str,
    expected_model_artifact_sha256: str,
    policy: _R1ProvenancePolicy,
) -> R1SupervisedTemplateProvenanceBinding:
    expected_model_training_receipt_sha256 = _require_sha256(
        expected_model_training_receipt_sha256,
        code="supervised_template_model_receipt_hash_invalid",
    )
    expected_model_artifact_sha256 = _require_sha256(
        expected_model_artifact_sha256,
        code="supervised_template_model_hash_invalid",
    )
    receipt = _read_canonical_dataset_receipt(
        path=dataset_receipt_path,
        expected_sha256=policy.dataset_receipt_sha256,
    )
    development_packet = _read_canonical_packet(
        path=development_packet_path,
        expected_sha256=policy.development_packet_receipt_sha256,
    )
    sealed_packet = _read_canonical_packet(
        path=sealed_packet_path,
        expected_sha256=policy.sealed_packet_receipt_sha256,
    )
    persisted_split = _read_canonical_training_split(
        path=training_split_path,
        expected_sha256=policy.training_split_receipt_sha256,
    )
    persisted_labels = _read_canonical_training_labels(
        path=training_labels_path,
        expected_sha256=policy.training_labels_receipt_sha256,
    )
    model_training_receipt = _read_canonical_model_training_receipt(
        path=model_training_receipt_path,
        expected_sha256=expected_model_training_receipt_sha256,
    )
    metadata_by_asset_id = _validate_r1_dataset_inventory(receipt=receipt, policy=policy)
    rederived_split = _build_rederived_training_split(
        receipt=receipt,
        development_packet=development_packet,
        sealed_packet=sealed_packet,
        metadata_by_asset_id=metadata_by_asset_id,
        policy=policy,
    )
    _validate_persisted_split(
        persisted=persisted_split,
        rederived=rederived_split,
        policy=policy,
    )
    _validate_persisted_labels(
        labels=persisted_labels,
        split=persisted_split,
        policy=policy,
    )
    _validate_model_training_receipt(
        receipt=model_training_receipt,
        split=persisted_split,
        labels=persisted_labels,
        expected_model_artifact_sha256=expected_model_artifact_sha256,
    )
    _require_absolute_path(model_artifact_path, code="supervised_template_model_path_invalid")
    _require_hash_named_path(
        path=model_artifact_path,
        digest=expected_model_artifact_sha256,
        code="supervised_template_model_filename_invalid",
    )
    try:
        model = load_hash_bound_native_pdf_semantic_model(
            artifact_path=model_artifact_path,
            expected_sha256=expected_model_artifact_sha256,
        )
        validate_r1_supervised_template_model(model.artifact)
    except (NativePdfSemanticModelError, SupervisedTemplateContractError) as error:
        raise SupervisedTemplateProvenanceError("supervised_template_model_artifact_invalid") from error
    if model.artifact_sha256 != model_training_receipt.model_artifact_sha256:
        raise SupervisedTemplateProvenanceError("supervised_template_model_receipt_binding_invalid")
    return R1SupervisedTemplateProvenanceBinding(
        model=model,
        dataset_receipt_sha256=policy.dataset_receipt_sha256,
        development_packet_receipt_sha256=policy.development_packet_receipt_sha256,
        sealed_packet_receipt_sha256=policy.sealed_packet_receipt_sha256,
        training_split_receipt_sha256=policy.training_split_receipt_sha256,
        training_labels_receipt_sha256=policy.training_labels_receipt_sha256,
        model_training_receipt_sha256=model_training_receipt.model_training_receipt_sha256,
        document_preview_root=dataset_receipt_path.parent.parent,
        model_artifact_path=model_artifact_path,
        expected_model_artifact_sha256=expected_model_artifact_sha256,
    )


def _document_preview_artifact_paths(
    *,
    document_preview_root: Path,
    model_artifact_path: Path,
    expected_model_training_receipt_sha256: str,
    expected_model_artifact_sha256: str,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    root = _require_absolute_path(
        document_preview_root, code="supervised_template_document_preview_root_invalid"
    )
    _reject_symlink_path_components(
        root, code="supervised_template_document_preview_root_invalid"
    )
    try:
        root_mode = root.lstat().st_mode
    except OSError as error:
        raise SupervisedTemplateProvenanceError(
            "supervised_template_document_preview_root_invalid"
        ) from error
    if not stat.S_ISDIR(root_mode):
        raise SupervisedTemplateProvenanceError("supervised_template_document_preview_root_invalid")
    expected_model_artifact_sha256 = _require_sha256(
        expected_model_artifact_sha256, code="supervised_template_model_hash_invalid"
    )
    expected_model_path = (
        root
        / "models"
        / "native-pdf-semantic"
        / expected_model_artifact_sha256
        / f"{expected_model_artifact_sha256}.json"
    )
    if model_artifact_path != expected_model_path:
        raise SupervisedTemplateProvenanceError("supervised_template_model_path_invalid")
    return (
        root
        / "source-attested-dataset-receipts"
        / f"{R1_SUPERVISED_TEMPLATE_DATASET_RECEIPT_SHA256}.json",
        root
        / "source-attested-packets"
        / f"{R1_SUPERVISED_TEMPLATE_DEVELOPMENT_PACKET_RECEIPT_SHA256}.json",
        root
        / "source-attested-packets"
        / f"{R1_SUPERVISED_TEMPLATE_SEALED_PACKET_RECEIPT_SHA256}.json",
        root
        / "supervised-template-training-splits"
        / f"{R1_SUPERVISED_TEMPLATE_TRAINING_SPLIT_RECEIPT_SHA256}.json",
        root
        / "supervised-template-training-labels"
        / f"{R1_SUPERVISED_TEMPLATE_TRAINING_LABELS_RECEIPT_SHA256}.json",
        root
        / "supervised-template-training-receipts"
        / f"{expected_model_training_receipt_sha256}.json",
    )


def _load_from_document_preview_root(
    *,
    document_preview_root: Path,
    model_artifact_path: Path,
    expected_model_training_receipt_sha256: str,
    expected_model_artifact_sha256: str,
) -> R1SupervisedTemplateProvenanceBinding:
    expected_model_training_receipt_sha256 = _require_sha256(
        expected_model_training_receipt_sha256,
        code="supervised_template_model_receipt_hash_invalid",
    )
    (
        dataset_receipt_path,
        development_packet_path,
        sealed_packet_path,
        training_split_path,
        training_labels_path,
        model_training_receipt_path,
    ) = _document_preview_artifact_paths(
        document_preview_root=document_preview_root,
        model_artifact_path=model_artifact_path,
        expected_model_training_receipt_sha256=expected_model_training_receipt_sha256,
        expected_model_artifact_sha256=expected_model_artifact_sha256,
    )
    return _load_with_policy(
        dataset_receipt_path=dataset_receipt_path,
        development_packet_path=development_packet_path,
        sealed_packet_path=sealed_packet_path,
        training_split_path=training_split_path,
        training_labels_path=training_labels_path,
        model_training_receipt_path=model_training_receipt_path,
        model_artifact_path=model_artifact_path,
        expected_model_training_receipt_sha256=expected_model_training_receipt_sha256,
        expected_model_artifact_sha256=expected_model_artifact_sha256,
        policy=_EXPECTED_R1_POLICY,
    )


def load_r1_supervised_template_provenance(
    *,
    document_preview_root: Path,
    model_artifact_path: Path,
    expected_model_training_receipt_sha256: str,
    expected_model_artifact_sha256: str,
) -> R1SupervisedTemplateProvenanceBinding:
    """Load only the exact r1 supervised-template provenance chain.

    ``expected_model_*`` values belong to the runtime configuration.  Passing
    them independently of the files prevents path substitution from selecting
    a different, internally self-consistent local model chain.
    """

    return _load_from_document_preview_root(
        document_preview_root=document_preview_root,
        model_artifact_path=model_artifact_path,
        expected_model_training_receipt_sha256=expected_model_training_receipt_sha256,
        expected_model_artifact_sha256=expected_model_artifact_sha256,
    )


__all__ = [
    "R1SupervisedTemplateProvenanceBinding",
    "R1_SUPERVISED_TEMPLATE_DATASET_RECEIPT_SHA256",
    "R1_SUPERVISED_TEMPLATE_DEVELOPMENT_PACKET_RECEIPT_SHA256",
    "R1_SUPERVISED_TEMPLATE_SEALED_PACKET_RECEIPT_SHA256",
    "R1_SUPERVISED_TEMPLATE_TRAINING_LABELS_RECEIPT_SHA256",
    "R1_SUPERVISED_TEMPLATE_TRAINING_SPLIT_RECEIPT_SHA256",
    "SupervisedTemplateProvenanceError",
    "load_r1_supervised_template_provenance",
]
