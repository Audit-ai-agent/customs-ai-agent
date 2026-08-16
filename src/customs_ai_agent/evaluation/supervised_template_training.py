"""Local-only family-safe labels for supervised synthetic-PDF template work.

This is deliberately a *training preparation* boundary.  It does not expose a
runtime candidate, does not alter the OCR evaluator, and must not be imported
by a document-preview runtime.  It creates two immutable local artifacts:

* a receipt for the only allowed training partition; and
* compact semantic-slot labels derived directly from the final PDF content
  stream.

The records contain only semantic field names, field normalizers, and hashes
of normalized values, plus training-record identity and split binding.  They
never persist PDF text or field values, source-run ordinals, source-parser
transforms, coordinates, or a source-template profile.  A future trainer must
discover its own candidate-native selectors and transforms from immutable
training PDFs, and prove that it never trains on development, sealed, or
clean/error counterpart-family assets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from customs_ai_agent.evaluation.document_gold import DatasetAssetReceiptV1, DatasetReceiptV1
from customs_ai_agent.evaluation.document_ocr_quality import (
    EvaluationPacketV2,
    FieldNormalizer,
    SyntheticDocumentType,
    canonical_json_bytes,
    canonical_json_file_bytes,
    normalize_field_value,
    receipt_sha256,
    sha256_bytes,
)

# The private source-truth helpers below are intentionally confined to this
# training-only module.  They must not be imported by a runtime adapter or a
# candidate runner: this module reads only family-safe training PDFs and emits
# no raw values, source text, or profile coordinates for runtime use.
from customs_ai_agent.evaluation.source_attested_synthetic_pdf_truth import (
    EXPECTED_SOURCE_ATTESTED_SYNTHETIC_DATASET_SCOPE,
    SOURCE_ATTESTED_REQUIRED_FIELDS,
    SourceAttestedSyntheticDatasetScopeV1,
    SourceAttestedSyntheticPdfTruthError,
    _fields_from_profile,
    _local_input_directory,
    _local_output_directory,
    _parse_pdf_content_stream_v1,
    _read_verified_source_pdf,
    _require_profile,
    _write_hash_named_immutable,
    validate_source_attested_synthetic_dataset_scope,
    validate_source_attested_synthetic_packet_scope,
)


_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_SHA256 = r"^[a-f0-9]{64}$"
_FAMILY_SET_NUMBER = r"^[0-9]{4}$"

SUPERVISED_TEMPLATE_TRAINING_SPLIT_REVISION: Literal[
    "source-attested-family-safe-training-split-v1"
] = "source-attested-family-safe-training-split-v1"
SUPERVISED_TEMPLATE_TRAINING_LABEL_COMPILER_REVISION: Literal[
    "source-attested-family-safe-training-label-compiler-v1"
] = "source-attested-family-safe-training-label-compiler-v1"
SUPERVISED_TEMPLATE_TRAINING_LABEL_KIND: Literal["pdf-content-stream-semantic-slot-label-v1"] = (
    "pdf-content-stream-semantic-slot-label-v1"
)


class SupervisedTemplateTrainingError(ValueError):
    """Content-free failure safe for the local training-preparation summary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SupervisedTemplateTrainingAssetV1(_WireModel):
    """One exact source asset eligible for training after family exclusion."""

    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    source_sha256: str = Field(pattern=_SHA256)
    fixture_kind: Literal["clean", "error"]
    document_type: SyntheticDocumentType
    family_set_number: str = Field(pattern=_FAMILY_SET_NUMBER)


class SupervisedTemplateTrainingStratumV1(_WireModel):
    fixture_kind: Literal["clean", "error"]
    document_type: SyntheticDocumentType
    asset_count: int = Field(ge=0, le=100_000)


class SourceAttestedSupervisedTemplateTrainingSplitV1(_WireModel):
    """Immutable family-disjoint training partition for one source receipt."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_supervised_template_training_split"] = (
        "source_attested_supervised_template_training_split"
    )
    split_revision: Literal["source-attested-family-safe-training-split-v1"] = (
        SUPERVISED_TEMPLATE_TRAINING_SPLIT_REVISION
    )
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_receipt_sha256: str = Field(pattern=_SHA256)
    development_packet_receipt_sha256: str = Field(pattern=_SHA256)
    sealed_packet_receipt_sha256: str = Field(pattern=_SHA256)
    held_out_family_set_numbers: list[str] = Field(min_length=1, max_length=100_000)
    held_out_family_count: int = Field(ge=1, le=100_000)
    excluded_family_asset_count: int = Field(ge=1, le=100_000)
    training_assets: list[SupervisedTemplateTrainingAssetV1] = Field(
        min_length=1, max_length=100_000
    )
    training_asset_count: int = Field(ge=1, le=100_000)
    training_asset_list_sha256: str = Field(pattern=_SHA256)
    training_strata: list[SupervisedTemplateTrainingStratumV1] = Field(min_length=1, max_length=16)
    training_split_receipt_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _is_canonical_and_bound(self) -> "SourceAttestedSupervisedTemplateTrainingSplitV1":
        held_out = self.held_out_family_set_numbers
        if held_out != sorted(set(held_out)) or len(held_out) != self.held_out_family_count:
            raise ValueError("training split family set numbers must be sorted and unique")
        assets = self.training_assets
        if [asset.asset_id for asset in assets] != sorted(asset.asset_id for asset in assets):
            raise ValueError("training split assets must be sorted")
        if len({asset.asset_id for asset in assets}) != len(assets):
            raise ValueError("training split asset IDs must be unique")
        if any(asset.family_set_number in held_out for asset in assets):
            raise ValueError("training split includes a held-out family")
        if len(assets) != self.training_asset_count:
            raise ValueError("training split asset count mismatch")
        if self.training_asset_list_sha256 != training_asset_list_sha256(assets):
            raise ValueError("training split asset list SHA-256 mismatch")
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
        if self.training_split_receipt_sha256 != training_split_receipt_sha256(self):
            raise ValueError("training split receipt SHA-256 mismatch")
        return self


class SupervisedTemplateTrainingFieldLabelV1(_WireModel):
    """One content-free semantic target without a source layout selector."""

    field_name: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    normalizer: FieldNormalizer
    normalized_value_sha256: str = Field(pattern=_SHA256)


class SupervisedTemplateTrainingLabelRecordV1(_WireModel):
    """Labels compiled from one final PDF without retaining its text."""

    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    source_sha256: str = Field(pattern=_SHA256)
    fixture_kind: Literal["clean", "error"]
    document_type: SyntheticDocumentType
    family_set_number: str = Field(pattern=_FAMILY_SET_NUMBER)
    content_stream_sha256: str = Field(pattern=_SHA256)
    field_labels: list[SupervisedTemplateTrainingFieldLabelV1] = Field(
        min_length=1, max_length=1_000
    )
    training_label_record_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _is_complete_and_bound(self) -> "SupervisedTemplateTrainingLabelRecordV1":
        labels = self.field_labels
        names = [label.field_name for label in labels]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("training label field names must be sorted and unique")
        expected = SOURCE_ATTESTED_REQUIRED_FIELDS.get(self.document_type)
        if expected is None or set(names) != {name for name, _normalizer in expected}:
            raise ValueError("training label field schema mismatch")
        normalizers = {label.field_name: label.normalizer for label in labels}
        if normalizers != dict(expected):
            raise ValueError("training label normalizer mismatch")
        if self.training_label_record_sha256 != training_label_record_sha256(self):
            raise ValueError("training label record SHA-256 mismatch")
        return self


class SourceAttestedSupervisedTemplateTrainingLabelsV1(_WireModel):
    """Immutable compact labels for the family-safe training split only."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_supervised_template_training_labels"] = (
        "source_attested_supervised_template_training_labels"
    )
    label_kind: Literal["pdf-content-stream-semantic-slot-label-v1"] = (
        SUPERVISED_TEMPLATE_TRAINING_LABEL_KIND
    )
    label_compiler_revision: Literal["source-attested-family-safe-training-label-compiler-v1"] = (
        SUPERVISED_TEMPLATE_TRAINING_LABEL_COMPILER_REVISION
    )
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_receipt_sha256: str = Field(pattern=_SHA256)
    training_split_receipt_sha256: str = Field(pattern=_SHA256)
    record_count: int = Field(ge=1, le=100_000)
    field_label_count: int = Field(ge=1, le=1_000_000)
    records: list[SupervisedTemplateTrainingLabelRecordV1] = Field(min_length=1, max_length=100_000)
    training_labels_receipt_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _is_complete_and_bound(self) -> "SourceAttestedSupervisedTemplateTrainingLabelsV1":
        records = self.records
        if [record.asset_id for record in records] != sorted(record.asset_id for record in records):
            raise ValueError("training label records must be sorted")
        if len({record.asset_id for record in records}) != len(records):
            raise ValueError("training label record asset IDs must be unique")
        if self.record_count != len(records):
            raise ValueError("training label record count mismatch")
        if self.field_label_count != sum(len(record.field_labels) for record in records):
            raise ValueError("training label field count mismatch")
        if self.training_labels_receipt_sha256 != training_labels_receipt_sha256(self):
            raise ValueError("training labels receipt SHA-256 mismatch")
        return self


class SourceAttestedSupervisedTemplateModelTrainingReceiptV1(_WireModel):
    """Content-free binding a future model artifact to its safe training inputs.

    This type intentionally does not open a model file or provide a runtime
    loader.  A later trainer computes the model SHA-256 and writes this receipt;
    a later runtime configuration can bind to this receipt without receiving
    source values, asset IDs, source SHA values, or parser/profile coordinates.
    """

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_supervised_template_model_training_receipt"] = (
        "source_attested_supervised_template_model_training_receipt"
    )
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_receipt_sha256: str = Field(pattern=_SHA256)
    training_split_receipt_sha256: str = Field(pattern=_SHA256)
    training_labels_receipt_sha256: str = Field(pattern=_SHA256)
    model_kind: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    model_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    model_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    model_artifact_sha256: str = Field(pattern=_SHA256)
    training_recipe_sha256: str = Field(pattern=_SHA256)
    training_environment_inventory_sha256: str = Field(pattern=_SHA256)
    model_training_receipt_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _is_bound(self) -> "SourceAttestedSupervisedTemplateModelTrainingReceiptV1":
        if self.model_training_receipt_sha256 != model_training_receipt_sha256(self):
            raise ValueError("model training receipt SHA-256 mismatch")
        return self


def _split_payload(
    split: SourceAttestedSupervisedTemplateTrainingSplitV1 | Mapping[str, object],
) -> dict[str, object]:
    payload = (
        split.model_dump(mode="json", warnings=False)
        if isinstance(split, SourceAttestedSupervisedTemplateTrainingSplitV1)
        else dict(split)
    )
    payload.pop("training_split_receipt_sha256", None)
    return payload


def training_split_receipt_sha256(
    split: SourceAttestedSupervisedTemplateTrainingSplitV1 | Mapping[str, object],
) -> str:
    return sha256_bytes(canonical_json_bytes(_split_payload(split)))


def training_asset_list_sha256(
    assets: Sequence[SupervisedTemplateTrainingAssetV1],
) -> str:
    """Bind the exact asset IDs and source SHA values used for training."""

    return sha256_bytes(
        canonical_json_bytes(
            [{"asset_id": asset.asset_id, "source_sha256": asset.source_sha256} for asset in assets]
        )
    )


def _training_label_record_payload(
    record: SupervisedTemplateTrainingLabelRecordV1 | Mapping[str, object],
) -> dict[str, object]:
    payload = (
        record.model_dump(mode="json", warnings=False)
        if isinstance(record, SupervisedTemplateTrainingLabelRecordV1)
        else dict(record)
    )
    payload.pop("training_label_record_sha256", None)
    return payload


def training_label_record_sha256(
    record: SupervisedTemplateTrainingLabelRecordV1 | Mapping[str, object],
) -> str:
    return sha256_bytes(canonical_json_bytes(_training_label_record_payload(record)))


def _training_labels_payload(
    labels: SourceAttestedSupervisedTemplateTrainingLabelsV1 | Mapping[str, object],
) -> dict[str, object]:
    payload = (
        labels.model_dump(mode="json", warnings=False)
        if isinstance(labels, SourceAttestedSupervisedTemplateTrainingLabelsV1)
        else dict(labels)
    )
    payload.pop("training_labels_receipt_sha256", None)
    return payload


def training_labels_receipt_sha256(
    labels: SourceAttestedSupervisedTemplateTrainingLabelsV1 | Mapping[str, object],
) -> str:
    return sha256_bytes(canonical_json_bytes(_training_labels_payload(labels)))


def _model_training_receipt_payload(
    receipt: SourceAttestedSupervisedTemplateModelTrainingReceiptV1 | Mapping[str, object],
) -> dict[str, object]:
    payload = (
        receipt.model_dump(mode="json", warnings=False)
        if isinstance(receipt, SourceAttestedSupervisedTemplateModelTrainingReceiptV1)
        else dict(receipt)
    )
    payload.pop("model_training_receipt_sha256", None)
    return payload


def model_training_receipt_sha256(
    receipt: SourceAttestedSupervisedTemplateModelTrainingReceiptV1 | Mapping[str, object],
) -> str:
    return sha256_bytes(canonical_json_bytes(_model_training_receipt_payload(receipt)))


def _asset_family(
    asset: DatasetAssetReceiptV1,
) -> tuple[Literal["clean", "error"], SyntheticDocumentType, str]:
    """Derive the family from the receipt path, never from packet labels."""

    relative = asset.relative_path.replace("\\", "/")
    parts = relative.split("/")
    if len(parts) != 3:
        raise SupervisedTemplateTrainingError("supervised_training_receipt_path_invalid")
    fixture_directory, set_directory, filename = parts
    if fixture_directory not in {"clean_sets", "error_sets"} or not set_directory.startswith(
        "set_"
    ):
        raise SupervisedTemplateTrainingError("supervised_training_receipt_path_invalid")
    set_number = set_directory.removeprefix("set_")
    if len(set_number) != 4 or not set_number.isdigit():
        raise SupervisedTemplateTrainingError("supervised_training_receipt_path_invalid")
    prefix = "C" if fixture_directory == "clean_sets" else "E"
    if not filename.startswith(f"{prefix}{set_number}_") or not filename.endswith(".pdf"):
        raise SupervisedTemplateTrainingError("supervised_training_receipt_path_invalid")
    suffix = filename.removeprefix(f"{prefix}{set_number}_").removesuffix(".pdf")
    if suffix == "BL":
        document_type: SyntheticDocumentType = "bill_of_lading"
    elif suffix == "INV":
        document_type = "invoice"
    elif suffix == "PL":
        document_type = "packing_list"
    else:
        raise SupervisedTemplateTrainingError("supervised_training_receipt_path_invalid")
    fixture_kind: Literal["clean", "error"] = (
        "clean" if fixture_directory == "clean_sets" else "error"
    )
    return fixture_kind, document_type, set_number


def _validate_packets(
    *,
    receipt: DatasetReceiptV1,
    development_packet: EvaluationPacketV2,
    sealed_packet: EvaluationPacketV2,
    expected_dataset_scope: SourceAttestedSyntheticDatasetScopeV1,
) -> dict[str, DatasetAssetReceiptV1]:
    if development_packet.split != "development" or sealed_packet.split != "sealed":
        raise SupervisedTemplateTrainingError("supervised_training_packet_kind_invalid")
    if development_packet.packet_receipt_sha256 == sealed_packet.packet_receipt_sha256:
        raise SupervisedTemplateTrainingError("supervised_training_packet_overlap")
    try:
        validate_source_attested_synthetic_dataset_scope(
            receipt=receipt, expected_scope=expected_dataset_scope
        )
        validate_source_attested_synthetic_packet_scope(
            receipt=receipt,
            packet=development_packet,
            expected_scope=expected_dataset_scope,
        )
        validate_source_attested_synthetic_packet_scope(
            receipt=receipt,
            packet=sealed_packet,
            expected_scope=expected_dataset_scope,
        )
    except SourceAttestedSyntheticPdfTruthError as error:
        raise SupervisedTemplateTrainingError("supervised_training_packet_scope_invalid") from error
    assets_by_id = {asset.asset_id: asset for asset in receipt.assets}
    development_ids = {item.asset_id for item in development_packet.items}
    sealed_ids = {item.asset_id for item in sealed_packet.items}
    if development_ids & sealed_ids:
        raise SupervisedTemplateTrainingError("supervised_training_packet_overlap")
    return assets_by_id


def _held_out_families(
    *,
    packets: Sequence[EvaluationPacketV2],
    assets_by_id: Mapping[str, DatasetAssetReceiptV1],
) -> set[str]:
    family_set_numbers: set[str] = set()
    for packet in packets:
        for item in packet.items:
            asset = assets_by_id.get(item.asset_id)
            if asset is None or asset.sha256 != item.source_sha256:
                raise SupervisedTemplateTrainingError("supervised_training_packet_scope_invalid")
            fixture_kind, document_type, set_number = _asset_family(asset)
            expected_set_id = ("C" if fixture_kind == "clean" else "E") + set_number
            if (
                item.fixture_kind != fixture_kind
                or item.document_type != document_type
                or item.set_id != expected_set_id
            ):
                raise SupervisedTemplateTrainingError("supervised_training_packet_scope_invalid")
            family_set_numbers.add(set_number)
    if not family_set_numbers:
        raise SupervisedTemplateTrainingError("supervised_training_packet_scope_invalid")
    return family_set_numbers


def build_source_attested_supervised_template_training_split(
    *,
    receipt: DatasetReceiptV1,
    development_packet: EvaluationPacketV2,
    sealed_packet: EvaluationPacketV2,
    expected_dataset_scope: SourceAttestedSyntheticDatasetScopeV1 = (
        EXPECTED_SOURCE_ATTESTED_SYNTHETIC_DATASET_SCOPE
    ),
) -> SourceAttestedSupervisedTemplateTrainingSplitV1:
    """Exclude every clean/error document in every development/sealed family.

    This is intentionally stricter than excluding only the 60 development and
    20 sealed asset IDs.  A clean/error sibling has the same family template
    and would leak the held-out set's content into a supervised trainer.
    """

    assets_by_id = _validate_packets(
        receipt=receipt,
        development_packet=development_packet,
        sealed_packet=sealed_packet,
        expected_dataset_scope=expected_dataset_scope,
    )
    held_out_families = _held_out_families(
        packets=(development_packet, sealed_packet), assets_by_id=assets_by_id
    )
    packet_asset_ids = {
        item.asset_id for packet in (development_packet, sealed_packet) for item in packet.items
    }
    training_assets: list[SupervisedTemplateTrainingAssetV1] = []
    excluded_family_asset_count = 0
    for asset in receipt.assets:
        fixture_kind, document_type, set_number = _asset_family(asset)
        if set_number in held_out_families:
            excluded_family_asset_count += 1
            continue
        training_assets.append(
            SupervisedTemplateTrainingAssetV1(
                asset_id=asset.asset_id,
                source_sha256=asset.sha256,
                fixture_kind=fixture_kind,
                document_type=document_type,
                family_set_number=set_number,
            )
        )
    training_assets.sort(key=lambda asset: asset.asset_id)
    if not training_assets or any(asset.asset_id in packet_asset_ids for asset in training_assets):
        raise SupervisedTemplateTrainingError("supervised_training_family_exclusion_invalid")
    strata_counts: dict[tuple[str, SyntheticDocumentType], int] = {}
    for training_asset_entry in training_assets:
        key = (training_asset_entry.fixture_kind, training_asset_entry.document_type)
        strata_counts[key] = strata_counts.get(key, 0) + 1
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "source_attested_supervised_template_training_split",
        "split_revision": SUPERVISED_TEMPLATE_TRAINING_SPLIT_REVISION,
        "dataset_id": receipt.dataset_id,
        "dataset_revision": receipt.dataset_revision,
        "dataset_receipt_sha256": receipt_sha256(receipt),
        "development_packet_receipt_sha256": development_packet.packet_receipt_sha256,
        "sealed_packet_receipt_sha256": sealed_packet.packet_receipt_sha256,
        "held_out_family_set_numbers": sorted(held_out_families),
        "held_out_family_count": len(held_out_families),
        "excluded_family_asset_count": excluded_family_asset_count,
        "training_assets": [asset.model_dump(mode="json") for asset in training_assets],
        "training_asset_count": len(training_assets),
        "training_asset_list_sha256": training_asset_list_sha256(training_assets),
        "training_strata": [
            {
                "fixture_kind": fixture_kind,
                "document_type": document_type,
                "asset_count": count,
            }
            for (fixture_kind, document_type), count in sorted(strata_counts.items())
        ],
    }
    payload["training_split_receipt_sha256"] = training_split_receipt_sha256(payload)
    return SourceAttestedSupervisedTemplateTrainingSplitV1.model_validate(payload)


def _validated_training_split(
    *,
    split: SourceAttestedSupervisedTemplateTrainingSplitV1,
    receipt: DatasetReceiptV1,
    development_packet: EvaluationPacketV2,
    sealed_packet: EvaluationPacketV2,
    expected_dataset_scope: SourceAttestedSyntheticDatasetScopeV1,
) -> None:
    rebuilt = build_source_attested_supervised_template_training_split(
        receipt=receipt,
        development_packet=development_packet,
        sealed_packet=sealed_packet,
        expected_dataset_scope=expected_dataset_scope,
    )
    if rebuilt != split:
        raise SupervisedTemplateTrainingError("supervised_training_split_binding_invalid")


def build_source_attested_supervised_template_training_labels(
    *,
    receipt: DatasetReceiptV1,
    development_packet: EvaluationPacketV2,
    sealed_packet: EvaluationPacketV2,
    training_split: SourceAttestedSupervisedTemplateTrainingSplitV1,
    source_directory: Path,
    expected_dataset_scope: SourceAttestedSyntheticDatasetScopeV1 = (
        EXPECTED_SOURCE_ATTESTED_SYNTHETIC_DATASET_SCOPE
    ),
) -> SourceAttestedSupervisedTemplateTrainingLabelsV1:
    """Compile compact labels from final PDFs only, with no CSV input.

    The parser validates each final PDF against the pinned content-stream
    profile before it emits a semantic slot.  Raw field text is used only in
    memory to obtain a normalized value digest and is never written to this
    label artifact or a CLI summary.
    """

    _validated_training_split(
        split=training_split,
        receipt=receipt,
        development_packet=development_packet,
        sealed_packet=sealed_packet,
        expected_dataset_scope=expected_dataset_scope,
    )
    source_directory = _local_input_directory(
        source_directory, code="supervised_training_source_directory_invalid"
    )
    assets_by_id = {asset.asset_id: asset for asset in receipt.assets}
    records: list[SupervisedTemplateTrainingLabelRecordV1] = []
    for training_asset in training_split.training_assets:
        asset = assets_by_id.get(training_asset.asset_id)
        if asset is None or asset.sha256 != training_asset.source_sha256:
            raise SupervisedTemplateTrainingError("supervised_training_split_binding_invalid")
        fixture_kind, document_type, set_number = _asset_family(asset)
        if (
            fixture_kind != training_asset.fixture_kind
            or document_type != training_asset.document_type
            or set_number != training_asset.family_set_number
        ):
            raise SupervisedTemplateTrainingError("supervised_training_split_binding_invalid")
        try:
            pdf_bytes = _read_verified_source_pdf(source_directory=source_directory, asset=asset)
        except SourceAttestedSyntheticPdfTruthError as error:
            raise SupervisedTemplateTrainingError(
                "supervised_training_source_asset_invalid"
            ) from error
        try:
            parsed = _parse_pdf_content_stream_v1(pdf_bytes)
            profile = _require_profile(document_type=document_type, parsed=parsed)
            fields = _fields_from_profile(parsed=parsed, profile=profile)
        except SourceAttestedSyntheticPdfTruthError as error:
            raise SupervisedTemplateTrainingError(
                "supervised_training_source_label_invalid"
            ) from error
        fields_by_name = {field.field_name: field for field in fields}
        if len(fields_by_name) != len(fields):
            raise SupervisedTemplateTrainingError("supervised_training_source_label_invalid")
        labels: list[SupervisedTemplateTrainingFieldLabelV1] = []
        for slot in profile.field_slots:
            field = fields_by_name.get(slot.field_name)
            if field is None:
                raise SupervisedTemplateTrainingError("supervised_training_source_label_invalid")
            normalized_value = normalize_field_value(field.value, field.normalizer)
            if not normalized_value:
                raise SupervisedTemplateTrainingError("supervised_training_source_label_invalid")
            labels.append(
                SupervisedTemplateTrainingFieldLabelV1(
                    field_name=slot.field_name,
                    normalizer=slot.normalizer,
                    normalized_value_sha256=sha256_bytes(normalized_value.encode("utf-8")),
                )
            )
        labels.sort(key=lambda label: label.field_name)
        record_payload: dict[str, object] = {
            "asset_id": training_asset.asset_id,
            "source_sha256": training_asset.source_sha256,
            "fixture_kind": fixture_kind,
            "document_type": document_type,
            "family_set_number": set_number,
            "content_stream_sha256": parsed.decoded_sha256,
            "field_labels": [label.model_dump(mode="json") for label in labels],
        }
        record_payload["training_label_record_sha256"] = training_label_record_sha256(
            record_payload
        )
        records.append(SupervisedTemplateTrainingLabelRecordV1.model_validate(record_payload))
    records.sort(key=lambda record: record.asset_id)
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "source_attested_supervised_template_training_labels",
        "label_kind": SUPERVISED_TEMPLATE_TRAINING_LABEL_KIND,
        "label_compiler_revision": SUPERVISED_TEMPLATE_TRAINING_LABEL_COMPILER_REVISION,
        "dataset_id": receipt.dataset_id,
        "dataset_revision": receipt.dataset_revision,
        "dataset_receipt_sha256": receipt_sha256(receipt),
        "training_split_receipt_sha256": training_split.training_split_receipt_sha256,
        "record_count": len(records),
        "field_label_count": sum(len(record.field_labels) for record in records),
        "records": [record.model_dump(mode="json") for record in records],
    }
    payload["training_labels_receipt_sha256"] = training_labels_receipt_sha256(payload)
    return SourceAttestedSupervisedTemplateTrainingLabelsV1.model_validate(payload)


def build_source_attested_supervised_template_model_training_receipt(
    *,
    training_split: SourceAttestedSupervisedTemplateTrainingSplitV1,
    training_labels: SourceAttestedSupervisedTemplateTrainingLabelsV1,
    model_kind: str,
    model_id: str,
    model_revision: str,
    model_artifact_sha256: str,
    training_recipe_sha256: str,
    training_environment_inventory_sha256: str,
) -> SourceAttestedSupervisedTemplateModelTrainingReceiptV1:
    """Bind a future model's opaque SHA to these training-only artifacts.

    No model bytes are read here.  In particular, the resulting receipt has no
    asset IDs, source SHA values, source text, value digests, or profile/run
    coordinates, so it is safe for a future runtime configuration to reference
    as provenance only.
    """

    if (
        training_labels.dataset_id != training_split.dataset_id
        or training_labels.dataset_revision != training_split.dataset_revision
        or training_labels.dataset_receipt_sha256 != training_split.dataset_receipt_sha256
        or training_labels.training_split_receipt_sha256
        != training_split.training_split_receipt_sha256
    ):
        raise SupervisedTemplateTrainingError("supervised_training_model_binding_invalid")
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "source_attested_supervised_template_model_training_receipt",
        "dataset_id": training_split.dataset_id,
        "dataset_revision": training_split.dataset_revision,
        "dataset_receipt_sha256": training_split.dataset_receipt_sha256,
        "training_split_receipt_sha256": training_split.training_split_receipt_sha256,
        "training_labels_receipt_sha256": training_labels.training_labels_receipt_sha256,
        "model_kind": model_kind,
        "model_id": model_id,
        "model_revision": model_revision,
        "model_artifact_sha256": model_artifact_sha256,
        "training_recipe_sha256": training_recipe_sha256,
        "training_environment_inventory_sha256": training_environment_inventory_sha256,
    }
    payload["model_training_receipt_sha256"] = model_training_receipt_sha256(payload)
    try:
        return SourceAttestedSupervisedTemplateModelTrainingReceiptV1.model_validate(payload)
    except ValidationError as error:
        raise SupervisedTemplateTrainingError(
            "supervised_training_model_receipt_invalid"
        ) from error


def _read_canonical_split(path: Path) -> SourceAttestedSupervisedTemplateTrainingSplitV1:
    try:
        raw = path.read_bytes()
        value = SourceAttestedSupervisedTemplateTrainingSplitV1.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise SupervisedTemplateTrainingError("supervised_training_artifact_invalid") from error
    if raw != canonical_json_file_bytes(value.model_dump(mode="json", warnings=False)):
        raise SupervisedTemplateTrainingError("supervised_training_artifact_not_canonical")
    return value


def _read_canonical_labels(path: Path) -> SourceAttestedSupervisedTemplateTrainingLabelsV1:
    try:
        raw = path.read_bytes()
        value = SourceAttestedSupervisedTemplateTrainingLabelsV1.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise SupervisedTemplateTrainingError("supervised_training_artifact_invalid") from error
    if raw != canonical_json_file_bytes(value.model_dump(mode="json", warnings=False)):
        raise SupervisedTemplateTrainingError("supervised_training_artifact_not_canonical")
    return value


def _read_canonical_model_training_receipt(
    path: Path,
) -> SourceAttestedSupervisedTemplateModelTrainingReceiptV1:
    try:
        raw = path.read_bytes()
        value = SourceAttestedSupervisedTemplateModelTrainingReceiptV1.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise SupervisedTemplateTrainingError("supervised_training_artifact_invalid") from error
    if raw != canonical_json_file_bytes(value.model_dump(mode="json", warnings=False)):
        raise SupervisedTemplateTrainingError("supervised_training_artifact_not_canonical")
    return value


def write_source_attested_supervised_template_training_split(
    *, split: SourceAttestedSupervisedTemplateTrainingSplitV1, output_directory: Path
) -> Path:
    directory = _local_output_directory(
        output_directory, code="supervised_training_split_directory_invalid"
    )
    return _write_hash_named_immutable(
        directory=directory,
        digest=split.training_split_receipt_sha256,
        payload=split.model_dump(mode="json", warnings=False),
        code="supervised_training_split_immutable_conflict",
    )


def read_canonical_source_attested_supervised_template_training_split(
    path: Path,
) -> SourceAttestedSupervisedTemplateTrainingSplitV1:
    value = _read_canonical_split(path)
    if path.name != f"{value.training_split_receipt_sha256}.json":
        raise SupervisedTemplateTrainingError("supervised_training_split_filename_invalid")
    return value


def write_source_attested_supervised_template_training_labels(
    *, labels: SourceAttestedSupervisedTemplateTrainingLabelsV1, output_directory: Path
) -> Path:
    directory = _local_output_directory(
        output_directory, code="supervised_training_labels_directory_invalid"
    )
    return _write_hash_named_immutable(
        directory=directory,
        digest=labels.training_labels_receipt_sha256,
        payload=labels.model_dump(mode="json", warnings=False),
        code="supervised_training_labels_immutable_conflict",
    )


def read_canonical_source_attested_supervised_template_training_labels(
    path: Path,
) -> SourceAttestedSupervisedTemplateTrainingLabelsV1:
    value = _read_canonical_labels(path)
    if path.name != f"{value.training_labels_receipt_sha256}.json":
        raise SupervisedTemplateTrainingError("supervised_training_labels_filename_invalid")
    return value


def write_source_attested_supervised_template_model_training_receipt(
    *, receipt: SourceAttestedSupervisedTemplateModelTrainingReceiptV1, output_directory: Path
) -> Path:
    directory = _local_output_directory(
        output_directory, code="supervised_training_model_receipt_directory_invalid"
    )
    return _write_hash_named_immutable(
        directory=directory,
        digest=receipt.model_training_receipt_sha256,
        payload=receipt.model_dump(mode="json", warnings=False),
        code="supervised_training_model_receipt_immutable_conflict",
    )


def read_canonical_source_attested_supervised_template_model_training_receipt(
    path: Path,
) -> SourceAttestedSupervisedTemplateModelTrainingReceiptV1:
    value = _read_canonical_model_training_receipt(path)
    if path.name != f"{value.model_training_receipt_sha256}.json":
        raise SupervisedTemplateTrainingError("supervised_training_model_receipt_filename_invalid")
    return value


__all__ = [
    "SUPERVISED_TEMPLATE_TRAINING_LABEL_COMPILER_REVISION",
    "SUPERVISED_TEMPLATE_TRAINING_LABEL_KIND",
    "SUPERVISED_TEMPLATE_TRAINING_SPLIT_REVISION",
    "SourceAttestedSupervisedTemplateTrainingLabelsV1",
    "SourceAttestedSupervisedTemplateModelTrainingReceiptV1",
    "SourceAttestedSupervisedTemplateTrainingSplitV1",
    "SupervisedTemplateTrainingError",
    "SupervisedTemplateTrainingFieldLabelV1",
    "SupervisedTemplateTrainingLabelRecordV1",
    "SupervisedTemplateTrainingAssetV1",
    "build_source_attested_supervised_template_training_labels",
    "build_source_attested_supervised_template_model_training_receipt",
    "build_source_attested_supervised_template_training_split",
    "read_canonical_source_attested_supervised_template_training_labels",
    "read_canonical_source_attested_supervised_template_model_training_receipt",
    "read_canonical_source_attested_supervised_template_training_split",
    "model_training_receipt_sha256",
    "training_asset_list_sha256",
    "training_label_record_sha256",
    "training_labels_receipt_sha256",
    "training_split_receipt_sha256",
    "write_source_attested_supervised_template_training_labels",
    "write_source_attested_supervised_template_model_training_receipt",
    "write_source_attested_supervised_template_training_split",
]
