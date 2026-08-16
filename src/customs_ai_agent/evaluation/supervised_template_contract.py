"""Public contract for the r1 supervised native-PDF template profile.

This module deliberately contains only the *output* schema and fixed profile
identity.  It does not import the source-layout compiler, a content-stream
parser, Gold/CSV data, or a training-label reader.  The training process may
learn selectors from its family-safe partition, while runtime code uses this
contract solely to reject a model that claims to be the r1 profile but has a
different public field schema.

The profile proves compatibility with the fixed born-digital synthetic r1
template only.  It is not a claim about general OCR, scanned documents, or
production customs paperwork.
"""

from __future__ import annotations

from typing import Final, Literal, Mapping, cast

from customs_ai_agent.evaluation.document_ocr_quality import (
    FieldNormalizer,
    SyntheticDocumentType,
)
from customs_ai_agent.evaluation.native_pdf_semantic_model import (
    NativePdfSemanticModelArtifactV1,
)


SUPERVISED_TEMPLATE_EVALUATION_SCOPE: Final[
    Literal["synthetic_r1_supervised_native_template_compatibility"]
] = "synthetic_r1_supervised_native_template_compatibility"
SUPERVISED_TEMPLATE_CANDIDATE_PARSER_REVISION: Final[
    Literal["synthetic-pdf-supervised-template-normalizer-v11"]
] = "synthetic-pdf-supervised-template-normalizer-v11"
SUPERVISED_TEMPLATE_EXTRACTOR_REVISION: Final[
    Literal["pypdf-5.3.1-supervised-template-r1-v1"]
] = "pypdf-5.3.1-supervised-template-r1-v1"
SUPERVISED_TEMPLATE_MODEL_ID: Final[Literal["synthetic-customs-pdf-2000-r1-native-template"]] = (
    "synthetic-customs-pdf-2000-r1-native-template"
)
SUPERVISED_TEMPLATE_MODEL_REVISION: Final[Literal["supervised-native-selector-r1-v1"]] = (
    "supervised-native-selector-r1-v1"
)
SUPERVISED_TEMPLATE_MODEL_KIND: Final[Literal["native_pdf_semantic_model"]] = (
    "native_pdf_semantic_model"
)


# This is the public evaluated output schema, not source-template geometry or
# a source-parser profile.  Selector locations and transforms must be learned
# from candidate-native training evidence and are never declared here.
R1_SUPERVISED_TEMPLATE_REQUIRED_FIELDS: Final[
    Mapping[SyntheticDocumentType, tuple[tuple[str, FieldNormalizer], ...]]
] = {
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
R1_SUPERVISED_TEMPLATE_DOCUMENT_TYPES: Final[tuple[SyntheticDocumentType, ...]] = (
    "bill_of_lading",
    "invoice",
    "packing_list",
)


class SupervisedTemplateContractError(ValueError):
    """A content-free r1 profile validation failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def validate_r1_supervised_template_model(model: NativePdfSemanticModelArtifactV1) -> None:
    """Accept only the exact public r1 semantic-model contract.

    The generic artifact schema separately rejects unknown wire fields and
    invalid selector shapes.  This profile validator intentionally adds no
    location, value, label, source hash, or asset-ID knowledge.
    """

    if (
        model.artifact_kind != SUPERVISED_TEMPLATE_MODEL_KIND
        or model.model_id != SUPERVISED_TEMPLATE_MODEL_ID
        or model.model_revision != SUPERVISED_TEMPLATE_MODEL_REVISION
        or model.inference_revision != "native-template-selector-v1"
        or tuple(document.document_type for document in model.documents)
        != R1_SUPERVISED_TEMPLATE_DOCUMENT_TYPES
    ):
        raise SupervisedTemplateContractError("supervised_template_model_scope_invalid")
    for document in model.documents:
        if document.document_type not in R1_SUPERVISED_TEMPLATE_DOCUMENT_TYPES:
            raise SupervisedTemplateContractError("supervised_template_model_scope_invalid")
        document_type = cast(SyntheticDocumentType, document.document_type)
        expected_fields = R1_SUPERVISED_TEMPLATE_REQUIRED_FIELDS.get(document_type)
        signature = document.structural_signature
        if (
            expected_fields is None
            or signature.page_count != 1
            or signature.minimum_span_count < 1
            or signature.minimum_table_cell_count < 1
            or len(signature.required_selectors) != 1
            or signature.required_selectors[0].source_kind != "span_anchor"
            or tuple((field.field_name, field.normalizer) for field in document.fields)
            != expected_fields
        ):
            raise SupervisedTemplateContractError("supervised_template_model_scope_invalid")


__all__ = [
    "R1_SUPERVISED_TEMPLATE_DOCUMENT_TYPES",
    "R1_SUPERVISED_TEMPLATE_REQUIRED_FIELDS",
    "SUPERVISED_TEMPLATE_CANDIDATE_PARSER_REVISION",
    "SUPERVISED_TEMPLATE_EVALUATION_SCOPE",
    "SUPERVISED_TEMPLATE_EXTRACTOR_REVISION",
    "SUPERVISED_TEMPLATE_MODEL_ID",
    "SUPERVISED_TEMPLATE_MODEL_KIND",
    "SUPERVISED_TEMPLATE_MODEL_REVISION",
    "SupervisedTemplateContractError",
    "validate_r1_supervised_template_model",
]
