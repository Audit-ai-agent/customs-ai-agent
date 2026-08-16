from __future__ import annotations

import pytest

from customs_ai_agent.evaluation.native_pdf_semantic_model import (
    NativePdfSemanticModelArtifactV1,
)
from customs_ai_agent.evaluation.supervised_template_contract import (
    R1_SUPERVISED_TEMPLATE_DOCUMENT_TYPES,
    R1_SUPERVISED_TEMPLATE_REQUIRED_FIELDS,
    SUPERVISED_TEMPLATE_MODEL_ID,
    SUPERVISED_TEMPLATE_MODEL_REVISION,
    SupervisedTemplateContractError,
    validate_r1_supervised_template_model,
)


def _r1_model() -> NativePdfSemanticModelArtifactV1:
    documents: list[dict[str, object]] = []
    for document_index, document_type in enumerate(R1_SUPERVISED_TEMPLATE_DOCUMENT_TYPES):
        fields = [
            {
                "field_name": field_name,
                "normalizer": normalizer,
                "selectors": [
                    {
                        "source_kind": "span_reading_order",
                        "page_number": 1,
                        "reading_order": field_index + 2,
                    }
                ],
                "value_transform": "raw-v1",
            }
            for field_index, (field_name, normalizer) in enumerate(
                R1_SUPERVISED_TEMPLATE_REQUIRED_FIELDS[document_type]
            )
        ]
        documents.append(
            {
                "document_type": document_type,
                "structural_signature": {
                    "page_count": 1,
                    "minimum_span_count": 64,
                    "minimum_table_cell_count": 10,
                    "required_selectors": [
                        {
                            "source_kind": "span_anchor",
                            "page_number": 1,
                            "anchor_x": 0.1 + document_index / 10,
                            "anchor_y": 0.2,
                            "anchor_tolerance": 0.00001,
                        }
                    ],
                },
                "fields": fields,
            }
        )
    return NativePdfSemanticModelArtifactV1.model_validate(
        {
            "schema_version": "1.0",
            "artifact_kind": "native_pdf_semantic_model",
            "model_id": SUPERVISED_TEMPLATE_MODEL_ID,
            "model_revision": SUPERVISED_TEMPLATE_MODEL_REVISION,
            "inference_revision": "native-template-selector-v1",
            "documents": documents,
        }
    )


def test_accepts_exact_public_r1_supervised_template_model_schema() -> None:
    validate_r1_supervised_template_model(_r1_model())


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    [
        ("model_id", "another-model"),
        ("model_revision", "v2"),
    ],
)
def test_rejects_wrong_profile_identity(attribute: str, replacement: str) -> None:
    model = _r1_model()

    with pytest.raises(
        SupervisedTemplateContractError, match="supervised_template_model_scope_invalid"
    ):
        validate_r1_supervised_template_model(model.model_copy(update={attribute: replacement}))


def test_rejects_a_public_field_schema_change() -> None:
    model = _r1_model()
    mutated = model.model_copy(deep=True)
    invoice = mutated.documents[1]
    fields = list(invoice.fields)
    fields.pop()
    documents = list(mutated.documents)
    documents[1] = invoice.model_copy(update={"fields": fields})
    mutated = mutated.model_copy(update={"documents": documents})

    with pytest.raises(
        SupervisedTemplateContractError, match="supervised_template_model_scope_invalid"
    ):
        validate_r1_supervised_template_model(mutated)
