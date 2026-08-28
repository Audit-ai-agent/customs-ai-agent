from __future__ import annotations

import importlib.metadata
import hashlib

import pytest
from pydantic import ValidationError

import customs_ai_agent.evaluation.source_attested_bl_repair_audit as audit
from customs_ai_agent.evaluation.synthetic_raster import (
    PdfiumFinalPdfGeometryExtractorV1,
    RASTER_PROFILES,
    RasterGoldArtifactV1,
    SyntheticRasterPdfiumCompiler,
    compile_raster_field_gold,
    compile_raster_gold,
    verify_rendered_raster_output,
)
from customs_ai_agent.evaluation.synthetic_raster_text import (
    source_only_text_gold_from_raster_gold,
)


def _parse(*, family_index: int, fixture_kind: str):
    source_pdf = audit.build_bl_repair_audit_source_pdf(
        family_index=family_index,
        fixture_kind=fixture_kind,  # type: ignore[arg-type]
    )
    source = audit.BlRepairAuditContentStreamParserV1(
        family_id=f"blra-family-{family_index:03d}"
    ).parse_final_pdf_content_stream(
        source_pdf=source_pdf,
        asset_id=f"blra-{family_index:03d}-{fixture_kind}",
    )
    return source_pdf, source


def test_new_bl_source_is_deterministic_distinct_and_source_parsed(monkeypatch) -> None:
    first_pdf, clean = _parse(family_index=0, fixture_kind="clean")
    repeated_pdf = audit.build_bl_repair_audit_source_pdf(family_index=0, fixture_kind="clean")
    error_pdf, error = _parse(family_index=0, fixture_kind="error")

    assert first_pdf == repeated_pdf
    assert first_pdf != error_pdf
    assert len(clean.spans) == 27
    assert len(clean.fields) == 13
    assert {field.field_id for field in clean.fields} == {
        "exporter_name",
        "consignee",
        "load_port",
        "disch_port",
        "final_dest",
        "vessel",
        "voyage",
        "bl_no",
        "container_no",
        "seal_no",
        "good",
        "gross_weight",
        "containers",
    }
    clean_fields = {field.field_id: field.value for field in clean.fields}
    error_fields = {field.field_id: field.value for field in error.fields}
    assert sum(clean_fields[name] != error_fields[name] for name in clean_fields) == 1
    assert clean_fields["voyage"] != error_fields["voyage"]

    # Gold parsing is intentionally independent of the renderer-value helper.
    monkeypatch.setattr(
        audit,
        "_generated_field_values",
        lambda **_arguments: (_ for _ in ()).throw(AssertionError("generator value leak")),
    )
    reparsed = audit.BlRepairAuditContentStreamParserV1(
        family_id="blra-family-000"
    ).parse_final_pdf_content_stream(
        source_pdf=first_pdf,
        asset_id="blra-000-clean",
    )
    assert reparsed == clean


@pytest.mark.parametrize("family_index", [0, 1, 2])
def test_each_error_sibling_has_exactly_one_visible_source_mutation(family_index: int) -> None:
    _clean_pdf, clean = _parse(family_index=family_index, fixture_kind="clean")
    _error_pdf, error = _parse(family_index=family_index, fixture_kind="error")
    clean_fields = {field.field_id: field.value for field in clean.fields}
    error_fields = {field.field_id: field.value for field in error.fields}
    assert sum(clean_fields[name] != error_fields[name] for name in clean_fields) == 1


def test_pdfium_one_to_one_and_all_four_raster_profiles() -> None:
    source_pdf, source = _parse(family_index=7, fixture_kind="clean")
    pdfium_version = importlib.metadata.version("pypdfium2")
    geometry = PdfiumFinalPdfGeometryExtractorV1(pdfium_version).extract(
        source_pdf=source_pdf,
        source=source,
    )
    assert len(geometry) == len(source.spans)

    compiler = SyntheticRasterPdfiumCompiler(pdfium_version)
    for profile_id in RASTER_PROFILES:
        rendered = compiler.render(
            source_pdf=source_pdf,
            source=source,
            profile_id=profile_id,
        )
        verified = verify_rendered_raster_output(
            rendered,
            pdfium_version=pdfium_version,
        )
        assert verified.media_type == RASTER_PROFILES[profile_id].media_type
        gold = RasterGoldArtifactV1(
            asset_id=source.asset_id,
            family_id=source.family_id,
            document_type="bill_of_lading",
            source_pdf_sha256=source.source_pdf_sha256,
            profile_id=profile_id,
            spans=compile_raster_gold(
                source,
                profile_id=profile_id,
                pdfium_spans=geometry,
            ),
            fields=compile_raster_field_gold(
                source,
                profile_id=profile_id,
                pdfium_spans=geometry,
            ),
            table_cells=[],
        )
        text_gold = source_only_text_gold_from_raster_gold(gold)
        assert len(text_gold.critical_values) == 13


def _score(*, fixture_kind: str, profile_id: str, voyage_matches: int):
    return audit.BlRepairAuditScoreV1(
        fixture_kind=fixture_kind,  # type: ignore[arg-type]
        profile_id=profile_id,  # type: ignore[arg-type]
        character_errors=0,
        character_reference=10_000,
        critical_matches=1300,
        critical_total=1300,
        voyage_matches=voyage_matches,
        voyage_total=100,
        character_error_rate=0.0,
        critical_value_exact_match=1.0,
        voyage_exact_match=voyage_matches / 100,
        gate_passed=voyage_matches >= 98,
    )


def test_voyage_gate_cannot_be_diluted_by_other_twelve_fields() -> None:
    failed = _score(
        fixture_kind="clean",
        profile_id="jpeg-216-q75",
        voyage_matches=97,
    )
    assert not failed.gate_passed
    with pytest.raises(ValidationError, match="score is inconsistent"):
        audit.BlRepairAuditScoreV1(
            **{
                **failed.model_dump(mode="python"),
                "gate_passed": True,
            }
        )


def test_evaluation_binds_content_addressed_candidate_bundle() -> None:
    scores = sorted(
        (
            _score(fixture_kind=kind, profile_id=profile_id, voyage_matches=100)
            for kind in ("clean", "error")
            for profile_id in RASTER_PROFILES
        ),
        key=lambda item: (item.fixture_kind, item.profile_id),
    )
    evaluation = audit.BlRepairAuditEvaluationV1(
        dataset_receipt_sha256="1" * 64,
        candidate_profile_revision="synthetic-raster-text-v1:" + "2" * 24,
        runtime_configuration_sha256="3" * 64,
        extractor_revision="direct-paddle-v19b",
        candidate_bundle_relative_path=(
            f"{audit.BL_REPAIR_AUDIT_DATASET_ID}/candidate-bundles/{'4' * 64}.json"
        ),
        candidate_bundle_sha256="4" * 64,
        scores=scores,
        gate_passed=True,
    )
    assert audit.bl_repair_audit_evaluation_sha256(evaluation) == (
        audit.bl_repair_audit_evaluation_sha256(evaluation)
    )
    with pytest.raises(ValidationError, match="evaluation matrix is invalid"):
        audit.BlRepairAuditEvaluationV1(
            **{
                **evaluation.model_dump(mode="python"),
                "candidate_bundle_relative_path": (
                    f"{audit.BL_REPAIR_AUDIT_DATASET_ID}/candidate-bundles/{'5' * 64}.json"
                ),
            }
        )


def test_receipt_requires_exact_100_family_clean_error_four_profile_scope() -> None:
    sources = []
    rasters = []
    media = {profile_id: profile.media_type for profile_id, profile in RASTER_PROFILES.items()}
    suffix = {
        "png-216": ".png",
        "jpeg-216-q75": ".jpg",
        "jpeg-150-q75": ".jpg",
        "scan-pdf-216-q75": ".pdf",
    }
    for family_index in range(100):
        family_id = f"blra-family-{family_index:03d}"
        for fixture_kind in ("clean", "error"):
            asset_id = f"blra-{family_index:03d}-{fixture_kind}"
            source_sha = hashlib.sha256(asset_id.encode("ascii")).hexdigest()
            sources.append(
                audit.BlRepairAuditSourceBindingV1(
                    asset_id=asset_id,
                    family_id=family_id,
                    fixture_kind=fixture_kind,
                    source_pdf_relative_path=(
                        f"{audit.BL_REPAIR_AUDIT_DATASET_ID}/sources/{asset_id}.pdf"
                    ),
                    source_pdf_sha256=source_sha,
                )
            )
            for profile_id in RASTER_PROFILES:
                image_sha = hashlib.sha256(f"{asset_id}:{profile_id}".encode("ascii")).hexdigest()
                gold_sha = hashlib.sha256(
                    f"gold:{asset_id}:{profile_id}".encode("ascii")
                ).hexdigest()
                rasters.append(
                    audit.BlRepairAuditRasterBindingV1(
                        asset_id=asset_id,
                        family_id=family_id,
                        fixture_kind=fixture_kind,
                        profile_id=profile_id,
                        source_pdf_sha256=source_sha,
                        image_sha256=image_sha,
                        media_type=media[profile_id],
                        byte_size=100,
                        width=100,
                        height=100,
                        asset_relative_path=(
                            f"{audit.BL_REPAIR_AUDIT_DATASET_ID}/assets/"
                            f"{asset_id}-{profile_id}{suffix[profile_id]}"
                        ),
                        gold_relative_path=(
                            f"{audit.BL_REPAIR_AUDIT_DATASET_ID}/gold/{asset_id}-{profile_id}.json"
                        ),
                        gold_sha256=gold_sha,
                    )
                )
    receipt = audit.BlRepairAuditReceiptV1(
        generation_seed_sha256=audit._GENERATION_SEED_SHA256,
        generation_recipe_sha256=audit.BL_REPAIR_AUDIT_RECIPE_SHA256,
        pdfium_version=importlib.metadata.version("pypdfium2"),
        pillow_version=importlib.metadata.version("Pillow"),
        sources=sorted(sources, key=lambda item: item.asset_id),
        rasters=sorted(rasters, key=lambda item: (item.asset_id, item.profile_id)),
    )
    assert len(receipt.sources) == 200
    assert len(receipt.rasters) == 800
    assert len(audit.bl_repair_audit_receipt_sha256(receipt)) == 64
    with pytest.raises(ValidationError):
        audit.BlRepairAuditReceiptV1(
            **{
                **receipt.model_dump(mode="python"),
                "rasters": receipt.rasters[:-1],
            }
        )
