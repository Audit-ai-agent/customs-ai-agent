from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from customs_ai_agent.evaluation import synthetic_raster_text_activation as activation
from customs_ai_agent.evaluation.synthetic_raster import RASTER_PROFILES
from customs_ai_agent.evaluation.synthetic_raster_text import RasterTextMetricCountsV1


def _selection() -> activation.RasterTextFreshScopeSelectionReceiptV1:
    return activation.RasterTextFreshScopeSelectionReceiptV1(
        synthetic_raster_receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
        selection_seed_sha256=activation._FRESH_HOLDOUT_SELECTION_SEED_SHA256,  # noqa: SLF001
        fresh_calibration_family_ids=[f"cal-{index:03d}" for index in range(50)],
        fresh_sealed_family_ids=[f"sealed-{index:03d}" for index in range(100)],
    )


@pytest.mark.parametrize(
    ("kind", "expected_prefix", "expected_count"),
    [
        ("calibration-10-family-v1", "cal", 10),
        ("sealed-stage1-50-family-v1", "sealed", 50),
        ("sealed-stage2-100-family-v1", "sealed", 100),
    ],
)
def test_projection_family_membership_is_hash_ranked_without_results(
    kind: Any,
    expected_prefix: str,
    expected_count: int,
) -> None:
    selected = activation._fresh_projection_family_ids(  # noqa: SLF001
        selection=_selection(), projection_kind=kind
    )

    assert len(selected) == expected_count
    assert selected == sorted(selected)
    assert all(item.startswith(expected_prefix) for item in selected)
    source = (
        _selection().fresh_calibration_family_ids
        if expected_prefix == "cal"
        else _selection().fresh_sealed_family_ids
    )
    if kind != "sealed-stage2-100-family-v1":
        assert selected != source[:expected_count]
    else:
        assert selected == source


def _projection_payload(kind: str) -> dict[str, object]:
    selection = _selection()
    selection_sha = activation._digest_model(selection)  # noqa: SLF001
    scope = "fresh-calibration" if kind.startswith("calibration") else "fresh-sealed"
    family_ids = activation._fresh_projection_family_ids(  # noqa: SLF001
        selection=selection, projection_kind=kind
    )
    parent = kind == "sealed-stage2-100-family-v1"
    return {
        "source_scope": scope,
        "projection_kind": kind,
        "projection_seed_sha256": activation._FRESH_PROJECTION_SEED_SHA256,  # noqa: SLF001
        "synthetic_raster_receipt_sha256": "a" * 64,
        "selection_receipt_relative_path": (
            f"synthetic-raster-text-v1/fresh-scope-selections/{selection_sha}.json"
        ),
        "selection_receipt_sha256": selection_sha,
        "full_source_gold_receipt_relative_path": (
            f"synthetic-raster-text-v1/fresh-source-gold-receipts/{scope}/{'c' * 64}.json"
        ),
        "full_source_gold_receipt_sha256": "c" * 64,
        "family_ids": family_ids,
        "family_count": len(family_ids),
        "asset_count": len(family_ids) * 24,
        "per_stratum_asset_count": len(family_ids) * 2,
        "parent_stage1_decision_relative_path": (
            f"synthetic-raster-text-v1/fresh-sealed-stage-decisions/{'d' * 64}.json"
            if parent
            else None
        ),
        "parent_stage1_decision_sha256": "d" * 64 if parent else None,
    }


def test_projection_requires_complete_fixed_shape_and_stage2_parent() -> None:
    stage1 = activation.RasterTextFreshEvaluationProjectionReceiptV1(
        **_projection_payload("sealed-stage1-50-family-v1")
    )
    assert stage1.asset_count == 1_200
    with pytest.raises(ValidationError):
        activation.RasterTextFreshEvaluationProjectionReceiptV1(
            **{
                **_projection_payload("sealed-stage1-50-family-v1"),
                "family_ids": stage1.family_ids[:-1],
            }
        )
    with pytest.raises(ValidationError):
        activation.RasterTextFreshEvaluationProjectionReceiptV1(
            **{
                **_projection_payload("sealed-stage1-50-family-v1"),
                "projection_seed_sha256": "e" * 64,
            }
        )


def test_stage1_hash_membership_is_nested_in_complete_stage2() -> None:
    selection = _selection()
    stage1 = activation._fresh_projection_family_ids(  # noqa: SLF001
        selection=selection, projection_kind="sealed-stage1-50-family-v1"
    )
    stage2 = activation._fresh_projection_family_ids(  # noqa: SLF001
        selection=selection, projection_kind="sealed-stage2-100-family-v1"
    )

    assert len(stage1) == 50
    assert len(stage2) == 100
    assert set(stage1) < set(stage2)
    with pytest.raises(ValidationError):
        activation.RasterTextFreshEvaluationProjectionReceiptV1(
            **{
                **_projection_payload("sealed-stage2-100-family-v1"),
                "parent_stage1_decision_relative_path": None,
                "parent_stage1_decision_sha256": None,
            }
        )


def _counts(
    *,
    character_errors: int,
    character_reference: int,
    critical_errors: int,
    critical_total: int,
) -> dict[tuple[str, str], RasterTextMetricCountsV1]:
    return {
        (document_type, profile_id): RasterTextMetricCountsV1(
            character_errors=character_errors,
            character_reference=character_reference,
            critical_matches=critical_total - critical_errors,
            critical_total=critical_total,
        )
        for document_type in (
            "bill_of_lading",
            "commercial_invoice",
            "packing_list",
        )
        for profile_id in RASTER_PROFILES
    }


def test_stage1_early_pass_requires_simultaneous_lower_bounds() -> None:
    full = _counts(
        character_errors=0,
        character_reference=20_000,
        critical_errors=0,
        critical_total=2_400,
    )
    strong = activation._fresh_sealed_stage1_strata(  # noqa: SLF001
        stage_counts=_counts(
            character_errors=0,
            character_reference=10_000,
            critical_errors=0,
            critical_total=1_200,
        ),
        full_counts=full,
    )
    boundary = activation._fresh_sealed_stage1_strata(  # noqa: SLF001
        stage_counts=_counts(
            character_errors=0,
            character_reference=10_000,
            critical_errors=14,
            critical_total=1_200,
        ),
        full_counts=full,
    )

    assert activation._fresh_stage1_decision_from_strata(strong) == "early_pass"  # noqa: SLF001
    assert all(item.point_gate_passed for item in boundary)
    assert all(item.critical_accuracy_lower_bound < 0.98 for item in boundary)
    assert activation._fresh_stage1_decision_from_strata(boundary) == "continue"  # noqa: SLF001


def test_preview97_stage_policy_lowers_only_critical_accuracy() -> None:
    full = _counts(
        character_errors=0,
        character_reference=20_000,
        critical_errors=0,
        critical_total=2_400,
    )
    stage = _counts(
        character_errors=0,
        character_reference=10_000,
        critical_errors=14,
        critical_total=1_200,
    )
    strict = activation._fresh_sealed_stage1_strata(  # noqa: SLF001
        stage_counts=stage,
        full_counts=full,
    )
    preview = activation._fresh_sealed_stage1_strata(  # noqa: SLF001
        stage_counts=stage,
        full_counts=full,
        quality_gate_revision="preview-draft-critical97-cer2-v1",
    )

    assert activation._fresh_stage1_decision_from_strata(strict) == "continue"  # noqa: SLF001
    assert (
        activation._fresh_stage1_decision_from_strata(  # noqa: SLF001
            preview,
            quality_gate_revision="preview-draft-critical97-cer2-v1",
        )
        == "early_pass"
    )
    receipt = activation.RasterTextFreshSealedStageDecisionReceiptV2(
        metric_revision="source-attested-semantic-text-v3",
        quality_gate_revision="preview-draft-critical97-cer2-v1",
        source_gold_receipt_relative_path="fresh/source.json",
        source_gold_receipt_sha256="a" * 64,
        candidate_execution_relative_path="fresh/execution.json",
        candidate_execution_sha256="b" * 64,
        projection_receipt_relative_path="fresh/projection.json",
        projection_receipt_sha256="c" * 64,
        decision="early_pass",
        strata=preview,
    )
    assert receipt.schema_version == "2.0"
    with pytest.raises(ValidationError):
        activation.RasterTextFreshSealedStageDecisionReceiptV1(
            **receipt.model_dump(mode="json", exclude={"schema_version", "quality_gate_revision"})
        )


def test_stage1_fails_early_only_when_full_gate_is_impossible() -> None:
    full = _counts(
        character_errors=0,
        character_reference=20_000,
        critical_errors=0,
        critical_total=2_400,
    )
    recoverable = activation._fresh_sealed_stage1_strata(  # noqa: SLF001
        stage_counts=_counts(
            character_errors=300,
            character_reference=10_000,
            critical_errors=48,
            critical_total=1_200,
        ),
        full_counts=full,
    )
    impossible = activation._fresh_sealed_stage1_strata(  # noqa: SLF001
        stage_counts=_counts(
            character_errors=401,
            character_reference=10_000,
            critical_errors=49,
            critical_total=1_200,
        ),
        full_counts=full,
    )

    assert activation._fresh_stage1_decision_from_strata(recoverable) == "continue"  # noqa: SLF001
    assert activation._fresh_stage1_decision_from_strata(impossible) == "early_fail"  # noqa: SLF001


@pytest.mark.parametrize(
    ("updates"),
    [
        {"character_errors": 101, "character_reference": 100},
        {"critical_matches": 101, "critical_total": 100},
        {"character_reference": 101, "full_character_reference": 100},
        {"critical_total": 101, "full_critical_total": 100},
    ],
)
def test_stage_stratum_rejects_impossible_raw_count_nesting(
    updates: dict[str, int],
) -> None:
    payload: dict[str, object] = {
        "document_type": "packing_list",
        "profile_id": "jpeg-150-q75",
        "character_errors": 1,
        "character_reference": 100,
        "critical_matches": 99,
        "critical_total": 100,
        "full_character_reference": 200,
        "full_critical_total": 200,
        "character_accuracy_lower_bound": 0.9,
        "critical_accuracy_lower_bound": 0.9,
        "point_gate_passed": True,
        "full_gate_recoverable": True,
    }
    with pytest.raises(ValidationError):
        activation.RasterTextFreshSealedStageStratumV1(**{**payload, **updates})


def test_recalculation_overrides_injected_decision_fields() -> None:
    stage = _counts(
        character_errors=0,
        character_reference=10_000,
        critical_errors=14,
        critical_total=1_200,
    )
    full = _counts(
        character_errors=0,
        character_reference=20_000,
        critical_errors=0,
        critical_total=2_400,
    )
    expected = activation._fresh_sealed_stage1_strata(  # noqa: SLF001
        stage_counts=stage, full_counts=full
    )
    injected = expected[0].model_copy(
        update={
            "critical_accuracy_lower_bound": 1.0,
            "point_gate_passed": False,
            "full_gate_recoverable": False,
        }
    )

    assert (
        injected
        != activation._fresh_sealed_stage1_strata(  # noqa: SLF001
            stage_counts=stage, full_counts=full
        )[0]
    )


def test_stage_decision_verifier_rejects_injected_derived_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    strata = activation._fresh_sealed_stage1_strata(  # noqa: SLF001
        stage_counts=_counts(
            character_errors=0,
            character_reference=10_000,
            critical_errors=0,
            critical_total=1_200,
        ),
        full_counts=_counts(
            character_errors=0,
            character_reference=20_000,
            critical_errors=0,
            critical_total=2_400,
        ),
    )
    expected = activation.RasterTextFreshSealedStageDecisionReceiptV1(
        metric_revision="source-attested-semantic-text-v3",
        source_gold_receipt_relative_path="fresh/source.json",
        source_gold_receipt_sha256="a" * 64,
        candidate_execution_relative_path="fresh/execution.json",
        candidate_execution_sha256="b" * 64,
        projection_receipt_relative_path="fresh/projection.json",
        projection_receipt_sha256="c" * 64,
        decision="early_pass",
        strata=strata,
    )
    injected_strata = list(expected.strata)
    injected_strata[0] = injected_strata[0].model_copy(
        update={"critical_accuracy_lower_bound": 0.98}
    )
    injected = expected.model_copy(update={"strata": injected_strata})
    monkeypatch.setattr(
        activation,
        "_read_fresh_sealed_stage_decision",
        lambda **_kwargs: (injected, "d" * 64),
    )
    monkeypatch.setattr(
        activation,
        "verify_source_attested_fresh_raster_text_evidence",
        lambda **_kwargs: (object(), object(), object(), []),
    )
    monkeypatch.setattr(
        activation,
        "_build_fresh_stage1_decision_artifact",
        lambda **_kwargs: expected,
    )

    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^raster_text_fresh_stage_decision_invalid$",
    ):
        activation.verify_fresh_raster_text_sealed_stage1_decision(
            document_preview_root=root,
            decision_relative_path="fresh/decision.json",
        )
