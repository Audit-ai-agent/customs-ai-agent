import json
import hashlib
import sys
from pathlib import Path

from jsonschema import Draft202012Validator
from PIL import Image
import pytest

from customs_ai_agent.adapters import OcrLayoutProviderError
from customs_ai_agent.adapters.llm import LLMError
from customs_ai_agent.schemas.models import (
    BBox,
    ImagePageInput,
    OcrLayoutLedger,
    OcrLayoutPage,
    OcrLayoutSpan,
    stable_ocr_ledger_id,
)
from evals.combine_ocr_release_gate import REDACTION as OCR_RELEASE_REDACTION
from evals import run_claude_vision
from tools.build_trade_image_diagnostic_selection import (
    build_worklist,
    evaluate_gate,
)


def _forbid_analyzer_creation(*_args, **_kwargs):
    raise AssertionError("failed preflight must not create a provider client")


def _pending_adjudication_evidence(tmp_path):
    selection = json.loads(run_claude_vision.DEFAULT_SELECTION_PATH.read_text(encoding="utf-8"))
    worklist = build_worklist(selection["cases"])
    gate = evaluate_gate(selection["cases"], worklist)
    worklist_path = tmp_path / "adjudications.jsonl"
    worklist_path.write_text(
        "\n".join(json.dumps(entry, sort_keys=True, separators=(",", ":")) for entry in worklist)
        + "\n",
        encoding="utf-8",
    )
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    return worklist_path, gate_path


def _ready_adjudication_evidence(tmp_path):
    selection = json.loads(run_claude_vision.DEFAULT_SELECTION_PATH.read_text(encoding="utf-8"))
    worklist = build_worklist(selection["cases"])
    selected_by_id = {case["case_id"]: case for case in selection["cases"]}
    for entry in worklist:
        selected = selected_by_id[entry["case_id"]]
        image_hashes = [document["sha256"] for document in selected["documents"].values()]
        entry["reviews"] = [
            {
                "review_id": f"{entry['case_id']}-review-a",
                "reviewer_id": "reviewer-a",
                "independent_review_attested": True,
                "scope": "diagnostic_full_triplet",
                "conclusion": "source_matches_manifest_observation",
                "reviewed_document_sha256": image_hashes,
                "annotation_reference": f"reviews/{entry['case_id']}/a",
            },
            {
                "review_id": f"{entry['case_id']}-review-b",
                "reviewer_id": "reviewer-b",
                "independent_review_attested": True,
                "scope": "diagnostic_full_triplet",
                "conclusion": "source_matches_manifest_observation",
                "reviewed_document_sha256": image_hashes,
                "annotation_reference": f"reviews/{entry['case_id']}/b",
            },
        ]
        entry["resolution"] = {
            "method": "two_reviewer_consensus",
            "final_conclusion": "source_matches_manifest_observation",
            "reference": f"resolutions/{entry['case_id']}",
        }
        entry["status"] = "adjudicated"
        entry["gold_eligible"] = True
    gate = evaluate_gate(selection["cases"], worklist)
    worklist_path = tmp_path / "ready-adjudications.jsonl"
    worklist_path.write_text(
        "\n".join(json.dumps(entry, sort_keys=True, separators=(",", ":")) for entry in worklist)
        + "\n",
        encoding="utf-8",
    )
    gate_path = tmp_path / "ready-gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    return selection, worklist, gate, worklist_path, gate_path


def _failure_codes(report):
    return {item["code"] for item in report["failures"]}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release_documents(count):
    return [{"document_id": f"release-document-{index:02d}"} for index in range(count)]


def test_partial_vision_score_is_diagnostic_only_and_never_release_pass():
    all_documents = _release_documents(run_claude_vision.RELEASE_DOCUMENT_COUNT)
    case = {
        "case_ids": [
            f"release-case-{index:02d}" for index in range(run_claude_vision.RELEASE_CASE_COUNT)
        ]
    }

    eligible = run_claude_vision._release_denominator_is_exact(
        requested_case_id=None,
        requested_document_ids=None,
        max_documents=1,
        case=case,
        available_documents=all_documents,
        selected_documents=all_documents[:1],
    )

    assert eligible is False
    assert (
        run_claude_vision._score_status(
            passed=True,
            release_score_eligible=eligible,
        )
        == "DIAGNOSTIC_PASS"
    )


def test_vision_release_pass_requires_exact_12_case_36_document_denominator():
    all_documents = _release_documents(run_claude_vision.RELEASE_DOCUMENT_COUNT)
    case = {
        "case_ids": [
            f"release-case-{index:02d}" for index in range(run_claude_vision.RELEASE_CASE_COUNT)
        ]
    }
    missing_case = {"case_ids": case["case_ids"][:-1]}

    assert (
        run_claude_vision._release_denominator_is_exact(
            requested_case_id=None,
            requested_document_ids=None,
            max_documents=run_claude_vision.RELEASE_DOCUMENT_COUNT,
            case=missing_case,
            available_documents=all_documents,
            selected_documents=all_documents,
        )
        is False
    )
    eligible = run_claude_vision._release_denominator_is_exact(
        requested_case_id=None,
        requested_document_ids=None,
        max_documents=run_claude_vision.RELEASE_DOCUMENT_COUNT,
        case=case,
        available_documents=all_documents,
        selected_documents=all_documents,
    )
    assert eligible is True
    assert (
        run_claude_vision._score_status(
            passed=True,
            release_score_eligible=eligible,
        )
        == "PASS"
    )


def _release_gate_fixture(tmp_path, monkeypatch):
    descriptor_sha256 = "1" * 64
    manifest_sha256 = "2" * 64
    wheel_sha256 = "3" * 64
    environment_lock_path = tmp_path / "paddle.requirements.lock.txt"
    environment_lock_path.write_text(
        "fixture==1.0 --hash=sha256:" + "6" * 64,
        encoding="utf-8",
    )
    environment_lock_sha256 = _sha256(environment_lock_path)
    model_assets_lock_path = tmp_path / "model-assets.lock.json"
    model_assets_lock_path.write_text('{"fixture":"model-assets"}', encoding="utf-8")
    model_assets_lock_sha256 = _sha256(model_assets_lock_path)
    report_path = tmp_path / "ocr-release-gate.json"
    report = {
        "schema_version": "ocr-release-gate-report/v2",
        "benchmark_id": "rendered-trade-documents-ocr-v2",
        "status": "PASS",
        "selection_status": "SELECTED_SINGLE_COMMON_PROVIDER",
        "selected_provider": "paddle",
        "smoke_included": False,
        "release_document_count": 80,
        "descriptor_sha256": descriptor_sha256,
        "dataset_manifest_sha256": manifest_sha256,
        "agent_wheel_sha256": wheel_sha256,
        "environment_lock_sha256": environment_lock_sha256,
        "model_assets_lock_sha256": model_assets_lock_sha256,
        "selected_engine": {
            "provider_id": "paddle",
            "engine_id": "fixture-engine",
            "engine_version": "1.0",
        },
        "input_reports": {
            "locked": {"sha256": "4" * 64, "document_count": 60},
            "co_regression": {"sha256": "5" * 64, "document_count": 20},
        },
        "provider_eligibility": [
            {
                "provider": "paddle",
                "locked_status": "PASS",
                "co_regression_status": "PASS",
                "eligible": True,
            },
            {
                "provider": "docling",
                "locked_status": "FAIL",
                "co_regression_status": "FAIL",
                "eligible": False,
            },
        ],
        "runtime_tiebreak": None,
        "failure_codes": [],
        "redaction": OCR_RELEASE_REDACTION,
    }
    report_path.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        run_claude_vision,
        "load_benchmark",
        lambda **_kwargs: (descriptor_sha256, manifest_sha256, [], []),
    )
    return {
        "ocr_release_gate_report": report_path,
        "ocr_release_gate_report_sha256": _sha256(report_path),
        "ocr_release_descriptor_path": tmp_path / "descriptor.json",
        "ocr_release_rendered_root": tmp_path / "rendered",
        "expected_agent_wheel_sha256": wheel_sha256,
        "ocr_environment_lock": environment_lock_path,
        "ocr_environment_lock_sha256": environment_lock_sha256,
        "ocr_model_assets_lock": model_assets_lock_path,
        "ocr_model_assets_lock_sha256": model_assets_lock_sha256,
        "ocr_python": Path(sys.executable).resolve(),
        "wheel_attestor": lambda _executable: wheel_sha256,
        "environment_attestor": (lambda _executable, _lock: environment_lock_sha256),
        "model_assets_attestor": (lambda _provider, _lock, _engine: model_assets_lock_sha256),
    }


def test_invalidated_case_0500_is_never_scored_or_sent_to_provider(
    monkeypatch,
    tmp_path,
):
    worklist_path, gate_path = _pending_adjudication_evidence(tmp_path)
    monkeypatch.setattr(
        run_claude_vision.ClaudeDocumentAnalyzer,
        "from_settings",
        _forbid_analyzer_creation,
    )

    exit_code, report = run_claude_vision.evaluate(
        env_file=tmp_path / "unused.env",
        dataset_root=tmp_path / "missing-dataset",
        worklist_path=worklist_path,
        gate_path=gate_path,
    )

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["provider_called"] is False
    assert report["release_score_eligible"] is False
    assert report["scope"] == "TRUSTED_REPLACEMENT_GOLD_REQUIRED_HISTORICAL_GOLD_BLOCKED"
    assert report["legacy_case_0500_provenance"] == "invalidated_by_label_error"
    assert report["historical_adjudicated_gold_provenance"] == ("invalidated_by_label_error")
    assert report["replacement_gold_status"] == "not_available"
    assert {
        "legacy_case_0500_invalidated_by_label_error",
        "adjudicated_gold_manifest_required",
        "adjudication_gate_not_gold_ready",
    }.issubset(_failure_codes(report))


def test_tampered_invalidated_annotation_fails_semantic_and_hash_guards(
    monkeypatch,
    tmp_path,
):
    annotation = json.loads(run_claude_vision.DEFAULT_GOLD_PATH.read_text(encoding="utf-8"))
    annotation["cases"][0]["documents"][0]["observed_candidate_fields"]["invoice_no"] = "TAMPERED"
    tampered = tmp_path / "tampered-invalidated-annotation.json"
    tampered.write_text(json.dumps(annotation), encoding="utf-8")
    worklist_path, gate_path = _pending_adjudication_evidence(tmp_path)
    monkeypatch.setattr(
        run_claude_vision.ClaudeDocumentAnalyzer,
        "from_settings",
        _forbid_analyzer_creation,
    )

    exit_code, report = run_claude_vision.evaluate(
        env_file=tmp_path / "unused.env",
        dataset_root=tmp_path / "missing-dataset",
        gold_path=tampered,
        worklist_path=worklist_path,
        gate_path=gate_path,
    )

    assert exit_code == 1
    assert "invalidated_annotation_hash_mismatch" in _failure_codes(report)
    assert "legacy_case_0500_invalidated_by_label_error" in _failure_codes(report)


def test_forged_passed_gate_cannot_override_pending_worklist(
    monkeypatch,
    tmp_path,
):
    worklist_path, gate_path = _pending_adjudication_evidence(tmp_path)
    forged_gate = {
        "schema_version": "trade-image-diagnostic-gate/v1",
        "selection_id": "trade-images-hash-ranked-diagnostic-v1",
        "status": "passed_two_reviewer_adjudication",
        "gold_eligible": True,
        "selected_case_count": 12,
        "required_independent_review_count_per_case": 2,
        "status_counts": {"adjudicated": 12},
        "blockers": [],
    }
    gate_path.write_text(json.dumps(forged_gate), encoding="utf-8")
    monkeypatch.setattr(
        run_claude_vision.ClaudeDocumentAnalyzer,
        "from_settings",
        _forbid_analyzer_creation,
    )

    exit_code, report = run_claude_vision.evaluate(
        env_file=tmp_path / "unused.env",
        dataset_root=tmp_path / "missing-dataset",
        worklist_path=worklist_path,
        gate_path=gate_path,
    )

    assert exit_code == 1
    assert {
        "adjudication_gate_evidence_mismatch",
        "adjudication_gate_not_gold_ready",
    }.issubset(_failure_codes(report))


def test_ready_gate_still_cannot_revalidate_legacy_case_0500(
    monkeypatch,
    tmp_path,
):
    _, _, _, worklist_path, gate_path = _ready_adjudication_evidence(tmp_path)
    monkeypatch.setattr(
        run_claude_vision.ClaudeDocumentAnalyzer,
        "from_settings",
        _forbid_analyzer_creation,
    )

    exit_code, report = run_claude_vision.evaluate(
        env_file=tmp_path / "unused.env",
        dataset_root=tmp_path / "missing-dataset",
        worklist_path=worklist_path,
        gate_path=gate_path,
    )

    assert exit_code == 1
    assert "adjudication_gate_not_gold_ready" not in _failure_codes(report)
    assert {
        "legacy_case_0500_invalidated_by_label_error",
        "adjudicated_gold_manifest_required",
    }.issubset(_failure_codes(report))
    assert report["provider_called"] is False


def test_future_gold_contract_binds_selection_worklist_gate_and_source_hashes(
    tmp_path,
):
    (
        selection,
        worklist,
        gate,
        worklist_path,
        gate_path,
    ) = _ready_adjudication_evidence(tmp_path)
    document_type_map = {
        "commercial_invoice": "invoice",
        "packing_list": "packing_list",
        "bill_of_lading": "bill_of_lading",
    }
    gold_cases = []
    worklist_by_id = {entry["case_id"]: entry for entry in worklist}
    for selected in selection["cases"]:
        case_id = selected["case_id"]
        gold_cases.append(
            {
                "case_id": case_id,
                "triplet_fingerprint_sha256": selected["triplet_fingerprint_sha256"],
                "gold_eligible": True,
                "adjudication_status": "adjudicated",
                "annotation_references": [
                    review["annotation_reference"] for review in worklist_by_id[case_id]["reviews"]
                ],
                "documents": [
                    {
                        "document_id": f"{case_id}:{document_type}",
                        "source_document_type": document_type,
                        "relative_path": document["relative_path"],
                        "source_image_sha256": document["sha256"],
                        "expected_document_type": document_type_map[document_type],
                        "expected_fields": {"test_contract_field": "SYNTHETIC"},
                        "expected_line_item_count": 0,
                    }
                    for document_type, document in selected["documents"].items()
                ],
            }
        )
    gold = {
        "schema_version": "trade-image-adjudicated-gold/v1",
        "gold_set_id": "unit-test-adjudicated-gold",
        "selection_id": selection["selection_id"],
        "selection_sha256": _sha256(run_claude_vision.DEFAULT_SELECTION_PATH),
        "adjudication_worklist_sha256": _sha256(worklist_path),
        "adjudication_gate_sha256": _sha256(gate_path),
        "gold_eligible": True,
        "release_scoring_scope": "claude_vision_field_extraction_only",
        "cases": gold_cases,
    }

    assert (
        run_claude_vision._validate_gold_gate(
            selection=selection,
            worklist=worklist,
            gate=gate,
        )
        == []
    )
    assert (
        run_claude_vision._validate_gold_manifest(
            gold=gold,
            selection=selection,
            worklist=worklist,
            selection_path=run_claude_vision.DEFAULT_SELECTION_PATH,
            worklist_path=worklist_path,
            gate_path=gate_path,
            schema_path=(run_claude_vision.DEFAULT_ADJUDICATED_GOLD_SCHEMA_PATH),
        )
        == []
    )


def test_tampered_vision_prompt_hash_fails_before_provider_setup(
    monkeypatch,
    tmp_path,
):
    worklist_path, gate_path = _pending_adjudication_evidence(tmp_path)
    monkeypatch.setattr(
        run_claude_vision,
        "_vision_prompt_sha256",
        lambda: "0" * 64,
    )
    monkeypatch.setattr(
        run_claude_vision.ClaudeDocumentAnalyzer,
        "from_settings",
        _forbid_analyzer_creation,
    )

    exit_code, report = run_claude_vision.evaluate(
        env_file=tmp_path / "unused.env",
        dataset_root=tmp_path / "missing-dataset",
        worklist_path=worklist_path,
        gate_path=gate_path,
    )

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert "prompt_hash_mismatch" in _failure_codes(report)


def test_runtime_vision_prompt_change_remains_unapproved_and_fail_closed():
    assert run_claude_vision.EXPECTED_VISION_PROMPT_SHA256 == (
        "531ad80b62016c5bd65ea292ac584f36413dfb230b13cffb5761b6862d1705ba"
    )
    assert run_claude_vision._vision_prompt_sha256() == (
        "8d64bfa5673f90e092e56740e732ba07012e566fffec154457b59ca35590f3c8"
    )
    assert (
        run_claude_vision._vision_prompt_sha256() != run_claude_vision.EXPECTED_VISION_PROMPT_SHA256
    )


def test_unpinned_replacement_blocks_adjudicated_gold_after_reserialization(
    monkeypatch,
    tmp_path,
):
    worklist_path, gate_path = _pending_adjudication_evidence(tmp_path)
    monkeypatch.setattr(
        run_claude_vision,
        "EXPECTED_ADJUDICATED_GOLD_SHA256",
        None,
    )
    monkeypatch.setattr(
        run_claude_vision.ClaudeDocumentAnalyzer,
        "from_settings",
        _forbid_analyzer_creation,
    )
    gold = {
        "schema_version": "trade-image-adjudicated-gold/v1",
        "gold_set_id": "untrusted-candidate",
        "cases": [],
    }
    gold_paths = [
        tmp_path / "compact-gold.json",
        tmp_path / "pretty-gold.json",
    ]
    gold_paths[0].write_text(
        json.dumps(gold, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    gold_paths[1].write_text(
        json.dumps(gold, indent=4, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    assert _sha256(gold_paths[0]) != _sha256(gold_paths[1])

    reports = []
    for gold_path in gold_paths:
        exit_code, report = run_claude_vision.evaluate(
            env_file=tmp_path / "unused.env",
            dataset_root=tmp_path / "missing-dataset",
            gold_path=gold_path,
            worklist_path=worklist_path,
            gate_path=gate_path,
        )
        assert exit_code == 1
        assert report["provider_called"] is False
        assert "adjudicated_gold_replacement_not_provisioned" in _failure_codes(report)
        reports.append(report)

    assert reports[0]["gold_manifest_sha256"] != reports[1]["gold_manifest_sha256"]


def test_workflow_flag_is_blocked_without_separate_outcome_gold(
    monkeypatch,
    tmp_path,
):
    worklist_path, gate_path = _pending_adjudication_evidence(tmp_path)
    monkeypatch.setattr(
        run_claude_vision.ClaudeDocumentAnalyzer,
        "from_settings",
        _forbid_analyzer_creation,
    )

    exit_code, report = run_claude_vision.evaluate(
        env_file=tmp_path / "unused.env",
        dataset_root=tmp_path / "missing-dataset",
        worklist_path=worklist_path,
        gate_path=gate_path,
        run_workflow=True,
    )

    assert exit_code == 1
    assert "workflow_scoring_requires_separate_outcome_gold" in _failure_codes(report)


def test_adjudicated_gold_schema_is_valid_and_paths_are_repository_relative():
    repository_root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        run_claude_vision.DEFAULT_ADJUDICATED_GOLD_SCHEMA_PATH.read_text(encoding="utf-8")
    )

    Draft202012Validator.check_schema(schema)
    assert run_claude_vision.DEFAULT_GOLD_PATH.is_relative_to(repository_root)
    assert run_claude_vision.DEFAULT_LOCK_PATH.is_relative_to(repository_root)
    assert run_claude_vision.DEFAULT_SELECTION_PATH.is_relative_to(repository_root)
    assert run_claude_vision.DEFAULT_ADJUDICATED_GOLD_SCHEMA_PATH.is_relative_to(repository_root)


def test_ready_gold_without_explicit_live_flag_cannot_create_ocr_or_claude_provider(
    monkeypatch,
    tmp_path,
):
    image_path = tmp_path / "document.png"
    Image.new("RGB", (120, 80), "white").save(image_path)
    case = {
        "case_id": "diagnostic-case",
        "documents": [
            {
                "document_id": "diagnostic-case:invoice",
                "relative_path": image_path.name,
            }
        ],
    }
    monkeypatch.setattr(
        run_claude_vision,
        "_preflight",
        lambda **_kwargs: ([], case, {}, {}),
    )
    monkeypatch.setattr(
        run_claude_vision,
        "build_ocr_layout_provider",
        _forbid_analyzer_creation,
    )
    monkeypatch.setattr(
        run_claude_vision.ClaudeDocumentAnalyzer,
        "from_settings",
        _forbid_analyzer_creation,
    )

    exit_code, report = run_claude_vision.evaluate(
        env_file=tmp_path / "unused.env",
        dataset_root=tmp_path,
        max_documents=1,
        live=False,
        ocr_provider="paddle",
        ocr_python=tmp_path / "paddle-python.exe",
        max_calls=7,
        max_total_input_tokens=100_000,
        max_total_output_tokens=100_000,
    )

    assert exit_code == 1
    assert report["provider_called"] is False
    assert report["failures"] == [{"document_id": "preflight", "code": "live_flag_required"}]


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("tampered_hash", "ocr_release_gate_report_hash_mismatch"),
        ("old_single_sample", "ocr_release_gate_report_invalid"),
        ("combined_fail", "ocr_release_gate_status_not_passing"),
        ("quality_tie", "ocr_release_gate_status_not_passing"),
        ("no_provider", "ocr_release_gate_status_not_passing"),
        ("provider", "ocr_release_gate_provider_mismatch"),
        ("engine", "ocr_release_gate_engine_mismatch"),
        ("wheel", "ocr_release_gate_expected_wheel_mismatch"),
        (
            "environment_lock",
            "ocr_release_gate_environment_lock_hash_mismatch",
        ),
        (
            "model_assets_lock",
            "ocr_release_gate_model_assets_lock_hash_mismatch",
        ),
        (
            "selected_sample_failure",
            "ocr_release_gate_selected_provider_not_passing",
        ),
        ("installed_wheel", "ocr_release_gate_installed_wheel_mismatch"),
    ],
)
def test_release_gate_defects_block_vision_provider_and_claude(
    monkeypatch,
    tmp_path,
    mutation,
    expected_code,
):
    image_path = tmp_path / "document.png"
    Image.new("RGB", (120, 80), "white").save(image_path)
    case = {
        "case_id": "diagnostic-case",
        "documents": [
            {
                "document_id": "diagnostic-case:invoice",
                "relative_path": image_path.name,
            }
        ],
    }
    monkeypatch.setattr(
        run_claude_vision,
        "_preflight",
        lambda **_kwargs: ([], case, {}, {}),
    )
    release_args = _release_gate_fixture(tmp_path, monkeypatch)
    report_path = release_args["ocr_release_gate_report"]
    release_report = json.loads(report_path.read_text(encoding="utf-8"))
    if mutation == "tampered_hash":
        release_args["ocr_release_gate_report_sha256"] = "0" * 64
    elif mutation == "old_single_sample":
        release_report = {
            "schema_version": "ocr-benchmark-report/v1",
            "benchmark_id": "rendered-trade-documents-ocr-v1",
            "sample": "locked",
            "status": "PASS",
            "selected_provider": "paddle",
        }
    elif mutation == "combined_fail":
        release_report["status"] = "FAIL"
    elif mutation == "quality_tie":
        release_report["status"] = "FAIL"
        release_report["selection_status"] = "QUALITY_TIE_REQUIRES_RUNTIME_BENCHMARK"
        release_report["selected_provider"] = None
        release_report["selected_engine"] = None
        release_report["failure_codes"] = ["quality_tie_requires_runtime_benchmark"]
    elif mutation == "no_provider":
        release_report["status"] = "FAIL"
        release_report["selection_status"] = "NO_PROVIDER_PASSED_BOTH_RELEASE_SAMPLES"
        release_report["selected_provider"] = None
        release_report["selected_engine"] = None
        release_report["failure_codes"] = ["no_provider_passed_both_release_samples"]
    elif mutation == "provider":
        release_report["selected_provider"] = "docling"
        release_report["selected_engine"]["provider_id"] = "docling"
        release_report["provider_eligibility"] = [
            {
                "provider": "paddle",
                "locked_status": "FAIL",
                "co_regression_status": "FAIL",
                "eligible": False,
            },
            {
                "provider": "docling",
                "locked_status": "PASS",
                "co_regression_status": "PASS",
                "eligible": True,
            },
        ]
    elif mutation == "engine":
        release_report["selected_engine"]["provider_id"] = "docling"
    elif mutation == "wheel":
        release_report["agent_wheel_sha256"] = "6" * 64
    elif mutation == "environment_lock":
        release_report["environment_lock_sha256"] = "6" * 64
    elif mutation == "model_assets_lock":
        release_report["model_assets_lock_sha256"] = "6" * 64
    elif mutation == "selected_sample_failure":
        release_report["provider_eligibility"][0]["locked_status"] = "FAIL"
        release_report["provider_eligibility"][0]["eligible"] = False
    elif mutation == "installed_wheel":
        release_args["wheel_attestor"] = lambda _executable: "6" * 64

    if mutation not in {"tampered_hash", "installed_wheel"}:
        report_path.write_text(
            json.dumps(release_report, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        release_args["ocr_release_gate_report_sha256"] = _sha256(report_path)
    counters = {"provider": 0, "analyzer": 0}

    def provider_factory(_provider, _executable):
        counters["provider"] += 1
        return object()

    def analyzer_factory(_settings, _provider):
        counters["analyzer"] += 1
        raise AssertionError("Claude analyzer must not be created")

    exit_code, report = run_claude_vision.evaluate(
        env_file=tmp_path / "unused.env",
        dataset_root=tmp_path,
        max_documents=1,
        live=True,
        ocr_provider="paddle",
        **release_args,
        max_calls=7,
        max_total_input_tokens=100_000,
        max_total_output_tokens=100_000,
        provider_factory=provider_factory,
        analyzer_factory=analyzer_factory,
    )

    assert exit_code == 1
    assert report["failures"] == [{"document_id": "preflight", "code": expected_code}]
    assert report["provider_called"] is False
    assert counters == {"provider": 0, "analyzer": 0}


def test_runtime_ocr_engine_drift_stops_before_vision_transport(
    monkeypatch,
    tmp_path,
):
    image_path = tmp_path / "document.png"
    Image.new("RGB", (120, 80), "white").save(image_path)
    case = {
        "case_id": "diagnostic-case",
        "documents": [
            {
                "document_id": "diagnostic-case:invoice",
                "relative_path": image_path.name,
                "expected_document_type": "invoice",
                "expected_fields": {},
                "expected_line_item_count": 0,
            }
        ],
    }
    dataset_lock = {
        "dataset_id": "unit-test-dataset",
        "dataset_version": "v1",
        "status": "synthetic",
        "unsupported_claims": ["not_system_release_evidence"],
    }
    monkeypatch.setattr(
        run_claude_vision,
        "_preflight",
        lambda **_kwargs: ([], case, dataset_lock, {}),
    )
    release_args = _release_gate_fixture(tmp_path, monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ANTHROPIC_API_KEY=unit-test-placeholder\nCUSTOMS_AGENT_CLAUDE_MAX_TOKENS=1024\n",
        encoding="utf-8",
    )
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(
        json.dumps({"gold_set_id": "unit-test-gold"}),
        encoding="utf-8",
    )
    external_calls = 0

    class WrongEngineProvider:
        def analyze(self, *, document_id, pages):
            page = OcrLayoutPage(
                page_number=1,
                image_sha256=pages[0].sha256,
                width_px=pages[0].width_px,
                height_px=pages[0].height_px,
                spans=[
                    OcrLayoutSpan(
                        span_id="span-1",
                        page=1,
                        reading_order=0,
                        role="text",
                        text="synthetic",
                        bbox=BBox(page=1, x1=0, y1=0, x2=10, y2=10),
                        confidence=1.0,
                    )
                ],
            )
            ledger_id = stable_ocr_ledger_id(
                document_id=document_id,
                provider_id="paddle",
                engine_id="fixture-engine",
                engine_version="2.0",
                pages=[page],
            )
            return OcrLayoutLedger(
                ledger_id=ledger_id,
                document_id=document_id,
                provider_id="paddle",
                engine_id="fixture-engine",
                engine_version="2.0",
                pages=[page],
            )

    class OcrThenClaudeAnalyzer:
        model_id = "fixture-model"

        def __init__(self, provider):
            self.provider = provider

        def analyze(self, document):
            nonlocal external_calls
            try:
                self.provider.analyze(
                    document_id=str(document["document_id"]),
                    pages=[ImagePageInput.model_validate(page) for page in document["image_pages"]],
                )
            except OcrLayoutProviderError as exc:
                raise LLMError(exc.code, "runtime OCR identity rejected") from None
            external_calls += 1
            raise AssertionError("Claude transport must not be reached")

    exit_code, report = run_claude_vision.evaluate(
        env_file=env_file,
        dataset_root=tmp_path,
        gold_path=gold_path,
        max_documents=1,
        live=True,
        ocr_provider="paddle",
        **release_args,
        max_calls=7,
        max_total_input_tokens=100_000,
        max_total_output_tokens=10_000,
        provider_factory=lambda _provider, _executable: WrongEngineProvider(),
        analyzer_factory=lambda _settings, provider: OcrThenClaudeAnalyzer(provider),
    )

    assert exit_code == 1
    assert report["failures"] == [
        {
            "document_id": "diagnostic-case:invoice",
            "code": "ocr_release_gate_runtime_engine_mismatch",
        }
    ]
    assert external_calls == 0


def test_failed_vision_call_consumes_budget_without_sdk_retry(
    monkeypatch,
    tmp_path,
):
    image_path = tmp_path / "document.png"
    Image.new("RGB", (120, 80), "white").save(image_path)
    case = {
        "case_id": "diagnostic-case",
        "documents": [
            {
                "document_id": "diagnostic-case:invoice",
                "relative_path": image_path.name,
                "expected_document_type": "invoice",
                "expected_fields": {},
                "expected_line_item_count": 0,
            }
        ],
    }
    dataset_lock = {
        "dataset_id": "unit-test-dataset",
        "dataset_version": "v1",
        "status": "synthetic",
        "unsupported_claims": ["not_system_release_evidence"],
    }
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ANTHROPIC_API_KEY=unit-test-placeholder\n"
        "CUSTOMS_AGENT_CLAUDE_MAX_TOKENS=1024\n"
        "CUSTOMS_AGENT_CLAUDE_MAX_RETRIES=3\n",
        encoding="utf-8",
    )
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(
        json.dumps({"gold_set_id": "unit-test-gold"}),
        encoding="utf-8",
    )
    captured_settings = []

    class FailingAnalyzer:
        def analyze(self, _document):
            raise LLMError("provider_unavailable", "simulated provider failure")

    def analyzer_factory(settings, *, ocr_layout_provider):
        captured_settings.append((settings, ocr_layout_provider))
        return FailingAnalyzer()

    provider = object()
    monkeypatch.setattr(
        run_claude_vision,
        "_preflight",
        lambda **_kwargs: ([], case, dataset_lock, {}),
    )
    monkeypatch.setattr(
        run_claude_vision,
        "build_ocr_layout_provider",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr(
        run_claude_vision.ClaudeDocumentAnalyzer,
        "from_settings",
        staticmethod(analyzer_factory),
    )
    release_args = _release_gate_fixture(tmp_path, monkeypatch)

    exit_code, report = run_claude_vision.evaluate(
        env_file=env_file,
        dataset_root=tmp_path,
        gold_path=gold_path,
        max_documents=1,
        live=True,
        ocr_provider="paddle",
        **release_args,
        max_calls=7,
        max_total_input_tokens=100_000,
        max_total_output_tokens=10_000,
    )

    assert exit_code == 1
    assert report["failures"] == [
        {
            "document_id": "diagnostic-case:invoice",
            "code": "provider_unavailable",
        }
    ]
    assert report["call_accounting"] == {
        "attempted": 1,
        "succeeded": 0,
        "retried": 0,
        "main_attempted": 1,
        "crop_attempted": 0,
    }
    assert report["provider_called"] is True
    assert report["cost_budget"]["configured_max_retries"] == 0
    assert len(captured_settings) == 1
    assert captured_settings[0][0].max_retries == 0
    assert captured_settings[0][1]._provider is provider
