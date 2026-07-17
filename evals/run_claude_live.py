"""Opt-in Claude evaluation using synthetic repository fixtures only.

The report contains aggregate metrics and provider metadata. It deliberately
omits API keys, document text, extracted values, prompts, and raw responses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from customs_ai_agent.adapters.llm import (
    DEFAULT_CLAUDE_MODEL,
    ClaudeDocumentAnalyzer,
    DocumentAnalysis,
    LLMError,
)
from customs_ai_agent.adapters.retrieval import MockRegulationRetriever
from customs_ai_agent.fixtures.scenarios import get_scenario
from customs_ai_agent.prompts import (
    DOCUMENT_ANALYSIS_PROMPT_ID,
    DOCUMENT_ANALYSIS_PROMPT_VERSION,
    load_document_analysis_system_prompt,
)
from customs_ai_agent.runner import run_review
from customs_ai_agent.schemas.llm import ClaudeDocumentCandidate
from customs_ai_agent.schemas.models import ReviewRunRequest


EXPECTED_GOLD_SHA256_BY_ID = {
    "claude-text-synthetic-v1": (
        "2e8b95b850d7a4ea208ca49903982bdd44244d732bda9b11dc946056784a3a09"
    ),
    "claude-text-adversarial-v1": (
        "8e4da60e14b783fcc43e17cf388a45818d552e798f0406c3729b24d4ec8b8081"
    ),
}
EXPECTED_CANDIDATE_SCHEMA_SHA256 = (
    "049d9c0a4782945f6b6716d1f9d8ad02a407e88ce7fa0beaa2e2cd5575312b5c"
)
EXPECTED_PROMPT_SHA256 = (
    "58794169eff72ca4727dc59d5d002807424677024016402f9fb0686d542ec8af"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_schema_sha256() -> str:
    canonical = json.dumps(
        ClaudeDocumentCandidate.model_json_schema(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _prompt_sha256() -> str:
    return hashlib.sha256(
        load_document_analysis_system_prompt().encode("utf-8")
    ).hexdigest()


def _field_map(items: list[dict[str, Any]]) -> dict[str, str]:
    return {str(item["name"]): str(item["value"]) for item in items}


def _compare_fields(
    actual: dict[str, str],
    expected: dict[str, str],
) -> dict[str, int]:
    exact = sum(actual.get(name) == value for name, value in expected.items())
    wrong = sum(
        name in actual and actual[name] != value for name, value in expected.items()
    )
    missing = sum(name not in actual for name in expected)
    hallucinated = sum(name not in expected for name in actual)
    return {
        "expected": len(expected),
        "exact": exact,
        "wrong": wrong,
        "missing": missing,
        "hallucinated": hallucinated,
    }


def _add(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {name: left.get(name, 0) + right.get(name, 0) for name in right}


def evaluate(
    *,
    env_file: Path,
    gold_path: Path,
    max_documents: int,
    run_workflow: bool,
) -> tuple[int, dict[str, Any]]:
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold_sha256 = _sha256(gold_path)
    expected_gold_sha256 = EXPECTED_GOLD_SHA256_BY_ID.get(
        str(gold["gold_set_id"]),
        "NOT_CONFIGURED",
    )
    candidate_schema_sha256 = _candidate_schema_sha256()
    prompt_sha256 = _prompt_sha256()
    preflight_failures = []
    if gold_sha256 != expected_gold_sha256:
        preflight_failures.append("gold_set_hash_mismatch")
    if candidate_schema_sha256 != EXPECTED_CANDIDATE_SCHEMA_SHA256:
        preflight_failures.append("candidate_schema_hash_mismatch")
    if prompt_sha256 != EXPECTED_PROMPT_SHA256:
        preflight_failures.append("prompt_hash_mismatch")
    if preflight_failures:
        return 1, {
            "schema_version": "1.0",
            "scope": "SYNTHETIC_TEXT_ONLY_NOT_PDF_OCR_OR_RELEASE",
            "gold_set_id": gold["gold_set_id"],
            "gold_set_sha256": gold_sha256,
            "expected_gold_set_sha256": expected_gold_sha256,
            "candidate_schema_sha256": candidate_schema_sha256,
            "expected_candidate_schema_sha256": EXPECTED_CANDIDATE_SCHEMA_SHA256,
            "prompt_sha256": prompt_sha256,
            "expected_prompt_sha256": EXPECTED_PROMPT_SHA256,
            "status": "FAIL",
            "failures": [
                {"document_id": "preflight", "code": code}
                for code in preflight_failures
            ],
            "redaction": {
                "api_key": "OMITTED",
                "document_text": "OMITTED",
                "extracted_values": "OMITTED",
                "prompt": "OMITTED",
                "raw_response": "OMITTED",
            },
        }
    scenario = get_scenario(str(gold["scenario_id"]))
    documents = scenario["documents"][:max_documents]
    expected_documents = gold["documents"]
    analyzer = ClaudeDocumentAnalyzer.from_environment(env_file)
    analyses: dict[str, DocumentAnalysis] = {}
    field_totals = {
        "expected": 0,
        "exact": 0,
        "wrong": 0,
        "missing": 0,
        "hallucinated": 0,
    }
    type_correct = 0
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "latency_ms": 0,
    }
    calls: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    workflow_summary: dict[str, Any] | None = None

    if run_workflow:
        if max_documents != 4:
            raise ValueError("--workflow requires --max-documents 4")

        class RecordingAnalyzer:
            def analyze(self, document: Mapping[str, Any]) -> DocumentAnalysis:
                result = analyzer.analyze(document)
                analyses[str(document["document_id"])] = result
                return result

        state = run_review(
            ReviewRunRequest(
                case_id=str(scenario["case_id"]),
                documents=documents,
                llm_mode="llm",
            ),
            retriever=MockRegulationRetriever(),
            analyzer=RecordingAnalyzer(),
        )
        failures.extend(
            {
                "document_id": str(item["document_id"]),
                "code": str(item["code"]),
            }
            for item in state["model_errors"]
        )
        workflow_summary = {
            "retrieval_mode": "EXPLICIT_MOCK_NOT_LIVE_RAG",
            "status": state["final_status"],
            "requires_human_review": state["requires_human_review"],
            "validation_issue_count": len(state["validation_issues"]),
            "model_call_count": len(state["model_calls"]),
            "model_error_count": len(state["model_errors"]),
            "retrieval_status": state["retrieval_status"],
            "retrieval_coverage": state["retrieval_coverage"],
            "knowledge_snapshot_id": state["knowledge_snapshot_id"],
            "regulatory_finding_count": len(state["regulatory_findings"]),
            "evidence_count": len(state["rag_evidence"]),
        }

    for document in documents:
        document_id = str(document["document_id"])
        analysis = analyses.get(document_id)
        if analysis is None and not run_workflow:
            try:
                analysis = analyzer.analyze(document)
            except LLMError as exc:
                failures.append({"document_id": document_id, "code": exc.code})
                continue
        if analysis is None:
            continue
        expected = expected_documents[document_id]
        candidate = analysis.candidate.model_dump(mode="json")
        type_correct += int(candidate["document_type"] == expected["document_type"])
        field_totals = _add(
            field_totals,
            _compare_fields(_field_map(candidate["fields"]), expected["fields"]),
        )

        actual_lines = candidate["line_items"]
        expected_lines = expected["line_items"]
        paired_count = min(len(actual_lines), len(expected_lines))
        for index in range(paired_count):
            field_totals = _add(
                field_totals,
                _compare_fields(
                    _field_map(actual_lines[index]["fields"]),
                    expected_lines[index],
                ),
            )
        for expected_line in expected_lines[paired_count:]:
            field_totals["expected"] += len(expected_line)
            field_totals["missing"] += len(expected_line)
        for actual_line in actual_lines[paired_count:]:
            field_totals["hallucinated"] += len(actual_line["fields"])

        metadata = analysis.metadata
        usage["input_tokens"] += metadata.input_tokens
        usage["output_tokens"] += metadata.output_tokens
        usage["cache_creation_input_tokens"] += metadata.cache_creation_input_tokens
        usage["cache_read_input_tokens"] += metadata.cache_read_input_tokens
        usage["latency_ms"] += metadata.latency_ms
        calls.append(
            {
                "document_id": document_id,
                "provider": metadata.provider,
                "model_id": metadata.model_id,
                "prompt_id": metadata.prompt_id,
                "prompt_version": metadata.prompt_version,
                "request_id": metadata.request_id,
                "message_id": metadata.message_id,
                "stop_reason": metadata.stop_reason,
                "input_tokens": metadata.input_tokens,
                "output_tokens": metadata.output_tokens,
                "latency_ms": metadata.latency_ms,
            }
        )

    documents_attempted = len(documents)
    passed = (
        not failures
        and gold_sha256 == expected_gold_sha256
        and candidate_schema_sha256 == EXPECTED_CANDIDATE_SCHEMA_SHA256
        and prompt_sha256 == EXPECTED_PROMPT_SHA256
        and len(calls) == documents_attempted
        and all(
            call["provider"] == "anthropic"
            and call["model_id"] == DEFAULT_CLAUDE_MODEL
            and call["prompt_id"] == DOCUMENT_ANALYSIS_PROMPT_ID
            and call["prompt_version"] == DOCUMENT_ANALYSIS_PROMPT_VERSION
            for call in calls
        )
        and type_correct == documents_attempted
        and field_totals["exact"] == field_totals["expected"]
        and field_totals["wrong"] == 0
        and field_totals["missing"] == 0
        and field_totals["hallucinated"] == 0
        and (
            workflow_summary is None
            or (
                workflow_summary["status"] == "needs_review"
                and workflow_summary["requires_human_review"] is True
                and workflow_summary["validation_issue_count"] == 0
                and workflow_summary["model_call_count"] == documents_attempted
                and workflow_summary["model_error_count"] == 0
                and workflow_summary["retrieval_status"] == "ok"
                and workflow_summary["retrieval_coverage"] == "complete"
            )
        )
    )
    report = {
        "schema_version": "1.0",
        "scope": "SYNTHETIC_TEXT_ONLY_NOT_PDF_OCR_OR_RELEASE",
        "gold_set_id": gold["gold_set_id"],
        "gold_set_sha256": gold_sha256,
        "expected_gold_set_sha256": expected_gold_sha256,
        "candidate_schema_id": "claude-document-candidate",
        "candidate_schema_version": "1.0",
        "candidate_schema_sha256": candidate_schema_sha256,
        "expected_candidate_schema_sha256": EXPECTED_CANDIDATE_SCHEMA_SHA256,
        "prompt_id": DOCUMENT_ANALYSIS_PROMPT_ID,
        "prompt_version": DOCUMENT_ANALYSIS_PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
        "expected_prompt_sha256": EXPECTED_PROMPT_SHA256,
        "documents_attempted": documents_attempted,
        "documents_completed": len(calls),
        "document_type_correct": type_correct,
        "field_metrics": field_totals,
        "usage": usage,
        "calls": calls,
        "failures": failures,
        "workflow": workflow_summary,
        "status": "PASS" if passed else "FAIL",
        "redaction": {
            "api_key": "OMITTED",
            "document_text": "OMITTED",
            "extracted_values": "OMITTED",
            "prompt": "OMITTED",
            "raw_response": "OMITTED",
        },
    }
    return (0 if passed else 1), report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path(__file__).with_name("claude_text_gold_v1.json"),
    )
    parser.add_argument("--max-documents", type=int, choices=range(1, 5), default=1)
    parser.add_argument(
        "--workflow",
        action="store_true",
        help=(
            "Run the four Claude analyses through the full deterministic workflow "
            "with an explicitly labelled Mock RegulationRetriever."
        ),
    )
    parser.add_argument("--out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        exit_code, report = evaluate(
            env_file=args.env_file,
            gold_path=args.gold,
            max_documents=args.max_documents,
            run_workflow=args.workflow,
        )
    except LLMError as exc:
        exit_code = 1
        report = {
            "schema_version": "1.0",
            "scope": "SYNTHETIC_TEXT_ONLY_NOT_PDF_OCR_OR_RELEASE",
            "status": "FAIL",
            "failures": [{"document_id": "preflight", "code": exc.code}],
            "redaction": {
                "api_key": "OMITTED",
                "document_text": "OMITTED",
                "extracted_values": "OMITTED",
                "prompt": "OMITTED",
                "raw_response": "OMITTED",
            },
        }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
