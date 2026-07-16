from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from customs_ai_agent.adapters import ensure_llm_mode_available
from customs_ai_agent.adapters.retrieval import (
    HttpRegulationRetriever,
    fetch_supported_review_as_of,
)
from customs_ai_agent.fixtures.scenarios import list_scenarios
from customs_ai_agent.runner import answer_question, run_review
from customs_ai_agent.schemas.models import ReviewRunRequest


def _cmd_list_scenarios() -> int:
    for scenario in list_scenarios():
        print(scenario)
    return 0


def _configured_rag_url(cli_value: str | None) -> str | None:
    if cli_value is not None:
        return cli_value
    configured = os.getenv("EVIDENCE_SERVICE_BASE_URL")
    if configured is None:
        return None
    if not configured.strip():
        raise ValueError("EVIDENCE_SERVICE_BASE_URL cannot be blank when configured")
    return configured


def _configured_service_token() -> str | None:
    configured = os.getenv("EVIDENCE_SERVICE_TOKEN")
    return configured if configured else None


def _parse_review_as_of(value: str) -> date:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise argparse.ArgumentTypeError("review date must use YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("review date must use YYYY-MM-DD") from exc


def _cmd_run(args: argparse.Namespace) -> int:
    # Keep the intentionally unavailable LLM mode fail-fast and side-effect free.
    ensure_llm_mode_available(args.llm_mode)
    load_dotenv()
    rag_url = _configured_rag_url(getattr(args, "rag_url", None))
    review_as_of = getattr(args, "review_as_of", None)
    retriever = None
    if rag_url is not None:
        service_token = _configured_service_token()
        token_header = os.getenv("EVIDENCE_SERVICE_TOKEN_HEADER", "X-Demo-Token")
        retriever = HttpRegulationRetriever(
            rag_url,
            service_token=service_token,
            token_header=token_header,
        )
        if review_as_of is None:
            review_as_of = fetch_supported_review_as_of(
                rag_url,
                service_token=service_token,
                token_header=token_header,
            )
    request_values = {
        "scenario_id": args.scenario,
        "llm_mode": args.llm_mode,
    }
    if review_as_of is not None:
        request_values["review_as_of"] = review_as_of
    request = ReviewRunRequest.model_validate(request_values)
    state = run_review(request, retriever=retriever)
    payload = json.dumps(state, ensure_ascii=False, indent=2, default=str)

    report = state.get("report_json") or {}
    summary = report.get("summary") or {}
    print(f"case_id: {state['case_id']}")
    print(f"status: {report.get('overall_status', state.get('final_status'))}")
    print(f"issues: {summary.get('total_issues', 0)}")
    print(f"requires_human_review: {state.get('requires_human_review', False)}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"wrote: {out_path}")
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    report_path = Path(args.report)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    report = payload.get("report_json", payload)
    answer = answer_question(report, args.question)
    print(json.dumps(answer.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="customs-ai-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-scenarios")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--scenario", required=True)
    run_parser.add_argument("--out")
    run_parser.add_argument("--llm-mode", choices=["mock", "llm"], default="mock")
    run_parser.add_argument(
        "--review-as-of",
        type=_parse_review_as_of,
        help=(
            "Review date in YYYY-MM-DD. For Live RAG, omission uses the date "
            "advertised by the Evidence Service readiness endpoint."
        ),
    )
    run_parser.add_argument(
        "--rag-url",
        help=(
            "Evidence Service base URL; calls POST /internal/rag/v1/retrieve. "
            "Overrides EVIDENCE_SERVICE_BASE_URL. If neither is configured, uses "
            "the deterministic mock retriever. A failed live call never falls back."
        ),
    )

    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("--report", required=True)
    chat_parser.add_argument("--question", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list-scenarios":
        return _cmd_list_scenarios()
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "chat":
        return _cmd_chat(args)
    parser.error(f"unknown command: {args.command}")
    return 2
