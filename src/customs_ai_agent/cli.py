from __future__ import annotations

import argparse
import json
from pathlib import Path

from customs_ai_agent.adapters.retrieval import HttpRegulationRetriever
from customs_ai_agent.fixtures.scenarios import list_scenarios
from customs_ai_agent.runner import answer_question, run_review
from customs_ai_agent.schemas.models import ReviewRunRequest


def _cmd_list_scenarios() -> int:
    for scenario in list_scenarios():
        print(scenario)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    request = ReviewRunRequest(scenario_id=args.scenario, llm_mode=args.llm_mode)
    rag_url = getattr(args, "rag_url", None)
    retriever = HttpRegulationRetriever(rag_url) if rag_url else None
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
        "--rag-url",
        help=(
            "Evidence Service base URL; calls POST /internal/rag/v1/retrieve. "
            "Omit to use the deterministic mock retriever."
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
