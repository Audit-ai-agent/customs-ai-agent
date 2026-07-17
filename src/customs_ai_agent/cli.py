from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from customs_ai_agent.adapters import LLMConfigurationError, ensure_llm_mode_available
from customs_ai_agent.adapters.retrieval import (
    HttpRegulationRetriever,
    fetch_supported_review_as_of,
)
from customs_ai_agent.fixtures.scenarios import list_scenarios
from customs_ai_agent.runner import answer_question, run_review, to_review_run_result
from customs_ai_agent.schemas.models import ReviewRunRequest


def _cmd_list_scenarios() -> int:
    for scenario in list_scenarios():
        print(scenario)
    return 0


EVIDENCE_ENV_KEYS = frozenset(
    {
        "EVIDENCE_SERVICE_BASE_URL",
        "EVIDENCE_SERVICE_ALLOWED_ORIGINS",
        "EVIDENCE_SERVICE_TOKEN",
        "EVIDENCE_SERVICE_TOKEN_HEADER",
    }
)


def _explicit_env_values(
    env_file: str | os.PathLike[str] | None,
) -> dict[str, str]:
    if env_file is None:
        return {}
    path = Path(env_file).expanduser()
    if not path.is_file():
        raise LLMConfigurationError(
            "env_file_not_found",
            "The explicitly configured environment file does not exist.",
        )
    raw = dotenv_values(dotenv_path=path, verbose=False)
    return {
        name: value
        for name in EVIDENCE_ENV_KEYS
        if isinstance((value := raw.get(name)), str)
    }


def _configured_value(
    name: str,
    file_values: dict[str, str],
    default: str | None = None,
) -> str | None:
    if name in os.environ:
        return os.environ[name]
    return file_values.get(name, default)


def _configured_rag_url(
    cli_value: str | None,
    file_values: dict[str, str],
) -> str | None:
    if cli_value is not None:
        return cli_value
    configured = _configured_value("EVIDENCE_SERVICE_BASE_URL", file_values)
    if configured is None:
        return None
    if not configured.strip():
        raise ValueError("EVIDENCE_SERVICE_BASE_URL cannot be blank when configured")
    return configured


def _configured_service_token(file_values: dict[str, str]) -> str | None:
    configured = _configured_value("EVIDENCE_SERVICE_TOKEN", file_values)
    return configured if configured else None


def _configured_allowed_origins(file_values: dict[str, str]) -> frozenset[str]:
    configured = _configured_value(
        "EVIDENCE_SERVICE_ALLOWED_ORIGINS",
        file_values,
    )
    if not configured:
        return frozenset()
    return frozenset(
        origin.strip()
        for origin in configured.split(",")
        if origin.strip()
    )


def _parse_review_as_of(value: str) -> date:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise argparse.ArgumentTypeError("review date must use YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("review date must use YYYY-MM-DD") from exc


def _cmd_run(args: argparse.Namespace) -> int:
    # Validate provider configuration before any Evidence Service network call.
    ensure_llm_mode_available(args.llm_mode, getattr(args, "env_file", None))
    file_values = _explicit_env_values(getattr(args, "env_file", None))
    rag_url = _configured_rag_url(getattr(args, "rag_url", None), file_values)
    if args.llm_mode == "llm" and rag_url is None:
        raise LLMConfigurationError(
            "explicit_retriever_required",
            "llm mode requires --rag-url or EVIDENCE_SERVICE_BASE_URL; implicit Mock RAG is forbidden.",
        )
    review_as_of = getattr(args, "review_as_of", None)
    retriever = None
    if rag_url is not None:
        service_token = _configured_service_token(file_values)
        allowed_origins = _configured_allowed_origins(file_values)
        token_header = _configured_value(
            "EVIDENCE_SERVICE_TOKEN_HEADER",
            file_values,
            "X-Demo-Token",
        )
        if not token_header or not token_header.strip():
            raise ValueError("EVIDENCE_SERVICE_TOKEN_HEADER cannot be blank")
        retriever_options: dict[str, Any] = {
            "service_token": service_token,
            "token_header": token_header.strip(),
        }
        if allowed_origins:
            retriever_options["allowed_origins"] = allowed_origins
        retriever = HttpRegulationRetriever(rag_url, **retriever_options)
        if review_as_of is None:
            readiness_options = dict(retriever_options)
            review_as_of = fetch_supported_review_as_of(
                rag_url,
                **readiness_options,
            )
    request_values = {
        "scenario_id": args.scenario,
        "llm_mode": args.llm_mode,
    }
    if review_as_of is not None:
        request_values["review_as_of"] = review_as_of
    request = ReviewRunRequest.model_validate(request_values)
    if getattr(args, "env_file", None) is not None:
        state = run_review(request, retriever=retriever, env_file=args.env_file)
    else:
        state = run_review(request, retriever=retriever)
    public_result = to_review_run_result(state)
    payload = json.dumps(
        public_result.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        default=str,
    )

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
        "--env-file",
        help=(
            "Explicit local environment file for llm mode. Only ANTHROPIC_API_KEY "
            "and allowlisted Claude/Evidence settings are read without mutating "
            "the process environment. Existing process values win; the path is not logged."
        ),
    )
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
