from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from customs_ai_agent.adapters.llm import DocumentAnalyzer
from customs_ai_agent.adapters.retrieval import RegulationRetriever
from customs_ai_agent.graph.state import ReviewState
from customs_ai_agent.nodes import (
    classify_documents,
    compliance_review,
    extract_documents,
    generate_xai_report,
    load_case,
    normalize_documents,
    pair_documents,
    retrieve_regulations,
    route_to_human_or_complete,
    self_correct_extraction,
    validate_cross_documents,
)
from customs_ai_agent.nodes.self_correct import has_low_confidence_extraction


def _route_after_validation(
    state: ReviewState,
) -> Literal["self_correct_extraction", "retrieve_regulations"]:
    if has_low_confidence_extraction(state) and int(state.get("retry_count") or 0) < 2:
        return "self_correct_extraction"
    return "retrieve_regulations"


def build_review_graph(
    retriever: RegulationRetriever,
    analyzer: DocumentAnalyzer | None = None,
):
    workflow = StateGraph(ReviewState)

    workflow.add_node("load_case", load_case)
    workflow.add_node(
        "classify_documents",
        lambda state: classify_documents(state, analyzer),
    )
    workflow.add_node("extract_documents", extract_documents)
    workflow.add_node("normalize_documents", normalize_documents)
    workflow.add_node("pair_documents", pair_documents)
    workflow.add_node("validate_cross_documents", validate_cross_documents)
    workflow.add_node("self_correct_extraction", self_correct_extraction)
    workflow.add_node(
        "retrieve_regulations",
        lambda state: retrieve_regulations(state, retriever),
    )
    workflow.add_node("compliance_review", compliance_review)
    workflow.add_node("generate_xai_report", generate_xai_report)
    workflow.add_node("route_to_human_or_complete", route_to_human_or_complete)

    workflow.add_edge(START, "load_case")
    workflow.add_edge("load_case", "classify_documents")
    workflow.add_edge("classify_documents", "extract_documents")
    workflow.add_edge("extract_documents", "normalize_documents")
    workflow.add_edge("normalize_documents", "pair_documents")
    workflow.add_edge("pair_documents", "validate_cross_documents")
    workflow.add_conditional_edges(
        "validate_cross_documents",
        _route_after_validation,
        {
            "self_correct_extraction": "self_correct_extraction",
            "retrieve_regulations": "retrieve_regulations",
        },
    )
    workflow.add_edge("self_correct_extraction", "normalize_documents")
    workflow.add_edge("retrieve_regulations", "compliance_review")
    workflow.add_edge("compliance_review", "generate_xai_report")
    workflow.add_edge("generate_xai_report", "route_to_human_or_complete")
    workflow.add_edge("route_to_human_or_complete", END)

    return workflow.compile()
