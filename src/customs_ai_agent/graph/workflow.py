from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from customs_ai_agent.adapters.llm import DocumentAnalyzer
from customs_ai_agent.adapters.retrieval import RegulationRetriever
from customs_ai_agent.graph.checkpoint import SensitiveInputSafeCheckpointer
from customs_ai_agent.graph.state import ReviewState
from customs_ai_agent.nodes import (
    classify_documents,
    compliance_review,
    extract_documents,
    generate_xai_report,
    load_case,
    normalize_documents,
    pair_documents,
    redact_sensitive_inputs,
    retrieve_regulations,
    route_to_human_or_complete,
    self_correct_extraction,
    validate_cross_documents,
)
from customs_ai_agent.nodes.self_correct import has_low_confidence_extraction
from customs_ai_agent.nodes.redact import without_image_payloads
from customs_ai_agent.payloads import SensitiveDocumentPayloadStore


class _SensitiveInputSafeGraph:
    """Ensure even direct graph callers cannot checkpoint raw image payloads."""

    __slots__ = ("__compiled_graph",)

    _READ_ONLY_DELEGATES = frozenset(
        {
            "aget_state",
            "aget_state_history",
            "get_graph",
            "get_state",
            "get_state_history",
        }
    )

    def __init__(self, compiled_graph: Any) -> None:
        object.__setattr__(
            self,
            "_SensitiveInputSafeGraph__compiled_graph",
            compiled_graph,
        )

    def __getattribute__(self, name: str) -> Any:
        if name in {
            "_compiled_graph",
            "__compiled_graph",
            "_SensitiveInputSafeGraph__compiled_graph",
            "__dict__",
        }:
            raise AttributeError(
                "the raw compiled graph is not exposed by the sensitive-input-safe facade"
            )
        return object.__getattribute__(self, name)

    def invoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        return _compiled_graph_delegate(self).invoke(
            without_image_payloads(input),
            *args,
            **kwargs,
        )

    def stream(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        return _compiled_graph_delegate(self).stream(
            without_image_payloads(input),
            *args,
            **kwargs,
        )

    async def ainvoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        return await _compiled_graph_delegate(self).ainvoke(
            without_image_payloads(input),
            *args,
            **kwargs,
        )

    async def astream(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        async for item in _compiled_graph_delegate(self).astream(
            without_image_payloads(input),
            *args,
            **kwargs,
        ):
            yield item

    def update_state(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("direct_graph_state_mutation_forbidden")

    async def aupdate_state(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("direct_graph_state_mutation_forbidden")

    def bulk_update_state(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("direct_graph_state_mutation_forbidden")

    def __getattr__(self, name: str) -> Any:
        if name in self._READ_ONLY_DELEGATES:
            return getattr(_compiled_graph_delegate(self), name)
        raise AttributeError(
            f"{type(self).__name__} does not expose graph capability {name!r}; "
            "use invoke/stream or the read-only checkpoint methods"
        )


def _compiled_graph_delegate(graph: _SensitiveInputSafeGraph) -> Any:
    """Internal-only access that is absent from the returned facade's API."""

    return object.__getattribute__(
        graph,
        "_SensitiveInputSafeGraph__compiled_graph",
    )


def _route_after_validation(
    state: ReviewState,
) -> Literal["self_correct_extraction", "retrieve_regulations"]:
    if has_low_confidence_extraction(state) and int(state.get("retry_count") or 0) < 2:
        return "self_correct_extraction"
    return "retrieve_regulations"


def build_review_graph(
    retriever: RegulationRetriever,
    analyzer: DocumentAnalyzer | None = None,
    payload_store: SensitiveDocumentPayloadStore | None = None,
    checkpointer: Any | None = None,
):
    workflow = StateGraph(ReviewState)

    workflow.add_node("load_case", load_case)
    workflow.add_node(
        "classify_documents",
        lambda state: classify_documents(
            state,
            analyzer,
            payload_store=payload_store,
        ),
    )
    workflow.add_node(
        "extract_documents",
        extract_documents,
    )
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
    workflow.add_node("redact_sensitive_inputs", redact_sensitive_inputs)

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
    workflow.add_edge("route_to_human_or_complete", "redact_sensitive_inputs")
    workflow.add_edge("redact_sensitive_inputs", END)

    safe_checkpointer = (
        SensitiveInputSafeCheckpointer(checkpointer) if checkpointer is not None else None
    )
    return _SensitiveInputSafeGraph(workflow.compile(checkpointer=safe_checkpointer))
