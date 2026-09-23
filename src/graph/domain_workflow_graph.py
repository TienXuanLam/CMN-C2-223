"""Inner Cat 2 workflow for Fara deployment Q&A."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_status import AgentStatus

from src.nodes.hybrid_retrieve_node import HybridRetrieveNode
from src.nodes.response_generate_node import ResponseGenerateNode
from src.nodes.topic_route_node import TopicRouteNode
from src.nodes.workflow_input_node import WorkflowInputNode
from src.schemas.state import FaraQAState


class DomainWorkflowGraph(BaseGraph):
    @property
    def name(self) -> str:
        return "cmn_c2_223_fara_qa_workflow"

    @property
    def state_schema(self) -> type:
        return FaraQAState

    def _validate_config(self) -> None:
        top_k = self.config.get("top_k", 5)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be an integer between 1 and 20")
        llm_config = self.config.get("llm", {})
        if not isinstance(llm_config, dict):
            raise ValueError("llm must be an object")

    def register_nodes(self) -> None:
        self._nodes = {
            "workflow_input": WorkflowInputNode(),
            "topic_route": TopicRouteNode(),
            "hybrid_retrieve": HybridRetrieveNode(
                fara_kb_path=str(self.config.get("fara_kb_path", "kb/fara7b_core.jsonl")),
                japan_kb_path=str(self.config.get("japan_kb_path", "kb/japan_deployment.jsonl")),
                top_k=int(self.config.get("top_k", 5)),
            ),
            "response_generate": ResponseGenerateNode(
                llm_config=self.config.get("llm", {}),
                timeout_s=int(self.config.get("timeout_s", 120)),
            ),
        }

    def add_edges(self) -> None:
        self._sg.add_edge(START, "workflow_input")
        self._sg.add_edge("workflow_input", "topic_route")
        self._sg.add_edge("topic_route", "hybrid_retrieve")
        self._sg.add_edge("hybrid_retrieve", "response_generate")
        self._sg.add_edge("response_generate", END)

    def route(self, state: FaraQAState) -> str:
        return END

    def get_output(self, state: FaraQAState) -> dict[str, Any]:
        output = {
            key: state.get(key)
            for key in (
                "retrieval_plan",
                "kb_results",
                "citation_list",
                "generated_response",
                "error_code",
                "error_message",
            )
            if state.get(key) is not None
        }
        return {
            "output": output,
            "status": state.get("status", AgentStatus.ERROR.value),
            "trace_id": state.get("trace_id", ""),
            "correlation_id": state.get("correlation_id", ""),
            "node_history": state.get("node_history", []),
            "error_log": state.get("error_log", []),
        }
