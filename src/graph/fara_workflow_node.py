"""GraphNode boundary for the Fara Q&A domain workflow."""

from __future__ import annotations

import json
from typing import Any, ClassVar

from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.graph.domain_workflow_graph import DomainWorkflowGraph


class FaraWorkflowGraphNode(GraphNode):
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL
    error_strategy: ClassVar[str] = "propagate"
    propagate_hitl: ClassVar[bool] = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = dict(config or {})

    def get_subgraph(self) -> DomainWorkflowGraph:
        return DomainWorkflowGraph(config=self._config)

    def extract_input(self, state: AgentState) -> str:
        return json.dumps(
            {
                "query": state.get("query"),
                "normalized_query": state.get("normalized_query"),
                "topic": state.get("topic"),
                "run_id": state.get("run_id"),
            },
            ensure_ascii=False,
        )

    def merge_output(self, state: AgentState, sub_result: dict[str, Any]) -> dict[str, Any]:
        output = sub_result.get("output")
        if not isinstance(output, dict):
            return {
                "status": AgentStatus.ERROR.value,
                "error_code": "SUBGRAPH_INVALID_OUTPUT",
                "error_message": "Fara workflow returned a malformed output.",
                "error_log": ["SUBGRAPH_INVALID_OUTPUT: output must be an object"],
            }
        merged = dict(output)
        merged["status"] = sub_result.get("status", AgentStatus.ERROR.value)
        if sub_result.get("error_log"):
            merged["error_log"] = sub_result["error_log"]
        return merged
