"""Outer Cat 2 graph for the enterprise Fara deployment Q&A workflow."""

from framework.graph.agent_base_graph import AgentBaseGraph

from src.graph.fara_workflow_node import FaraWorkflowGraphNode
from src.nodes.post_process_node import ResponseValidateNode
from src.nodes.pre_process_node import QueryNormalizeNode
from src.schemas.state import FaraQAState


class CMNFaraQAAgent(AgentBaseGraph):
    @property
    def name(self) -> str:
        return "cmn_c2_fara_qa_agent"

    @property
    def state_schema(self) -> type:
        return FaraQAState

    def register_nodes(self) -> None:
        super().register_nodes()
        self._nodes["pre_process"] = QueryNormalizeNode()
        self._nodes["main"] = FaraWorkflowGraphNode(config=self.config)
        self._nodes["post_process"] = ResponseValidateNode()
