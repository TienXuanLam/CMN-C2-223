"""Proof that the migrated project follows the latest Cat 2 contract."""

from __future__ import annotations

import ast
from pathlib import Path

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.graph.base_graph import BaseGraph
from framework.nodes.function_node import FunctionNode
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel

from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.fara_workflow_node import FaraWorkflowGraphNode
from src.graph.graph import CMNFaraQAAgent
from src.nodes.hybrid_retrieve_node import HybridRetrieveNode
from src.nodes.post_process_node import ResponseValidateNode
from src.nodes.pre_process_node import QueryNormalizeNode
from src.nodes.response_generate_node import ResponseGenerateNode
from src.nodes.topic_route_node import TopicRouteNode
from src.nodes.workflow_input_node import WorkflowInputNode
from src.schemas.state import FaraQAState

_ROOT = Path(__file__).resolve().parents[2]


def test_cat2_parent_types_and_main_slot() -> None:
    assert issubclass(CMNFaraQAAgent, AgentBaseGraph)
    assert issubclass(FaraWorkflowGraphNode, GraphNode)
    assert issubclass(DomainWorkflowGraph, BaseGraph)
    graph = CMNFaraQAAgent(config={"top_k": 5, "llm": {}})
    graph.register_nodes()
    assert isinstance(graph._nodes["main"], FaraWorkflowGraphNode)


def test_outer_graph_does_not_override_framework_edges() -> None:
    assert "add_edges" not in CMNFaraQAAgent.__dict__
    assert "route" not in CMNFaraQAAgent.__dict__


def test_inner_graph_has_expected_reachable_topology() -> None:
    source = (_ROOT / "src/graph/domain_workflow_graph.py").read_text(encoding="utf-8")
    for node_name in ("workflow_input", "topic_route", "hybrid_retrieve", "response_generate"):
        assert node_name in source


def test_all_domain_nodes_are_function_nodes_with_explicit_trust() -> None:
    classes = (WorkflowInputNode, QueryNormalizeNode, TopicRouteNode, HybridRetrieveNode, ResponseGenerateNode, ResponseValidateNode)
    for node_class in classes:
        assert issubclass(node_class, FunctionNode)
        assert node_class.required_trust_level is TrustLevel.VERIFIED_EXTERNAL
        assert "_security_gate_input" not in node_class.__dict__
        assert "_security_gate_output" not in node_class.__dict__


def test_state_is_flat_and_checkpoint_safe() -> None:
    assert issubclass(FaraQAState, dict)
    inherited = set(AgentState.__annotations__)
    for name, annotation in FaraQAState.__annotations__.items():
        if name not in inherited:
            assert "dict" not in str(annotation).lower()
            assert "list" not in str(annotation).lower()


def test_no_level_zero_imports() -> None:
    violations: list[str] = []
    for path in (_ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for item in ast.walk(tree):
            module = ""
            if isinstance(item, ast.ImportFrom):
                module = item.module or ""
            elif isinstance(item, ast.Import):
                module = item.names[0].name
            if module == "agenticstar" or module.startswith("agenticstar."):
                violations.append(f"{path}:{item.lineno}")
    assert not violations


def test_no_scaffold_examples_or_composite_main_remain() -> None:
    assert not (_ROOT / "src/examples").exists() or not list((_ROOT / "src/examples").glob("*.py"))
    assert not (_ROOT / "src/nodes/main_node.py").exists()
