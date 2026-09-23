"""Map supported Fara topics to retrieval plans."""

from __future__ import annotations

import json
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import FaraQAState

TOPIC_ROUTING: dict[str, dict[str, Any]] = {
    "benchmark": {"kb_targets": ["fara_kb"], "dense": 0.8, "sparse": 0.2},
    "vllm_hosting": {"kb_targets": ["fara_kb"], "dense": 0.7, "sparse": 0.3},
    "magentic_ui": {"kb_targets": ["fara_kb"], "dense": 0.7, "sparse": 0.3},
    "fara1_5_harness": {"kb_targets": ["fara_kb"], "dense": 0.7, "sparse": 0.3, "incomplete_kb": True},
    "rpa_comparison": {"kb_targets": ["fara_kb", "japan_kb"], "dense": 0.6, "sparse": 0.4},
    "japan_deployment": {"kb_targets": ["japan_kb", "fara_kb"], "dense": 0.5, "sparse": 0.5},
    "general": {"kb_targets": ["fara_kb", "japan_kb"], "dense": 0.6, "sparse": 0.4},
}


class TopicRouteNode(FunctionNode):
    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: FaraQAState) -> dict[str, Any]:
        emit_trace_event("topic_route_started", {}, state)
        topic = state.get("topic") or "general"
        plan = TOPIC_ROUTING.get(topic, TOPIC_ROUTING["general"])
        return {"retrieval_plan": json.dumps(plan), "status": AgentStatus.SUCCESS.value}
