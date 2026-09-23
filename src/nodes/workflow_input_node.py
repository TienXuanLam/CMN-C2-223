"""Decode the validated outer state at the inner graph boundary."""

from __future__ import annotations

import json
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import FaraQAState


class WorkflowInputNode(FunctionNode):
    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: FaraQAState) -> dict[str, Any]:
        emit_trace_event("workflow_input_decode_started", {}, state)
        raw = state.get("user_input", "")
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return self._error("SUBGRAPH_INPUT_INVALID", "Workflow input must be valid JSON.")
        if not isinstance(payload, dict):
            return self._error("SUBGRAPH_INPUT_INVALID", "Workflow input must be an object.")
        query = payload.get("query")
        normalized = payload.get("normalized_query")
        topic = payload.get("topic")
        if not all(isinstance(value, str) and value for value in (query, normalized, topic)):
            return self._error("SUBGRAPH_INPUT_INVALID", "Workflow query and topic fields are required.")
        return {
            "query": query,
            "normalized_query": normalized,
            "topic": topic,
            "run_id": str(payload.get("run_id") or ""),
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _error(code: str, message: str) -> dict[str, Any]:
        return {
            "error_code": code,
            "error_message": message,
            "error_log": [f"{code}: {message}"],
            "status": AgentStatus.ERROR.value,
        }
