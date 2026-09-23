"""Validate and normalize enterprise Fara questions."""

from __future__ import annotations

import re
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from framework.security import detect_credentials_in_value
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import FaraQAState

_MAX_QUERY_CHARS = 2000
_TOPIC_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fara1_5_harness", ("fara1.5", "fara 1.5", "fara1_5", "1.5 harness")),
    ("benchmark", ("webtailbench", "benchmark", "benchmarks", "score", "gpt-4o compare", "performance test")),
    ("vllm_hosting", ("vllm", "port 5000", "inference server", "serving endpoint")),
    ("magentic_ui", ("magentic", "magentic-ui", "magentic_ui", "sap ui", "computer use", "screen automation")),
    ("rpa_comparison", ("winactor", "uipath", "blue prism", "rpa", "migrate from", "migration")),
    ("japan_deployment", ("japan", "japanese", "kanji", "windows server", "on-prem", "fisc", "onsite")),
)


def classify_topic(query: str) -> str:
    lowered = query.casefold()
    for topic, keywords in _TOPIC_RULES:
        if any(keyword in lowered for keyword in keywords):
            return topic
    return "general"


class QueryNormalizeNode(FunctionNode):
    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def _extra_security_gate_input(self, state: FaraQAState) -> FaraQAState:
        # Framework defaults already cover this field: _PII_SCAN_FIELDS masks PII
        # in user_input before this hook runs, and _INJECTION_BLOCKING_FIELDS
        # rejects high-confidence prompt injection in user_input. Credential
        # scanning is not part of either default, so it remains this node's
        # responsibility.
        raw_query: object = state.get("user_input", "")
        if isinstance(raw_query, str) and detect_credentials_in_value(raw_query):
            return self._blocked(state, "S2_CREDENTIAL_DETECTED", "Credentials are not accepted in questions.")
        return state

    def execute(self, state: FaraQAState) -> dict[str, Any]:
        query = state.get("user_input", "")
        if not isinstance(query, str) or not query.strip():
            return self._error("S1_EMPTY_QUERY", "query is empty or missing.")
        if len(query) > _MAX_QUERY_CHARS:
            return self._error("S1_QUERY_TOO_LONG", f"query exceeds {_MAX_QUERY_CHARS} characters.")
        normalized = re.sub(r"\s+", " ", query.strip())
        topic = classify_topic(normalized)
        input_context = state.get("input_context")
        run_id = str(input_context.get("run_id", "")) if isinstance(input_context, dict) else ""
        emit_trace_event(
            "query_normalize_ok",
            {"topic": topic, "query_length": len(normalized), "run_id": run_id},
            state,
        )
        return {
            "query": query.strip(),
            "normalized_query": normalized,
            "topic": topic,
            "run_id": run_id,
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _blocked(state: FaraQAState, code: str, message: str) -> FaraQAState:
        updated = dict(state)
        updated.update({"error_code": code, "error_message": message, "status": AgentStatus.ERROR.value})
        updated["error_log"] = [*state.get("error_log", []), f"{code}: {message}"]
        return updated  # type: ignore[return-value]

    @staticmethod
    def _error(code: str, message: str) -> dict[str, Any]:
        return {
            "error_code": code,
            "error_message": message,
            "error_log": [f"{code}: {message}"],
            "status": AgentStatus.ERROR.value,
        }
