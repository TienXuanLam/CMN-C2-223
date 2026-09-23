"""Validate and format the grounded Fara response."""

from __future__ import annotations

import json
import re
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import FaraQAState

_INTERNAL_URL = re.compile(r"https?://(?:localhost|127\.0\.0\.1|[^/\s]+\.(?:internal|local))(?:[/\s]|$)", re.I)
_FARA15_WARNING = "fara1_5_harness: Fara1.5 harness documentation is not yet publicly released."


class ResponseValidateNode(FunctionNode):
    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def _extra_security_gate_output(self, result: dict[str, Any]) -> dict[str, Any]:
        if any(_INTERNAL_URL.search(value) for value in result.values() if isinstance(value, str)):
            raise RuntimeError("S-3 output gate: internal URL detected")
        return result

    def execute(self, state: FaraQAState) -> dict[str, Any]:
        emit_trace_event("response_validation_started", {}, state)
        response = state.get("generated_response")
        if not isinstance(response, str) or not response.strip():
            return self._error("S3_EMPTY_RESPONSE", "Generated response is empty.")
        try:
            citations = json.loads(state.get("citation_list") or "[]")
            hits = json.loads(state.get("kb_results") or "[]")
        except json.JSONDecodeError:
            return self._error("S3_INVALID_STATE", "Retrieval output is malformed.")
        if not isinstance(citations, list) or not all(isinstance(item, str) for item in citations):
            return self._error("S3_INVALID_STATE", "Citation list is malformed.")
        if not isinstance(hits, list):
            return self._error("S3_INVALID_STATE", "KB results are malformed.")
        flags: list[str] = []
        if state.get("topic") == "fara1_5_harness":
            flags.append(_FARA15_WARNING)
        if not hits:
            flags.append("no_kb_results: No relevant knowledge base documents were found.")
        envelope = {
            "response": response.strip(),
            "topic": state.get("topic") or "general",
            "citations": citations,
            "validation_flags": flags,
            "kb_hit_count": len(hits),
        }
        return {
            "validation_flags": json.dumps(flags),
            "formatted_output": json.dumps(envelope, ensure_ascii=False),
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
