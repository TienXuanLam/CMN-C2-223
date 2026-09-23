"""Focused unit tests for the independent CMN-C2-223 use case."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets.inmemory_provider import InMemoryProvider

from src.nodes.post_process_node import ResponseValidateNode
from src.nodes.pre_process_node import QueryNormalizeNode, classify_topic
from src.nodes.response_generate_node import ResponseGenerateNode
from src.nodes.topic_route_node import TopicRouteNode

_SECRETS = InMemoryProvider(
    {
        "AZURE_OPENAI_API_KEY": "unit-test-key",
        "AZURE_OPENAI_ENDPOINT": "https://test.services.ai.azure.com",
        "AZURE_OPENAI_DEPLOYMENT": "test-deployment",
    }
)


def _mock_azure_client(response: object = None, side_effect: Exception | None = None) -> MagicMock:
    instance = MagicMock()
    if side_effect is not None:
        instance.complete.side_effect = side_effect
    else:
        instance.complete.return_value = (
            response if response is not None else {"content": "Grounded answer [Source: docs]."}
        )
    return instance


def state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "user_input": "How does Fara-7B run with vLLM?",
        "input_context": {"run_id": "unit"},
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "caller_id": "unit",
        "correlation_id": "unit",
        "session_id": "unit",
        "thread_id": "unit",
        "trace_id": "unit",
        "hitl_allowed": True,
        "node_history": [],
        "error_log": [],
        "execution_time": {},
        "status": AgentStatus.PENDING.value,
    }
    base.update(overrides)
    return base


def test_topic_classification_is_domain_specific() -> None:
    assert classify_topic("WebTailBench score") == "benchmark"
    assert classify_topic("Deploy on Windows Server in Japan") == "japan_deployment"
    assert classify_topic("Fara1.5 harness") == "fara1_5_harness"


def test_preprocess_normalizes_and_routes() -> None:
    result = QueryNormalizeNode().execute(state(user_input="  vLLM   hosting setup  "))
    assert result["normalized_query"] == "vLLM hosting setup"
    assert result["topic"] == "vllm_hosting"


def test_preprocess_blocks_empty_and_oversized() -> None:
    assert QueryNormalizeNode().execute(state(user_input=""))["error_code"] == "S1_EMPTY_QUERY"
    oversized = "x" * 2001
    assert QueryNormalizeNode().execute(state(user_input=oversized))["error_code"] == "S1_QUERY_TOO_LONG"


def test_preprocess_security_hook_masks_pii_via_framework_default() -> None:
    raw = "Contact user@example.com about Fara"
    with bound_secrets(_SECRETS):
        result = QueryNormalizeNode()(state(user_input=raw))
    assert result["status"] == AgentStatus.SUCCESS.value
    assert "user@example.com" not in result["normalized_query"]


def test_preprocess_security_hook_rejects_credentials() -> None:
    raw = "My API key is sk-" + "x" * 30
    with bound_secrets(_SECRETS):
        result = QueryNormalizeNode()(state(user_input=raw))
    assert result["status"] == AgentStatus.ERROR.value
    assert result["error_code"] == "S2_CREDENTIAL_DETECTED"


def test_topic_route_uses_two_kbs_for_rpa() -> None:
    result = TopicRouteNode().execute(state(topic="rpa_comparison"))
    assert json.loads(result["retrieval_plan"])["kb_targets"] == ["fara_kb", "japan_kb"]


def test_generation_accepts_only_canonical_response() -> None:
    good = _mock_azure_client()
    node = ResponseGenerateNode(llm_config={})
    with patch("src.nodes.response_generate_node.AzureOpenAIClient", return_value=good), bound_secrets(_SECRETS):
        result = node.execute(state(topic="general", normalized_query="question", kb_results="[]"))
    assert result["status"] == AgentStatus.SUCCESS.value
    assert good.complete.call_count == 1

    malformed_node = ResponseGenerateNode(llm_config={})
    with (
        patch("src.nodes.response_generate_node.AzureOpenAIClient", return_value=_mock_azure_client("legacy string")),
        bound_secrets(_SECRETS),
    ):
        result = malformed_node.execute(state(kb_results="[]"))
    assert result["error_code"] == "LLM_RESPONSE_INVALID"


def test_generation_fails_closed_on_llm_error() -> None:
    node = ResponseGenerateNode(llm_config={})
    with (
        patch(
            "src.nodes.response_generate_node.AzureOpenAIClient",
            return_value=_mock_azure_client(side_effect=RuntimeError("unavailable")),
        ),
        bound_secrets(_SECRETS),
    ):
        result = node.execute(state(kb_results="[]"))
    assert result["status"] == AgentStatus.ERROR.value
    assert result["error_code"] == "LLM_CALL_FAILED"


def test_generation_fails_closed_on_missing_secret() -> None:
    node = ResponseGenerateNode(llm_config={})
    result = node.execute(state(kb_results="[]"))
    assert result["status"] == AgentStatus.ERROR.value
    assert result["error_code"] == "LLM_CALL_FAILED"


def test_postprocess_envelope_and_fara15_warning() -> None:
    result = ResponseValidateNode().execute(
        state(generated_response="Grounded answer.", topic="fara1_5_harness", citation_list="[]", kb_results="[]")
    )
    envelope = json.loads(result["formatted_output"])
    assert envelope["topic"] == "fara1_5_harness"
    assert envelope["kb_hit_count"] == 0
    assert len(envelope["validation_flags"]) == 2


def test_every_domain_node_has_verified_trust() -> None:
    assert QueryNormalizeNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL
    assert TopicRouteNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL
    assert ResponseGenerateNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL
    assert ResponseValidateNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL
