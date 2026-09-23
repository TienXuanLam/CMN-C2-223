"""PB-6: verify framework lifecycle order around the Cat 2 GraphNode."""

from unittest.mock import MagicMock, patch

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets.inmemory_provider import InMemoryProvider

from src.graph.graph import CMNFaraQAAgent

_SECRETS = InMemoryProvider(
    {
        "AZURE_OPENAI_API_KEY": "pb6-test-key",
        "AZURE_OPENAI_ENDPOINT": "https://test.services.ai.azure.com",
        "AZURE_OPENAI_DEPLOYMENT": "test-deployment",
    }
)


def test_outer_invoke_order() -> None:
    graph = CMNFaraQAAgent(config={"top_k": 5, "llm": {}})
    graph.compile()
    mock_llm = MagicMock()
    mock_llm.complete.return_value = {"content": "The available evidence is insufficient for a precise recommendation."}
    with patch("src.nodes.response_generate_node.AzureOpenAIClient", return_value=mock_llm), bound_secrets(_SECRETS):
        result = graph.invoke(
            "How should Fara-7B be hosted with vLLM?",
            ctx=InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL),
            input_context={"run_id": "pb6"},
        )
    assert result["node_history"] == [
        "InitializeNode",
        "QueryNormalizeNode",
        "FaraWorkflowGraphNode",
        "ResponseValidateNode",
        "FinalizeNode",
    ]
