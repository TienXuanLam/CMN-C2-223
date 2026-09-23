# Template Design Specification — CMN-C2-223

## Position in AgentCore

- Template: `CMN-C2-223` / `CMNFaraQAAgent`
- Category: Cat 2
- L1 Base: `AgentBaseGraph`
- Required caller trust: `VERIFIED_EXTERNAL`
- State: flat `FaraQAState(AgentState, total=False)`; structured values are JSON strings.

## Topology

The outer graph uses the unmodified `AgentBaseGraph` lifecycle:

```text
START → initialize → QueryNormalizeNode → FaraWorkflowGraphNode → ResponseValidateNode → finalize → END
```

`FaraWorkflowGraphNode(GraphNode)` invokes `src/graph/domain_workflow_graph.py`:

```text
START → WorkflowInputNode → TopicRouteNode → HybridRetrieveNode → ResponseGenerateNode → END
```

The inner graph is a `BaseGraph`; no domain node is called manually through `.execute()`. This preserves framework trust, S-2/S-3, audit, history, and error behavior at every boundary.

## Domain behavior

Topic routing supports benchmark, vLLM hosting, Magentic-UI, Fara1.5 harness, RPA comparison, Japan deployment, and general questions. Retrieval uses bounded JSONL files within the project root, validates record schema and size, evaluates each document as untrusted content, then combines token-overlap and BM25 rankings with RRF.

The generator accepts only the canonical LLM response object with a non-empty string `content`. Missing clients, provider errors, malformed results, and oversized output produce error status. Empty KBs are allowed but surfaced as a validation warning.

## Security

- S-1: `VERIFIED_EXTERNAL` on all domain nodes; query length 1–2000.
- S-2: framework injection/PII handling plus hard rejection for credentials and high-confidence PII. Retrieved content is evaluated at the KB trust boundary.
- S-3: framework credential scan plus internal-URL rejection.
- S-4: framework lifecycle audit; no node overrides final gate methods.
- S-5: secrets come from `InvocationContext` / secret provider and never enter State.

## Configuration

`config/agent.yaml` is the current registry schema and points to `src.graph.graph.CMNFaraQAAgent`. `config/config.yaml` contains retry, KB paths, top-k, and LLM settings (model/temperature/max_tokens). The LLM client itself is never constructed or injected via config — `ResponseGenerateNode` resolves `AZURE_OPENAI_API_KEY`/`AZURE_OPENAI_ENDPOINT`/`AZURE_OPENAI_DEPLOYMENT` per-invocation via `InvocationContext.from_state(state).secrets.require(...)` and builds a fresh `AzureOpenAIClient` inside `execute()`; it is never cached on `self` or serialized into State.
