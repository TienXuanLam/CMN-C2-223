# Test Specification — CMN-C2-223

This suite verifies this project as an independent Fara deployment Q&A use case.

## Unit coverage

- Topic classification and two-KB routing behavior.
- Empty/oversized input, high-confidence PII, credentials, and injection rejection.
- Canonical LLM response validation and fail-closed provider errors.
- Output envelope and Fara1.5 / empty-KB warnings.
- KB path containment, record schema, size bounds, unique citations, top-k, and untrusted-content blocking.

## Integration and proof-of-boundary coverage

- Full outer graph and inner domain graph invocation with an isolated deterministic LLM.
- Anonymous denial, injection denial, credential-output denial, and Japan-topic propagation.
- `AgentBaseGraph → GraphNode → BaseGraph` inheritance and topology.
- Explicit `VERIFIED_EXTERNAL` trust on every domain node.
- No outer edge override, no composite `main_node.py`, no scaffold examples, no Level-0 imports.
- Flat/checkpoint-safe State and framework TC06/TC07/PB7 checks.

## Required local gates

Run:

```bash
export AGENTCORE_WHEEL_SPEC='agenticstar-agentcore[openai]==1.0.1'
.claude/common-scripts/check-security.sh
.claude/common-scripts/check-local.sh
```

Acceptance requires Ruff lint/format, strict mypy, project tests, proof-of-boundary, security checks, and Stage 5 Overall to pass.
