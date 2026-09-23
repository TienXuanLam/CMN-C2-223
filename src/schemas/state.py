"""Flat, checkpoint-safe state for CMN-C2-223."""

from framework.schemas.agent_state import AgentState


class FaraQAState(AgentState, total=False):  # type: ignore[call-arg]
    query: str | None
    normalized_query: str | None
    topic: str | None
    retrieval_plan: str | None
    kb_results: str | None
    citation_list: str | None
    generated_response: str | None
    validation_flags: str | None
    formatted_output: str | None
    run_id: str | None
    error_code: str | None
    error_message: str | None
