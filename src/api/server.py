"""Standalone HTTP adapter for CMNFaraQAAgent."""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from framework.utils.config_loader import load_config
from shared.secrets import factory as secrets_factory

from src.graph.graph import CMNFaraQAAgent

app = FastAPI(title="CMNFaraQAAgent")
logger = logging.getLogger(__name__)


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
_config = load_config(str(_CONFIG_PATH)) if _CONFIG_PATH.exists() else {}
_secrets_provider = secrets_factory(namespace="cmn", agent_name="cmn-c2-223")
# LLM credentials are resolved per-invocation inside ResponseGenerateNode via
# InvocationContext.from_state(state).secrets.require(...) -- no LLM client
# is built here at module import time.
agent = CMNFaraQAAgent(config=_config)
agent.compile()
agent.provision_secrets(_secrets_provider)


class InvokeRequest(BaseModel):
    input: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="", max_length=256)
    input_context: dict[str, Any] = Field(default_factory=dict)


def _bearer_matches(supplied: str, expected: str) -> bool:
    return secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode())


def _resolve_standalone_trust(
    current: TrustLevel,
    authorization: str,
    invoke_auth_token: str | None,
    internal_runner_token: str | None,
) -> TrustLevel:
    if current is not TrustLevel.ANONYMOUS:
        return current
    if internal_runner_token and _bearer_matches(authorization, internal_runner_token):
        return TrustLevel.INTERNAL
    if invoke_auth_token and _bearer_matches(authorization, invoke_auth_token):
        return TrustLevel.VERIFIED_EXTERNAL
    if internal_runner_token or invoke_auth_token:
        raise HTTPException(status_code=401, detail="Token is invalid or expired.")
    return TrustLevel.ANONYMOUS


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> dict[str, Any]:
    trust = _resolve_standalone_trust(
        getattr(request.state, "trust_level", TrustLevel.ANONYMOUS),
        request.headers.get("authorization", ""),
        os.environ.get("INVOKE_AUTH_TOKEN"),
        os.environ.get("STG_INTERNAL_RUNNER_TOKEN"),
    )
    with bound_secrets(agent._secrets_provider):
        input_context = dict(req.input_context)
        input_context.update({"run_id": str(uuid4())})
        context = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        return cast("dict[str, Any]", agent.invoke(req.input, ctx=context, input_context=input_context))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "CMNFaraQAAgent"}
