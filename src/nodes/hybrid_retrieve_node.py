"""Bounded file-backed hybrid retrieval for the Fara knowledge bases."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from framework.security import evaluate_untrusted_content
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import FaraQAState
from src.services.kb_path_policy import resolve_project_file

_LOGGER = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MAX_KB_BYTES = 5 * 1024 * 1024
_MAX_KB_LINES = 10_000
_MAX_CONTENT_CHARS = 8_000


class HybridRetrieveNode(FunctionNode):
    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, fara_kb_path: str, japan_kb_path: str, top_k: int = 5) -> None:
        self._kb_paths = {"fara_kb": fara_kb_path, "japan_kb": japan_kb_path}
        self._top_k = top_k
        self._kb_cache: dict[str, list[dict[str, str]]] = {}

    def execute(self, state: FaraQAState) -> dict[str, Any]:
        emit_trace_event("hybrid_retrieval_started", {}, state)
        query = state.get("normalized_query") or state.get("query") or ""
        try:
            plan = json.loads(state.get("retrieval_plan") or "{}")
        except json.JSONDecodeError:
            return self._error("RETRIEVAL_PLAN_INVALID", "Retrieval plan is malformed.")
        if not isinstance(plan, dict):
            return self._error("RETRIEVAL_PLAN_INVALID", "Retrieval plan must be an object.")
        targets = plan.get("kb_targets", ["fara_kb"])
        if not isinstance(targets, list) or not all(target in self._kb_paths for target in targets):
            return self._error("RETRIEVAL_PLAN_INVALID", "Retrieval plan contains an unknown KB target.")
        try:
            dense_weight = float(plan.get("dense", 0.7))
            sparse_weight = float(plan.get("sparse", 0.3))
        except (TypeError, ValueError):
            return self._error("RETRIEVAL_PLAN_INVALID", "Retrieval weights must be numeric.")
        if dense_weight < 0 or sparse_weight < 0 or abs(dense_weight + sparse_weight - 1.0) > 0.001:
            return self._error("RETRIEVAL_PLAN_INVALID", "Retrieval weights must be non-negative and sum to 1.")

        documents: list[dict[str, str]] = []
        for target in targets:
            try:
                loaded = self._load_kb(target)
            except ValueError as exc:
                return self._error("KB_INVALID", str(exc))
            for document in loaded:
                checked = evaluate_untrusted_content(
                    document["content"],
                    source=f"{target}:{document['source']}",
                    state=state,
                    node_name=type(self).__name__,
                )
                if checked.get("status") == AgentStatus.ERROR.value:
                    return {
                        "error_code": "KB_UNTRUSTED_CONTENT",
                        "error_message": "Knowledge base content failed the injection policy.",
                        "error_log": checked.get("error_log", ["KB_UNTRUSTED_CONTENT"]),
                        "status": AgentStatus.ERROR.value,
                    }
                documents.append({**document, "kb_type": target})

        if not documents:
            return {
                "kb_results": "[]",
                "citation_list": "[]",
                "status": AgentStatus.SUCCESS.value,
            }

        query_tokens = query.casefold().split()
        dense = [
            (index, self._dense_score(query, doc["content"]) * dense_weight) for index, doc in enumerate(documents)
        ]
        sparse = [
            (index, self._bm25_score(query_tokens, doc["content"]) * sparse_weight)
            for index, doc in enumerate(documents)
        ]
        merged = self._rrf_merge(
            sorted(dense, key=lambda item: item[1], reverse=True),
            sorted(sparse, key=lambda item: item[1], reverse=True),
        )[: self._top_k]
        hits: list[dict[str, Any]] = []
        citations: list[str] = []
        dense_by_index = dict(dense)
        sparse_by_index = dict(sparse)
        for index in merged:
            document = documents[index]
            source = document["source"]
            hits.append(
                {
                    "source": source,
                    "content": document["content"][:800],
                    "kb_type": document["kb_type"],
                    "retrieval_method": "hybrid_rrf",
                    "score": round(dense_by_index[index] + sparse_by_index[index], 4),
                }
            )
            if source not in citations:
                citations.append(source)
        return {
            "kb_results": json.dumps(hits, ensure_ascii=False),
            "citation_list": json.dumps(citations, ensure_ascii=False),
            "status": AgentStatus.SUCCESS.value,
        }

    def _load_kb(self, kb_name: str) -> list[dict[str, str]]:
        if kb_name in self._kb_cache:
            return self._kb_cache[kb_name]
        configured = self._kb_paths[kb_name]
        try:
            resolved = resolve_project_file(configured, _PROJECT_ROOT)
        except ValueError as exc:
            raise ValueError(f"KB path escapes project root: {kb_name}") from exc
        if not resolved.exists():
            _LOGGER.warning("KB file not found: %s; using an empty corpus", resolved)
            self._kb_cache[kb_name] = []
            return []
        if not resolved.is_file() or resolved.stat().st_size > _MAX_KB_BYTES:
            raise ValueError(f"KB file is invalid or exceeds {_MAX_KB_BYTES} bytes: {kb_name}")
        documents: list[dict[str, str]] = []
        with resolved.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if line_number > _MAX_KB_LINES:
                    raise ValueError(f"KB exceeds {_MAX_KB_LINES} records: {kb_name}")
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"KB has invalid JSON at line {line_number}: {kb_name}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"KB record must be an object at line {line_number}: {kb_name}")
                content = record.get("content", record.get("text"))
                source = record.get("source", record.get("title"))
                if not isinstance(content, str) or not content.strip() or len(content) > _MAX_CONTENT_CHARS:
                    raise ValueError(f"KB content is invalid at line {line_number}: {kb_name}")
                if not isinstance(source, str) or not source.strip() or len(source) > 256:
                    raise ValueError(f"KB source is invalid at line {line_number}: {kb_name}")
                documents.append({"source": source.strip(), "content": content.strip()})
        self._kb_cache[kb_name] = documents
        return documents

    @staticmethod
    def _dense_score(query: str, document: str) -> float:
        query_tokens = set(query.casefold().split())
        document_tokens = set(document.casefold().split())
        if not query_tokens or not document_tokens:
            return 0.0
        return len(query_tokens & document_tokens) / math.sqrt(len(query_tokens) * len(document_tokens))

    @staticmethod
    def _bm25_score(query_tokens: list[str], document: str) -> float:
        tokens = document.casefold().split()
        frequencies = {token: tokens.count(token) for token in set(tokens)}
        score = 0.0
        for token in query_tokens:
            frequency = frequencies.get(token, 0)
            if frequency:
                score += (frequency * 2.5) / (frequency + 1.5 * (0.25 + 0.75 * len(tokens) / 200.0))
        return score

    @staticmethod
    def _rrf_merge(dense: list[tuple[int, float]], sparse: list[tuple[int, float]], k: int = 60) -> list[int]:
        scores: dict[int, float] = {}
        for ranking in (dense, sparse):
            for rank, (index, _) in enumerate(ranking, start=1):
                scores[index] = scores.get(index, 0.0) + 1.0 / (k + rank)
        return sorted(scores, key=scores.__getitem__, reverse=True)

    @staticmethod
    def _error(code: str, message: str) -> dict[str, Any]:
        return {
            "error_code": code,
            "error_message": message,
            "error_log": [f"{code}: {message}"],
            "status": AgentStatus.ERROR.value,
        }
