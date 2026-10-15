"""LangChain tool-calling retrieval over the MVEU knowledge base."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from cryptoslice.closure import (
    TEMPLATE_REQUIRED_SLOTS,
    evaluate_mveu_closure,
    required_slots_for_record,
)
from cryptoslice.meta import stable_json
from cryptoslice.schemas import TopKItem, TopKResult

DEFAULT_AGENT_MODEL = "openai/gpt-4o-mini"
DEFAULT_AGENT_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")


class EmptyToolInput(BaseModel):
    """No-argument tool schema for LangChain structured tools."""


class SearchMVEUInput(BaseModel):
    """Arguments for metadata-first MVEU search."""

    query: str = Field(description="Natural-language search text or slot requirement.")
    template: str | None = Field(default=None, description="Optional MVEU template name.")
    primitive: str | None = Field(default=None, description="Optional primitive/cipher name.")
    analysis_model: str | None = Field(default=None, description="Optional analysis model.")
    paper_id: str | None = Field(default=None, description="Optional paper identifier.")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum candidates to return.")


class InspectMVEUInput(BaseModel):
    """Arguments for fetching one MVEU in detail."""

    mveu_id: str = Field(description="The exact mveu_id returned by search_mveu_metadata.")


class SelectMVEUInput(BaseModel):
    """Arguments for recording the final MVEUs the agent will use."""

    mveu_ids: list[str] = Field(
        description="Ordered MVEU ids the agent will actually use for the answer or code."
    )
    rationale: str | None = Field(
        default=None,
        description="Brief reason for this final ordered selection.",
    )


@dataclass(frozen=True)
class AgenticMVEURetrieval:
    """Agent top-k result plus audit trace."""

    result: TopKResult
    audit: dict[str, Any]


class MVEUToolkit:
    """Compact MVEU search and inspection tools exposed to the agent."""

    def __init__(self, records: list[dict[str, Any]], *, default_limit: int = 20) -> None:
        self.default_limit = default_limit
        self.records = [_normalize_record(record) for record in records]
        self.by_id = {str(record["mveu_id"]): record for record in self.records}
        self.selected_mveu_ids: list[str] = []
        self.selection_rationale: list[dict[str, str]] = []

    def template_catalog(self) -> dict[str, list[str]]:
        """Return all MVEU templates and their required semantic slots."""

        catalog = {name: set(slots) for name, slots in TEMPLATE_REQUIRED_SLOTS.items()}
        for record in self.records:
            template = str(record.get("template") or "")
            if not template:
                continue
            catalog.setdefault(template, set()).update(
                required_slots_for_record(record.get("raw", record))
            )
        return {name: sorted(slots) for name, slots in sorted(catalog.items())}

    def template_catalog_json(self) -> str:
        return stable_json({"templates": self.template_catalog()})

    def search(
        self,
        *,
        query: str,
        template: str | None = None,
        primitive: str | None = None,
        analysis_model: str | None = None,
        paper_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search MVEUs by natural-language requirements and optional filters."""

        size = max(1, min(int(limit or self.default_limit), 50))
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for record in self.records:
            if not _passes_filters(
                record,
                template=template,
                primitive=primitive,
                analysis_model=analysis_model,
                paper_id=paper_id,
            ):
                continue
            score = _score_record(
                query,
                record,
                template=template,
                primitive=primitive,
                analysis_model=analysis_model,
                paper_id=paper_id,
            )
            scored.append((score, str(record["mveu_id"]), record))
        ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
        return [
            _compact_candidate(record, rank=index + 1, score=score)
            for index, (score, _, record) in enumerate(ranked[:size])
        ]

    def inspect(self, mveu_id: str) -> dict[str, Any]:
        """Return a compact but traceable evidence-only view of one MVEU."""

        record = self.by_id.get(str(mveu_id))
        if record is None:
            return {"status": "not_found", "mveu_id": str(mveu_id)}
        evidence = record.get("evidence", {})
        return {
            "status": "found",
            "mveu_id": record["mveu_id"],
            "template": evidence.get("template"),
            "scope": evidence.get("scope", {}),
            "template_slots": evidence.get("template_slots", {}),
            "source_evidence": evidence.get("source_evidence", []),
            "variables": evidence.get("variables", []),
            "objective": evidence.get("objective", {}),
            "round_function": evidence.get("round_function", {}),
            "sbox_table": evidence.get("sbox_table", {}),
            "state_layout": evidence.get("state_layout", {}),
        }

    def select(self, mveu_ids: list[str], rationale: str | None = None) -> dict[str, Any]:
        """Record the ordered MVEU ids the agent decided to use."""

        normalized_ids = [str(mveu_id).strip() for mveu_id in mveu_ids if str(mveu_id).strip()]
        valid_ids = _dedupe([mveu_id for mveu_id in normalized_ids if mveu_id in self.by_id])
        missing_ids = _dedupe([mveu_id for mveu_id in normalized_ids if mveu_id not in self.by_id])
        reason = str(rationale or "").strip()
        self.selected_mveu_ids = valid_ids
        self.selection_rationale = [
            {"mveu_id": mveu_id, "rationale": reason} for mveu_id in valid_ids
        ]
        return {
            "status": "ok" if valid_ids else "empty",
            "selected_mveu_ids": valid_ids,
            "missing_mveu_ids": missing_ids,
        }

    def to_topk_result(
        self,
        *,
        query_id: str,
        selected_ids: list[str],
        top_k: int,
        fill_query: str,
        fill_missing: bool = True,
    ) -> TopKResult:
        """Convert selected MVEU ids into the unified TopKResult contract."""

        unique_ids = _dedupe([mveu_id for mveu_id in selected_ids if mveu_id in self.by_id])
        if fill_missing and len(unique_ids) < top_k:
            for candidate in self.search(query=fill_query, limit=top_k * 2):
                candidate_id = str(candidate["mveu_id"])
                if candidate_id not in unique_ids:
                    unique_ids.append(candidate_id)
                if len(unique_ids) >= top_k:
                    break
        items = []
        for rank, mveu_id in enumerate(unique_ids[:top_k], start=1):
            record = self.by_id[mveu_id]
            items.append(
                TopKItem(
                    rank=rank,
                    mveu_id_or_chunk_id=mveu_id,
                    score=round(1.0 / rank, 6),
                    payload={
                        "mveu_id": record["mveu_id"],
                        "paper_id": record.get("paper_id"),
                        "template": record.get("template"),
                        "evidence": record.get("evidence", {}),
                        "provenance": record.get("provenance", []),
                        "agentic": True,
                    },
                )
            )
        return TopKResult(query_id=query_id, method="cryptoslice", items=items)

    def langchain_tools(self) -> list[Any]:
        """Build LangChain tools lazily so non-agent tests do not import LangChain."""

        try:
            from langchain_core.tools import StructuredTool
        except ImportError as exc:
            raise RuntimeError(
                "LangChain is required for agentic MVEU retrieval. "
                "Install the project dependencies or run `pip install langchain langchain-openai`."
            ) from exc

        return [
            StructuredTool.from_function(
                name="inspect_mveu_templates",
                description=(
                    "List every available MVEU template and the semantic slots that "
                    "must be satisfied for that template."
                ),
                func=lambda: self.template_catalog_json(),
                args_schema=EmptyToolInput,
            ),
            StructuredTool.from_function(
                name="search_mveu_metadata",
                description=(
                    "Search the MVEU KB using a natural-language requirement plus optional "
                    "filters such as template, primitive, analysis_model, or paper_id. "
                    "Returns traceable candidate MVEU ids, slots, closure, and source pages."
                ),
                func=self._search_tool,
                args_schema=SearchMVEUInput,
            ),
            StructuredTool.from_function(
                name="inspect_mveu",
                description=(
                    "Inspect a single MVEU by id, including template slots, source evidence, "
                    "scope, variables, and evidence-only modeling fields."
                ),
                func=self._inspect_tool,
                args_schema=InspectMVEUInput,
            ),
            StructuredTool.from_function(
                name="select_mveu",
                description=(
                    "Record the ordered MVEU ids you will actually use for the final "
                    "answer or modeling code. This tool call is the retrieval log used "
                    "by the main-comparison experiment."
                ),
                func=self._select_tool,
                args_schema=SelectMVEUInput,
            ),
        ]

    def _search_tool(
        self,
        query: str,
        template: str | None = None,
        primitive: str | None = None,
        analysis_model: str | None = None,
        paper_id: str | None = None,
        limit: int = 10,
    ) -> str:
        return stable_json(
            {
                "candidates": self.search(
                    query=query,
                    template=template,
                    primitive=primitive,
                    analysis_model=analysis_model,
                    paper_id=paper_id,
                    limit=limit,
                )
            }
        )

    def _inspect_tool(self, mveu_id: str) -> str:
        return _truncate(stable_json(self.inspect(mveu_id)), limit=12000)

    def _select_tool(self, mveu_ids: list[str], rationale: str | None = None) -> str:
        return stable_json(self.select(mveu_ids, rationale=rationale))


def retrieve_from_mveu_kb_agent(
    *,
    query_id: str,
    question: str,
    records: list[dict[str, Any]],
    top_k: int = 5,
    model: str = DEFAULT_AGENT_MODEL,
    api_base_url: str = DEFAULT_AGENT_API_BASE_URL,
    api_key_env: str = "OPENAI_API_KEY",
    candidate_limit: int = 20,
    max_steps: int = 6,
) -> AgenticMVEURetrieval:
    """Let a LangChain tool-calling model analyze a raw query and find MVEUs."""

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set; cannot run agentic MVEU retrieval")

    try:
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "LangChain is required for agentic MVEU retrieval. "
            "Install the project dependencies or run `pip install langchain langchain-openai`."
        ) from exc

    toolkit = MVEUToolkit(records, default_limit=candidate_limit)
    tools = toolkit.langchain_tools()
    tool_by_name = {tool.name: tool for tool in tools}
    llm = ChatOpenAI(
        model=model,
        base_url=api_base_url,
        api_key=api_key,
        temperature=0,
        timeout=120,
        max_retries=2,
    ).bind_tools(tools)

    messages: list[Any] = [
        SystemMessage(content=build_mveu_agent_system_prompt(toolkit)),
        HumanMessage(content=build_mveu_agent_user_prompt(question, top_k)),
    ]
    trace: list[dict[str, Any]] = []
    final_content = ""
    stopped_by = "max_steps"

    for step in range(max_steps):
        response = invoke_with_retries(llm, messages)
        messages.append(response)
        tool_calls = list(getattr(response, "tool_calls", []) or [])
        if not tool_calls:
            final_content = _message_content_to_text(getattr(response, "content", ""))
            stopped_by = "final_answer"
            break
        for call in tool_calls:
            tool_name = str(call.get("name") or "")
            tool_args = _coerce_tool_args(call.get("args"))
            tool_call_id = str(call.get("id") or f"call_{step}_{tool_name}")
            if tool_name not in tool_by_name:
                output = stable_json({"error": f"unknown tool: {tool_name}"})
            else:
                try:
                    output = str(tool_by_name[tool_name].invoke(tool_args))
                except Exception as exc:  # pragma: no cover - defensive audit path
                    output = stable_json({"error": type(exc).__name__, "message": str(exc)})
            messages.append(ToolMessage(content=output, tool_call_id=tool_call_id, name=tool_name))
            trace.append(
                {
                    "step": step + 1,
                    "tool": tool_name,
                    "args": tool_args,
                    "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                    "output_preview": _truncate(output, limit=800),
                }
            )

    parsed = parse_agent_final_json(final_content)
    tool_selected_ids = list(toolkit.selected_mveu_ids)
    legacy_final_ids = extract_selected_mveu_ids(parsed)
    if tool_selected_ids:
        selected_ids = tool_selected_ids
        selection_source = "select_mveu_tool"
    elif legacy_final_ids:
        selected_ids = legacy_final_ids
        selection_source = "final_response_legacy"
    else:
        selected_ids = [
            str(candidate["mveu_id"])
            for candidate in toolkit.search(query=question, limit=max(top_k, candidate_limit))
        ]
        selection_source = "fallback_search"
    fallback_used = selection_source == "fallback_search"
    result = toolkit.to_topk_result(
        query_id=query_id,
        selected_ids=selected_ids,
        top_k=top_k,
        fill_query=question,
        fill_missing=False,
    )
    audit = {
        "query_id": query_id,
        "raw_query_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "raw_query": question,
        "model": model,
        "api_base_url": api_base_url,
        "candidate_limit": candidate_limit,
        "max_steps": max_steps,
        "stopped_by": stopped_by,
        "fallback_used": fallback_used,
        "selection_source": selection_source,
        "final_response": parsed,
        "final_output_preview": _truncate(final_content, limit=4000),
        "selected_mveu_ids": selected_ids,
        "selected_mveu_ids_from_tool": tool_selected_ids,
        "selected_mveu_ids_from_final_response": legacy_final_ids,
        "retrieved_mveu_ids": [item.mveu_id_or_chunk_id for item in result.items],
        "selection_rationale": toolkit.selection_rationale,
        "tool_trace": trace,
    }
    return AgenticMVEURetrieval(result=result, audit=audit)


def invoke_with_retries(llm: Any, messages: list[Any], *, attempts: int = 3) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return llm.invoke(messages)
        except Exception as exc:  # pragma: no cover - provider/network failure path
            last_error = exc
            if attempt + 1 >= attempts:
                break
            time.sleep(2 * (attempt + 1))
    assert last_error is not None
    raise last_error


def build_mveu_agent_system_prompt(toolkit: MVEUToolkit) -> str:
    """Build the system prompt that teaches the model how to use MVEUs."""

    template_json = json.dumps(toolkit.template_catalog(), ensure_ascii=False, sort_keys=True)
    return (
        "You are the CryptoSlice MVEU retrieval agent. Your input is one raw "
        "natural-language symmetric-cryptography query. Do not ask the user for a "
        "structured query. Instead, infer the needed primitive, analysis model, "
        "round scope, MVEU template, and semantic slots yourself, then use tools to "
        "find traceable MVEUs.\n\n"
        "The tools expose public index metadata plus evidence-only MVEU details: "
        "evidence.mveu_id, evidence.template, evidence.scope (primitive, "
        "analysis_model, round_range), evidence.template_slots, variables, "
        "source_evidence, objective, round_function, sbox_table, and state_layout. "
        "Hidden eval/gold fields are not available to you. The available templates "
        f"and required slots are: {template_json}\n\n"
        "Use tools as follows:\n"
        "1. Use inspect_mveu_templates when template/slot requirements are unclear.\n"
        "2. Use search_mveu_metadata to discover candidate MVEUs. Search multiple "
        "times with different templates or slot words if the query is ambiguous.\n"
        "3. Use inspect_mveu on promising ids before final selection when details "
        "or provenance are needed.\n"
        "4. Before producing the final answer or modeling code, you MUST call "
        "select_mveu with the ordered mveu_id values you will actually use. "
        "Select only ids returned by tools. Never invent an id.\n\n"
        "Your final answer may be a concise explanation or modeling-language code "
        "such as CLAASP. It does not need to repeat the selected MVEU ids because "
        "the select_mveu tool call is the retrieval log."
    )


def build_mveu_agent_user_prompt(question: str, top_k: int) -> str:
    return (
        f"Raw query:\n{question}\n\n"
        f"Find up to {top_k} best MVEUs. Favor units whose template slots directly "
        "support the requested retrieval/modeling answer and whose source evidence "
        "is traceable. After inspecting enough candidates, call select_mveu with "
        "the ordered ids, then produce the final answer or code."
    )


def parse_agent_final_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {"raw_final_text": text}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {"raw_final_text": text}
        except json.JSONDecodeError:
            return {"raw_final_text": text}


def extract_selected_mveu_ids(parsed: dict[str, Any]) -> list[str]:
    direct = parsed.get("selected_mveu_ids") or parsed.get("mveu_ids")
    if isinstance(direct, list):
        return _dedupe([str(item) for item in direct if str(item).strip()])
    results = parsed.get("results")
    if isinstance(results, list):
        ids = []
        for item in results:
            if isinstance(item, dict):
                value = item.get("mveu_id") or item.get("id")
                if value:
                    ids.append(str(value))
        return _dedupe(ids)
    return []


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    evidence = record.get("evidence", {}) if isinstance(record.get("evidence"), dict) else {}
    eval_block = record.get("eval", {}) if isinstance(record.get("eval"), dict) else {}
    scope = evidence.get("scope", {}) if isinstance(evidence.get("scope"), dict) else {}
    mveu_id = evidence.get("mveu_id") or eval_block.get("query_id") or "unknown"
    provenance = eval_block.get("provenance") or evidence.get("source_evidence", [])
    return {
        "raw": record,
        "mveu_id": str(mveu_id),
        "paper_id": str(eval_block.get("paper_id") or evidence.get("paper_id") or ""),
        "template": str(evidence.get("template") or ""),
        "scope": scope,
        "primitive": str(scope.get("primitive") or ""),
        "analysis_model": str(scope.get("analysis_model") or ""),
        "round_range": scope.get("round_range"),
        "status": eval_block.get("status", "valid"),
        "closure": evaluate_mveu_closure(record).as_dict(),
        "evidence": evidence,
        "provenance": provenance,
    }


def _passes_filters(
    record: dict[str, Any],
    *,
    template: str | None,
    primitive: str | None,
    analysis_model: str | None,
    paper_id: str | None,
) -> bool:
    if record.get("status") not in {"valid", None}:
        return False
    if template and _norm(template) != _norm(record.get("template")):
        return False
    if paper_id and _norm_id(paper_id) != _norm_id(record.get("paper_id")):
        return False
    if primitive and not _soft_contains(primitive, record.get("primitive")):
        return False
    if analysis_model and not _soft_contains(analysis_model, record.get("analysis_model")):
        return False
    return True


def _score_record(
    query: str,
    record: dict[str, Any],
    *,
    template: str | None,
    primitive: str | None,
    analysis_model: str | None,
    paper_id: str | None,
) -> float:
    query_lower = query.lower()
    haystack = _record_haystack(record)
    query_tokens = set(_tokenize(query_lower))
    haystack_tokens = set(_tokenize(haystack))
    score = float(len(query_tokens & haystack_tokens))
    if record.get("primitive") and str(record["primitive"]).lower() in query_lower:
        score += 10.0
    if record.get("analysis_model") and str(record["analysis_model"]).lower() in query_lower:
        score += 7.0
    score += sum(
        2.0 for word in _split_template_name(record.get("template", "")) if word in query_lower
    )
    if template:
        score += 6.0
    if primitive:
        score += 4.0
    if analysis_model:
        score += 3.0
    if paper_id:
        score += 5.0
    closure = record.get("closure") or {}
    score += sum(0.25 for key in ("syn", "type", "sem") if closure.get(key) is True)
    return score


def _record_haystack(record: dict[str, Any]) -> str:
    evidence = record.get("evidence", {})
    slots = evidence.get("template_slots", {}) if isinstance(evidence, dict) else {}
    slot_text = stable_json(slots) if isinstance(slots, dict) else ""
    parts = [
        record.get("mveu_id", ""),
        record.get("paper_id", ""),
        record.get("template", ""),
        record.get("primitive", ""),
        record.get("analysis_model", ""),
        slot_text,
    ]
    return " ".join(str(part).lower() for part in parts)


def _compact_candidate(record: dict[str, Any], *, rank: int, score: float) -> dict[str, Any]:
    evidence = record.get("evidence", {})
    source_evidence = evidence.get("source_evidence", []) if isinstance(evidence, dict) else []
    pages = []
    for item in source_evidence:
        if isinstance(item, dict) and item.get("page") is not None:
            pages.append(item["page"])
    template_slots = evidence.get("template_slots", {}) if isinstance(evidence, dict) else {}
    return {
        "rank": rank,
        "score": round(score, 4),
        "mveu_id": record.get("mveu_id"),
        "paper_id": record.get("paper_id"),
        "template": record.get("template"),
        "primitive": record.get("primitive"),
        "analysis_model": record.get("analysis_model"),
        "round_range": record.get("round_range"),
        "slot_names": sorted(template_slots) if isinstance(template_slots, dict) else [],
        "source_pages": pages,
    }


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _coerce_tool_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _truncate(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...<truncated>"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _tokenize(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9_+^.-]+", value.lower()) if len(token) > 1]


def _split_template_name(template: str) -> list[str]:
    return [part.lower() for part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", str(template))]


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _norm_id(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _norm(value)).strip("_")


def _soft_contains(needle: Any, haystack: Any) -> bool:
    left = _norm(needle)
    right = _norm(haystack)
    return bool(left and right and (left in right or right in left))
