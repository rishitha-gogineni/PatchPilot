"""Bounded multi-agent workflow built on the existing LangGraph safety gate."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict
from uuid import uuid4

from .graph_workflow import _langgraph_api
from .llm import OpenAIPlanner, PlannerError, create_edit_proposal
from .logging import JsonlRunLogger
from .tools import ToolRegistry, build_repository_tool_registry


class MultiAgentState(TypedDict, total=False):
    task: str
    repository: str
    test_command: Optional[str]
    timeout: float
    max_revisions: int
    revision_count: int
    selected_files: list[str]
    plan: dict[str, Any]
    retrieved_context: list[dict[str, Any]]
    proposal: dict[str, Any]
    diff: str
    review: dict[str, Any]
    approved: bool
    applied: bool
    test_success: Optional[bool]
    status: str
    trace_id: str


Agent = Callable[[MultiAgentState, ToolRegistry], dict[str, Any]]


@dataclass(frozen=True)
class MultiAgentAgents:
    planner: Agent
    proposer: Agent
    reviewer: Agent


def _logger(state: MultiAgentState) -> JsonlRunLogger:
    repository = Path(state["repository"]).expanduser().resolve()
    trace_id = state.get("trace_id")
    return JsonlRunLogger.for_repository(repository, run_id=trace_id, trace_id=trace_id)


def openai_agents(planner: OpenAIPlanner) -> MultiAgentAgents:
    """Create production agents sharing the resilient model client."""

    def plan_agent(state: MultiAgentState, tools: ToolRegistry) -> dict[str, Any]:
        inspection = tools.call("planner", "inspect_repository")
        result = planner.create_plan(state["task"], inspection)
        return {
            "plan": asdict(result.plan),
            "selected_files": list(result.plan.files_to_inspect),
            "model": result.model,
            "usage": result.usage,
        }

    def proposal_agent(state: MultiAgentState, tools: ToolRegistry) -> dict[str, Any]:
        inspection = tools.call("implementer", "inspect_repository")
        selected = tuple(state.get("selected_files", ()))
        if not selected:
            raise PlannerError("planner selected no files for implementation")
        contents = {
            path: tools.call("implementer", "read_file", path=path)
            for path in selected
        }
        revision = state.get("review", {})
        task = state["task"]
        if revision and revision.get("suggestions"):
            task += "\nReviewer suggestions:\n" + "\n".join(revision["suggestions"])
        result = create_edit_proposal(planner, task, inspection, selected, contents)
        diff = tools.call(
            "reviewer",
            "preview_edit",
            path=result.proposal.path,
            new_content=result.proposal.new_content,
        )
        return {
            "proposal": asdict(result.proposal),
            "diff": diff,
            "model": result.model,
            "usage": result.usage,
        }

    def reviewer_agent(state: MultiAgentState, tools: ToolRegistry) -> dict[str, Any]:
        proposal = state.get("proposal")
        if not proposal:
            return {"approved": False, "issues": ["proposal is missing"], "suggestions": []}
        prompt = json.dumps({
            "task": state["task"],
            "proposal": proposal,
            "diff": state.get("diff", ""),
            "retrieved_context": state.get("retrieved_context", []),
            "test_command": state.get("test_command") or proposal.get("test_command"),
        }, sort_keys=True)
        system = (
            "You are PatchPilot's read-only code reviewer. Return only JSON with "
            "approved (boolean), issues (array of strings), suggestions (array of strings), "
            "and confidence (number 0 to 1). Reject unsafe, incomplete, or task-mismatched proposals."
        )
        try:
            response = planner.complete(
                temperature=0,
                max_tokens=800,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            )
            payload = json.loads(response.choices[0].message.content)
            approved = payload.get("approved") is True
            issues = payload.get("issues", [])
            suggestions = payload.get("suggestions", [])
            if not isinstance(issues, list) or not all(isinstance(item, str) for item in issues):
                raise ValueError("issues must be a string list")
            if not isinstance(suggestions, list) or not all(isinstance(item, str) for item in suggestions):
                raise ValueError("suggestions must be a string list")
            return {
                "approved": approved,
                "issues": issues[:10],
                "suggestions": suggestions[:10],
                "confidence": float(payload.get("confidence", 0.0)),
                "model": planner.last_model,
            }
        except (PlannerError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return {"approved": False, "issues": [f"review failed: {type(exc).__name__}"], "suggestions": []}

    return MultiAgentAgents(plan_agent, proposal_agent, reviewer_agent)


def _plan_node(state: MultiAgentState, agents: MultiAgentAgents) -> MultiAgentState:
    logger = _logger(state)
    logger.record("agent_started", agent="planner")
    result = agents.planner(state, build_repository_tool_registry(Path(state["repository"]), timeout=float(state.get("timeout", 60.0))))
    logger.record("agent_completed", agent="planner", selected_files=len(result.get("selected_files", [])))
    return result


def _retrieve_node(state: MultiAgentState) -> MultiAgentState:
    logger = _logger(state)
    logger.record("agent_started", agent="retriever")
    tools = build_repository_tool_registry(Path(state["repository"]), timeout=float(state.get("timeout", 60.0)))
    contexts = tools.call("retriever", "retrieve_context", query=state["task"])
    result = {"retrieved_context": [asdict(item) for item in contexts]}
    logger.record("agent_completed", agent="retriever", results=len(contexts))
    return result


def _proposal_node(state: MultiAgentState, agents: MultiAgentAgents) -> MultiAgentState:
    logger = _logger(state)
    revision_count = int(state.get("revision_count", 0))
    if state.get("review"):
        revision_count += 1
    next_state = dict(state)
    next_state["revision_count"] = revision_count
    logger.record("agent_started", agent="implementer", revision=revision_count)
    result = agents.proposer(next_state, build_repository_tool_registry(Path(state["repository"]), timeout=float(state.get("timeout", 60.0))))
    logger.record("agent_completed", agent="implementer", revision=revision_count)
    return {**result, "revision_count": revision_count, "status": "reviewing"}


def _review_node(state: MultiAgentState, agents: MultiAgentAgents) -> MultiAgentState:
    logger = _logger(state)
    logger.record("agent_started", agent="reviewer", revision=state.get("revision_count", 0))
    review = agents.reviewer(state, build_repository_tool_registry(Path(state["repository"]), timeout=float(state.get("timeout", 60.0))))
    approved = bool(review.get("approved"))
    logger.record("agent_completed", agent="reviewer", approved=approved)
    return {"review": review, "status": "awaiting_approval" if approved else "revision_needed"}


def _route_review(state: MultiAgentState) -> str:
    if state.get("review", {}).get("approved") is True:
        return "approval"
    if int(state.get("revision_count", 0)) < int(state.get("max_revisions", 2)):
        return "proposal"
    return "reject"


def _approval_node(state: MultiAgentState) -> MultiAgentState:
    if state.get("approved") is True:
        return {"status": "approved"}
    from .graph_workflow import _langgraph_api

    logger = _logger(state)
    _, _, _, _, _, interrupt, _ = _langgraph_api()
    logger.record("workflow_approval_waiting", agent="reviewer")
    decision = interrupt({
        "type": "multi_agent_edit_approval",
        "path": state["proposal"]["path"],
        "diff": state.get("diff", ""),
        "review": state.get("review", {}),
        "message": "Approve this reviewed multi-agent proposal before applying it?",
    })
    approved = decision is True or (isinstance(decision, dict) and decision.get("approved") is True)
    logger.record("workflow_approval_received", approved=approved)
    return {"approved": approved, "status": "approved" if approved else "rejected"}


def _route_approval(state: MultiAgentState) -> str:
    return "apply" if state.get("approved") else "reject"


def _apply_node(state: MultiAgentState) -> MultiAgentState:
    logger = _logger(state)
    tools = build_repository_tool_registry(Path(state["repository"]), timeout=float(state.get("timeout", 60.0)))
    proposal = state["proposal"]
    logger.record("executor_started", path=proposal["path"])
    tools.call("executor", "apply_edit", path=proposal["path"], new_content=proposal["new_content"])
    command = state.get("test_command") or proposal.get("test_command")
    if not command:
        return {"applied": True, "test_success": None, "status": "completed"}
    summary = tools.call("executor", "run_tests", command=command)
    logger.record("executor_completed", test_success=summary.success)
    return {"applied": True, "test_success": summary.success, "status": "completed" if summary.success else "tests_failed"}


def _reject_node(state: MultiAgentState) -> MultiAgentState:
    _logger(state).record("multi_agent_rejected", revision=state.get("revision_count", 0))
    return {"status": "rejected", "applied": False}


def build_multi_agent_graph(
    agents: MultiAgentAgents,
    *,
    checkpointer: Any | None = None,
) -> Any:
    """Build a bounded planner/retriever/implementer/reviewer graph."""
    StateGraph, START, END, MemorySaver, _, _, _ = _langgraph_api()
    graph = StateGraph(MultiAgentState)
    graph.add_node("planning", lambda state: _plan_node(state, agents))
    graph.add_node("retrieval", _retrieve_node)
    graph.add_node("proposing", lambda state: _proposal_node(state, agents))
    graph.add_node("reviewing", lambda state: _review_node(state, agents))
    graph.add_node("human_approval", _approval_node)
    graph.add_node("executing", _apply_node)
    graph.add_node("rejected", _reject_node)
    graph.add_edge(START, "planning")
    graph.add_edge("planning", "retrieval")
    graph.add_edge("retrieval", "proposing")
    graph.add_edge("proposing", "reviewing")
    graph.add_conditional_edges("reviewing", _route_review, {"proposal": "proposing", "approval": "human_approval", "reject": "rejected"})
    graph.add_conditional_edges("human_approval", _route_approval, {"apply": "executing", "reject": "rejected"})
    graph.add_edge("executing", END)
    graph.add_edge("rejected", END)
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())


def build_sqlite_multi_agent_graph(
    database_path: Path | str,
    agents: MultiAgentAgents,
) -> tuple[Any, sqlite3.Connection]:
    """Build the multi-agent graph with restart-safe SQLite checkpoints."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SQLite persistence is not installed; use pip install -e '.[persistence]'") from exc
    path = Path(database_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    return build_multi_agent_graph(agents, checkpointer=checkpointer), connection


def initial_multi_agent_state(
    repository: Path,
    task: str,
    *,
    test_command: str | None = None,
    timeout: float = 60.0,
    max_revisions: int = 2,
    trace_id: str | None = None,
) -> MultiAgentState:
    if not task.strip():
        raise ValueError("task cannot be empty")
    if max_revisions < 0 or max_revisions > 2:
        raise ValueError("max_revisions must be between 0 and 2")
    return {
        "task": task.strip(),
        "repository": str(repository.expanduser().resolve()),
        "test_command": test_command,
        "timeout": timeout,
        "max_revisions": max_revisions,
        "revision_count": 0,
        "trace_id": trace_id or uuid4().hex,
    }
