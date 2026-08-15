from __future__ import annotations

import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, TypedDict

from .logging import JsonlRunLogger
from .workflow import _load_validated_proposal, apply_proposal


class CodingWorkflowState(TypedDict, total=False):
    repository: str
    proposal_file: str
    recovery_proposal_file: Optional[str]
    approve_recovery: bool
    test_command: Optional[str]
    timeout: float
    proposal: dict[str, Any]
    diff: str
    approved: bool
    applied: bool
    test_success: Optional[bool]
    recovery_attempted: bool
    status: str
    error: Optional[str]


def _langgraph_api() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Command, interrupt
    except ImportError as exc:  # pragma: no cover - exercised when optional extra is absent
        raise RuntimeError("LangGraph is not installed; use pip install -e '.[graph]'") from exc
    return StateGraph, START, END, MemorySaver, Command, interrupt, None


def _prepare_node(state: CodingWorkflowState) -> CodingWorkflowState:
    repository = Path(state["repository"]).expanduser().resolve()
    proposal = _load_validated_proposal(repository, Path(state["proposal_file"]))
    from .editor import preview_edit

    return {
        "proposal": asdict(proposal),
        "diff": preview_edit(repository, proposal.path, proposal.new_content),
        "status": "awaiting_approval",
    }


def _approval_node(state: CodingWorkflowState) -> CodingWorkflowState:
    if state.get("approved") is True:
        return {"status": "approved"}
    _, _, _, _, _, interrupt, _ = _langgraph_api()
    decision = interrupt({
        "type": "edit_approval",
        "path": state["proposal"]["path"],
        "diff": state.get("diff", ""),
        "message": "Approve this complete-file proposal before applying it?",
    })
    approved = decision is True or (isinstance(decision, dict) and decision.get("approved") is True)
    return {"approved": approved, "status": "approved" if approved else "rejected"}


def _route_after_approval(state: CodingWorkflowState) -> str:
    return "apply" if state.get("approved") else "reject"


def _apply_node(state: CodingWorkflowState) -> CodingWorkflowState:
    repository = Path(state["repository"]).expanduser().resolve()
    logger = JsonlRunLogger.for_repository(repository)
    result = apply_proposal(
        repository,
        Path(state["proposal_file"]),
        approved=True,
        test_command=state.get("test_command"),
        timeout=float(state.get("timeout", 60.0)),
        recovery_proposal_file=(Path(state["recovery_proposal_file"]) if state.get("recovery_proposal_file") else None),
        recovery_approved=bool(state.get("approve_recovery", False)),
        logger=logger.record,
    )
    summary = result.test_summary
    return {
        "applied": result.applied,
        "test_success": summary.success if summary else None,
        "recovery_attempted": summary.recovery_attempted if summary else False,
        "status": "completed" if summary is None or summary.success else "tests_failed",
    }


def _reject_node(state: CodingWorkflowState) -> CodingWorkflowState:
    return {"status": "rejected", "applied": False}


def build_apply_graph(checkpointer: Any | None = None) -> Any:
    """Build the approval-gated proposal graph.

    The default ``MemorySaver`` supports pause/resume within one process. A
    durable checkpointer can be injected by an API or worker deployment.
    """
    StateGraph, START, END, MemorySaver, _, _, _ = _langgraph_api()
    graph = StateGraph(CodingWorkflowState)
    graph.add_node("prepare", _prepare_node)
    graph.add_node("approval", _approval_node)
    graph.add_node("apply", _apply_node)
    graph.add_node("reject", _reject_node)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "approval")
    graph.add_conditional_edges("approval", _route_after_approval, {"apply": "apply", "reject": "reject"})
    graph.add_edge("apply", END)
    graph.add_edge("reject", END)
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())


def build_sqlite_graph(database_path: Path | str) -> tuple[Any, sqlite3.Connection]:
    """Build a graph backed by a SQLite checkpoint database.

    The caller owns the returned connection and must close it when the process
    or application shuts down.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:  # pragma: no cover - exercised when optional extra is absent
        raise RuntimeError("SQLite persistence is not installed; use pip install -e '.[persistence]'") from exc
    path = Path(database_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    return build_apply_graph(checkpointer=checkpointer), connection


def pending_interrupts(graph: Any, config: dict[str, Any]) -> list[Any]:
    snapshot = graph.get_state(config)
    return [
        interrupt
        for task in snapshot.tasks
        for interrupt in (getattr(task, "interrupts", ()) or ())
    ]


def initial_state(
    repository: Path,
    proposal_file: Path,
    *,
    test_command: str | None = None,
    timeout: float = 60.0,
    recovery_proposal_file: Path | None = None,
    approve_recovery: bool = False,
) -> CodingWorkflowState:
    return {
        "repository": str(repository.expanduser().resolve()),
        "proposal_file": str(proposal_file.expanduser().resolve()),
        "test_command": test_command,
        "timeout": timeout,
        "recovery_proposal_file": str(recovery_proposal_file.expanduser().resolve()) if recovery_proposal_file else None,
        "approve_recovery": approve_recovery,
    }


def resume_approval(graph: Any, config: dict[str, Any], approved: bool) -> Any:
    """Resume a paused graph with a human approval decision."""
    _, _, _, _, Command, _, _ = _langgraph_api()
    return graph.invoke(Command(resume={"approved": approved}), config=config)
