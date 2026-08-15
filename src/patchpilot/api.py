import re
from pathlib import Path
from typing import Any, Optional, Union

from .graph_workflow import build_sqlite_graph, initial_state, pending_interrupts, resume_approval
from .llm import PlannerError
from .safety import UnsafeCommand


_THREAD_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value") and hasattr(value, "resumable"):
        return {
            "value": _jsonable(value.value),
            "resumable": bool(value.resumable),
            "namespace": list(getattr(value, "ns", ()) or ()),
        }
    return value


def _validate_thread_id(thread_id: str) -> str:
    if not _THREAD_ID.fullmatch(thread_id):
        raise ValueError("thread_id must contain only letters, digits, '.', '_', ':', or '-' and be <=128 characters")
    return thread_id


def create_app(checkpoint_db: Union[Path, str] = ".patchpilot/checkpoints.sqlite") -> Any:
    """Create the optional FastAPI service backed by SQLite checkpoints."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover - exercised when optional extra is absent
        raise RuntimeError("FastAPI is not installed; use pip install -e '.[api]'") from exc

    class StartRequest(BaseModel):
        thread_id: str = Field(min_length=1, max_length=128)
        repository: str
        proposal_file: str
        test_command: Optional[str] = None
        timeout: float = Field(default=60.0, gt=0, le=600.0)
        recovery_proposal_file: Optional[str] = None
        approve_recovery: bool = False

    class ResumeRequest(BaseModel):
        approved: bool

    graph, connection = build_sqlite_graph(checkpoint_db)
    app = FastAPI(title="PatchPilot Workflow API", version="0.1.0")
    app.state.graph = graph
    app.state.checkpoint_connection = connection

    def config_for(thread_id: str) -> dict[str, Any]:
        try:
            return {"configurable": {"thread_id": _validate_thread_id(thread_id)}}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def result_payload(thread_id: str, result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        interrupts = pending_interrupts(graph, config)
        return {
            "thread_id": thread_id,
            "status": result.get("status"),
            "pending_approval": bool(interrupts),
            "interrupts": _jsonable(interrupts),
            "state": _jsonable(result),
        }

    def workflow_error(exc: Exception) -> HTTPException:
        return HTTPException(status_code=400, detail=str(exc))

    @app.post("/v1/workflows")
    def start_workflow(request: StartRequest) -> dict[str, Any]:
        config = config_for(request.thread_id)
        try:
            result = graph.invoke(
                initial_state(
                    Path(request.repository),
                    Path(request.proposal_file),
                    test_command=request.test_command,
                    timeout=request.timeout,
                    recovery_proposal_file=(Path(request.recovery_proposal_file) if request.recovery_proposal_file else None),
                    approve_recovery=request.approve_recovery,
                ),
                config=config,
            )
        except (PlannerError, UnsafeCommand, ValueError, OSError) as exc:
            raise workflow_error(exc) from exc
        return result_payload(request.thread_id, result, config)

    @app.get("/v1/workflows/{thread_id}")
    def workflow_status(thread_id: str) -> dict[str, Any]:
        config = config_for(thread_id)
        try:
            snapshot = graph.get_state(config)
        except (ValueError, OSError) as exc:
            raise workflow_error(exc) from exc
        if not snapshot.values and not snapshot.next:
            raise HTTPException(status_code=404, detail="workflow thread not found")
        return result_payload(thread_id, dict(snapshot.values), config)

    @app.post("/v1/workflows/{thread_id}/resume")
    def resume_workflow(thread_id: str, request: ResumeRequest) -> dict[str, Any]:
        config = config_for(thread_id)
        if not pending_interrupts(graph, config):
            raise HTTPException(status_code=409, detail="workflow is not waiting for approval")
        try:
            result = resume_approval(graph, config, approved=request.approved)
        except (PlannerError, UnsafeCommand, ValueError, OSError) as exc:
            raise workflow_error(exc) from exc
        return result_payload(thread_id, result, config)

    @app.on_event("shutdown")
    def close_checkpoint_connection() -> None:
        connection.close()

    return app

def app_factory() -> Any:
    return create_app()
