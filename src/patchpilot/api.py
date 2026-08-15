import re
import time
import hmac
import os
from pathlib import Path
from typing import Any, Optional, Union
from uuid import uuid4

from .graph_workflow import build_sqlite_graph, initial_state, pending_interrupts, resume_approval
from .logging import JsonlRunLogger
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


def create_app(
    checkpoint_db: Union[Path, str] = ".patchpilot/checkpoints.sqlite",
    api_key: Optional[str] = None,
) -> Any:
    """Create the optional FastAPI service backed by SQLite checkpoints.

    Authentication is enabled when ``api_key`` or ``PATCHPILOT_API_KEY`` is
    configured. The health endpoint remains public for deployment probes.
    """
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse
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
    api_logger = JsonlRunLogger(
        Path(checkpoint_db).expanduser().resolve().parent / "api-runs.jsonl",
        trace_id="api",
    )
    configured_api_key = api_key if api_key is not None else os.getenv("PATCHPILOT_API_KEY")
    configured_api_key = configured_api_key.strip() if configured_api_key else None
    if configured_api_key and len(configured_api_key) < 16:
        raise ValueError("PATCHPILOT_API_KEY must be at least 16 characters")

    @app.middleware("http")
    async def request_trace(request: Request, call_next: Any) -> Any:
        candidate = request.headers.get("X-Request-ID", "").strip()
        request_id = candidate if 0 < len(candidate) <= 128 and _THREAD_ID.fullmatch(candidate) else uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            if configured_api_key and request.url.path.startswith("/v1/"):
                authorization = request.headers.get("Authorization", "")
                presented = request.headers.get("X-API-Key", "")
                if authorization.startswith("Bearer "):
                    presented = authorization[7:].strip()
                if not presented or not hmac.compare_digest(presented, configured_api_key):
                    response = JSONResponse(
                        status_code=401,
                        content={"detail": "authentication required"},
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                else:
                    response = await call_next(request)
            else:
                response = await call_next(request)
        except Exception:
            api_logger.record(
                "api_request",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=500,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                error="unhandled_exception",
            )
            raise
        response.headers["X-Request-ID"] = request_id
        api_logger.record(
            "api_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return response

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    def config_for(thread_id: str) -> dict[str, Any]:
        try:
            return {"configurable": {"thread_id": _validate_thread_id(thread_id)}}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def result_payload(
        thread_id: str,
        result: dict[str, Any],
        config: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        interrupts = pending_interrupts(graph, config)
        return {
            "thread_id": thread_id,
            "request_id": request_id,
            "status": result.get("status"),
            "pending_approval": bool(interrupts),
            "interrupts": _jsonable(interrupts),
            "state": _jsonable(result),
        }

    def workflow_error(exc: Exception) -> HTTPException:
        return HTTPException(status_code=400, detail=str(exc))

    @app.post("/v1/workflows")
    def start_workflow(payload: StartRequest, http_request: Request) -> dict[str, Any]:
        request_id = http_request.state.request_id
        config = config_for(payload.thread_id)
        try:
            result = graph.invoke(
                initial_state(
                    Path(payload.repository),
                    Path(payload.proposal_file),
                    test_command=payload.test_command,
                    timeout=payload.timeout,
                    recovery_proposal_file=(Path(payload.recovery_proposal_file) if payload.recovery_proposal_file else None),
                    approve_recovery=payload.approve_recovery,
                    trace_id=request_id,
                ),
                config=config,
            )
        except (PlannerError, UnsafeCommand, ValueError, OSError) as exc:
            raise workflow_error(exc) from exc
        return result_payload(payload.thread_id, result, config, request_id)

    @app.get("/v1/workflows/{thread_id}")
    def workflow_status(thread_id: str, http_request: Request) -> dict[str, Any]:
        config = config_for(thread_id)
        try:
            snapshot = graph.get_state(config)
        except (ValueError, OSError) as exc:
            raise workflow_error(exc) from exc
        if not snapshot.values and not snapshot.next:
            raise HTTPException(status_code=404, detail="workflow thread not found")
        return result_payload(thread_id, dict(snapshot.values), config, http_request.state.request_id)

    @app.post("/v1/workflows/{thread_id}/resume")
    def resume_workflow(thread_id: str, payload: ResumeRequest, http_request: Request) -> dict[str, Any]:
        config = config_for(thread_id)
        if not pending_interrupts(graph, config):
            raise HTTPException(status_code=409, detail="workflow is not waiting for approval")
        try:
            result = resume_approval(graph, config, approved=payload.approved)
        except (PlannerError, UnsafeCommand, ValueError, OSError) as exc:
            raise workflow_error(exc) from exc
        return result_payload(thread_id, result, config, http_request.state.request_id)

    @app.on_event("shutdown")
    def close_checkpoint_connection() -> None:
        connection.close()

    return app

def app_factory() -> Any:
    return create_app()
