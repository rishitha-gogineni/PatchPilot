"""Optional model planning behind the deterministic PatchPilot safety boundary."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Protocol

from .models import Inspection, ModelPlan, PlannerResult


class PlannerError(ValueError):
    """Raised when a model plan is unavailable or fails schema validation."""


class Planner(Protocol):
    def create_plan(self, task: str, inspection: Inspection) -> PlannerResult:
        ...


def build_planner_context(task: str, inspection: Inspection, *, max_files: int = 80) -> str:
    """Build a bounded context summary; source files and secrets are not sent."""
    if not task.strip():
        raise PlannerError("task cannot be empty")
    files = inspection.files[:max_files]
    return json.dumps({
        "task": task.strip(),
        "project_type": inspection.project_type,
        "repository_markers": list(inspection.markers),
        "files": list(files),
        "files_truncated": len(inspection.files) > max_files,
        "detected_test_commands": list(inspection.test_commands),
    }, sort_keys=True)


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlannerError(f"plan field '{field}' must be a non-empty string")
    return value.strip()


def _string_list(value: Any, field: str, *, max_items: int = 20) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > max_items or any(not isinstance(item, str) or not item.strip() for item in value):
        raise PlannerError(f"plan field '{field}' must be a list of at most {max_items} non-empty strings")
    return tuple(item.strip() for item in value)


def validate_model_plan(payload: Any, inspection: Inspection) -> ModelPlan:
    if not isinstance(payload, dict):
        raise PlannerError("model output must be a JSON object")
    goal = _required_string(payload.get("goal"), "goal")
    files = _string_list(payload.get("files_to_inspect"), "files_to_inspect")
    changes = _string_list(payload.get("proposed_changes"), "proposed_changes")
    risks = _string_list(payload.get("risks"), "risks")
    test_command = payload.get("test_command")
    if test_command is not None and (not isinstance(test_command, str) or not test_command.strip()):
        raise PlannerError("plan field 'test_command' must be a string or null")
    if test_command and inspection.test_commands and test_command.strip() not in inspection.test_commands:
        raise PlannerError("model-selected test command was not detected in the repository")
    root_files = set(inspection.files)
    unknown = [path for path in files if path not in root_files]
    if unknown:
        raise PlannerError(f"model referenced files not present in inspection: {unknown}")
    return ModelPlan(goal, files, changes, test_command.strip() if test_command else None, risks)


class OpenAIPlanner:
    """Generate a JSON plan; it cannot execute tools or edit repositories."""

    def __init__(self, model: str | None = None, client: Any | None = None):
        self.model = model or os.getenv("PATCHPILOT_MODEL", "gpt-4o-mini")
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise PlannerError("OPENAI_API_KEY is required only when running llm-plan")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise PlannerError("install the optional llm dependency: pip install -e '.[llm]'") from exc
            self._client = OpenAI(api_key=api_key)
        return self._client

    def create_plan(self, task: str, inspection: Inspection) -> PlannerResult:
        context = build_planner_context(task, inspection)
        system = (
            "You are PatchPilot's planning component. Return only valid JSON with keys "
            "goal, files_to_inspect, proposed_changes, test_command, risks. "
            "Do not invent files or commands. You do not edit files or run tools."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=1000,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": context}],
            )
            content = response.choices[0].message.content
            payload = json.loads(content)
        except PlannerError:
            raise
        except Exception as exc:
            raise PlannerError(f"model planning request failed: {type(exc).__name__}") from exc
        plan = validate_model_plan(payload, inspection)
        raw_usage = getattr(response, "usage", None)
        usage = {
            "input_tokens": int(getattr(raw_usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(raw_usage, "completion_tokens", 0) or 0),
        }
        return PlannerResult(plan=plan, model=self.model, usage=usage)


def plan_to_dict(result: PlannerResult) -> dict[str, Any]:
    return {"model": result.model, "usage": result.usage, "plan": asdict(result.plan)}
