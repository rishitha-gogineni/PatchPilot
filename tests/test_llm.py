from pathlib import Path
from types import SimpleNamespace

import pytest

from patchpilot.llm import OpenAIPlanner, PlannerError, build_planner_context, validate_model_plan
from patchpilot.models import Inspection


def inspection() -> Inspection:
    return Inspection(Path("/repo"), "python", ("pyproject.toml", "src/app.py"), ("python -m pytest",), ("pyproject.toml",))


def valid_payload() -> dict[str, object]:
    return {
        "goal": "Fix the parser",
        "files_to_inspect": ["src/app.py"],
        "proposed_changes": ["Update the parser branch"],
        "test_command": "python -m pytest",
        "risks": ["Existing parser behavior may change"],
    }


def test_context_is_bounded_and_contains_no_source_content() -> None:
    context = build_planner_context("fix it", inspection())
    assert "src/app.py" in context
    assert "detected_test_commands" in context
    assert "source" not in context


def test_model_plan_rejects_unknown_files_and_commands() -> None:
    payload = valid_payload()
    payload["files_to_inspect"] = ["missing.py"]
    with pytest.raises(PlannerError):
        validate_model_plan(payload, inspection())
    payload = valid_payload()
    payload["test_command"] = "rm -rf ."
    with pytest.raises(PlannerError):
        validate_model_plan(payload, inspection())


class FakeCompletions:
    def create(self, **kwargs: object) -> object:
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["temperature"] == 0
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"goal":"Fix the parser","files_to_inspect":["src/app.py"],"proposed_changes":["Update parser"],"test_command":"python -m pytest","risks":[]}'))],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=40),
        )


class FakeClient:
    chat = SimpleNamespace(completions=FakeCompletions())


def test_openai_planner_returns_validated_result_without_network() -> None:
    result = OpenAIPlanner(model="test-model", client=FakeClient()).create_plan("fix parser", inspection())
    assert result.model == "test-model"
    assert result.plan.goal == "Fix the parser"
    assert result.usage == {"input_tokens": 120, "output_tokens": 40}


def test_api_key_is_not_needed_until_live_client_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    planner = OpenAIPlanner(client=None)
    with pytest.raises(PlannerError, match="OPENAI_API_KEY"):
        _ = planner.client
