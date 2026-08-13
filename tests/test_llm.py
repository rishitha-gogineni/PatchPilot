from pathlib import Path
from types import SimpleNamespace

import pytest

from patchpilot.llm import OpenAIPlanner, PlannerError, build_planner_context, create_edit_proposal, validate_edit_proposal, validate_model_plan
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


def test_edit_proposal_rejects_unselected_path() -> None:
    payload = {"path": "pyproject.toml", "new_content": "x", "explanation": "change", "risks": [], "test_command": "python -m pytest"}
    with pytest.raises(PlannerError):
        validate_edit_proposal(payload, inspection(), ("src/app.py",))


def test_edit_proposal_accepts_single_risk_string() -> None:
    payload = {"path": "src/app.py", "new_content": "x", "explanation": "change", "risks": "none", "test_command": "python -m pytest"}
    proposal = validate_edit_proposal(payload, inspection(), ("src/app.py",))
    assert proposal.risks == ("none",)


def test_edit_proposal_context_rejects_truncated_replacement() -> None:
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeProposalCompletions()))
    long_source = "x = 1\n" * 200
    with pytest.raises(PlannerError, match="truncated"):
        create_edit_proposal(OpenAIPlanner(model="test-model", client=client), "fix parser", inspection(), ("src/app.py",), {"src/app.py": long_source})


def test_edit_proposal_rejects_control_characters() -> None:
    payload = {"path": "src/app.py", "new_content": "ok\x14\n", "explanation": "change", "risks": [], "test_command": "python -m pytest"}
    with pytest.raises(PlannerError, match="control characters"):
        validate_edit_proposal(payload, inspection(), ("src/app.py",))


class FakeProposalCompletions:
    def create(self, **kwargs: object) -> object:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"path":"src/app.py","new_content":"print(\\"fixed\\")\\n","explanation":"Fix parser output","risks":[],"test_command":"python -m pytest"}'))],
            usage=SimpleNamespace(prompt_tokens=200, completion_tokens=60),
        )


def test_edit_proposal_is_validated_without_network() -> None:
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeProposalCompletions()))
    result = create_edit_proposal(OpenAIPlanner(model="test-model", client=client), "fix parser", inspection(), ("src/app.py",), {"src/app.py": "print('old')\n"})
    assert result.proposal.path == "src/app.py"
    assert "fixed" in result.proposal.new_content
    assert result.usage == {"input_tokens": 200, "output_tokens": 60}
