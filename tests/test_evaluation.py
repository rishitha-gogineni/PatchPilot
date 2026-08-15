import json
from pathlib import Path
from types import SimpleNamespace

from patchpilot.evaluation import evaluate_live_tasks, evaluate_tasks, load_tasks
from patchpilot.llm import OpenAIPlanner


def test_fixture_loads_five_tasks() -> None:
    path = Path(__file__).parent / "fixtures" / "coding_tasks.json"
    tasks = load_tasks(path)
    assert len(tasks) == 5
    assert tasks[0].task_id == "normalize-name"


def test_deterministic_benchmark_reports_success(tmp_path: Path) -> None:
    fixture = tmp_path / "tasks.json"
    fixture.write_text(json.dumps([{
        "task_id": "tiny",
        "description": "fix value",
        "test_command": "python -m pytest",
        "files": {
            "pyproject.toml": "[project]\nname='tiny'\n",
            "app.py": "def value():\n    raise NotImplementedError\n",
            "test_app.py": "from app import value\n\ndef test_value():\n    assert value() == 1\n",
        },
        "proposal": {
            "path": "app.py",
            "new_content": "def value():\n    return 1\n",
            "explanation": "Return the expected value.",
            "risks": [],
            "test_command": "python -m pytest",
        },
    }]), encoding="utf-8")
    report = evaluate_tasks(fixture)
    assert report["tasks"] == 1
    assert report["proposal_validity_rate"] == 1.0
    assert report["test_pass_rate"] == 1.0
    assert report["task_success_rate"] == 1.0
    assert report["model_calls"] == 0


class FakeLiveCompletions:
    def create(self, **kwargs: object) -> object:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "path": "app.py",
                "new_content": "def value():\n    return 1\n",
                "explanation": "Return the expected value.",
                "risks": [],
                "test_command": "python -m pytest",
            })))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )


def test_live_benchmark_reports_usage_and_cost_without_network(tmp_path: Path) -> None:
    fixture = tmp_path / "live-tasks.json"
    fixture.write_text(json.dumps([{
        "task_id": "tiny-live",
        "description": "fix value",
        "test_command": "python -m pytest",
        "files": {
            "pyproject.toml": "[project]\nname='tiny'\n",
            "app.py": "def value():\n    raise NotImplementedError\n",
            "test_app.py": "from app import value\n\ndef test_value():\n    assert value() == 1\n",
        },
        "proposal": {"path": "app.py"},
    }]), encoding="utf-8")
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeLiveCompletions()))
    report = evaluate_live_tasks(
        fixture,
        OpenAIPlanner(model="test-model", client=client),
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )
    assert report["model"] == "test-model"
    assert report["model_calls"] == 1
    assert report["input_tokens"] == 100
    assert report["output_tokens"] == 20
    assert report["estimated_cost"] == 0.00014
    assert report["task_success_rate"] == 1.0
