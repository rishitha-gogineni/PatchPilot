import json
from pathlib import Path

from patchpilot.evaluation import evaluate_tasks, load_tasks


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
