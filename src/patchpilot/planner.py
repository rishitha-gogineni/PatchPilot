from .models import Inspection, PlanStep, TaskPlan


def make_plan(task: str, inspection: Inspection) -> TaskPlan:
    task = task.strip()
    if not task:
        raise ValueError("task cannot be empty")
    test_command = inspection.test_commands[0] if inspection.test_commands else "No test command detected"
    steps = (
        PlanStep(1, "Inspect relevant files", "Understand the repository before editing."),
        PlanStep(2, "Implement the smallest targeted change", "Keep the patch focused and reviewable."),
        PlanStep(3, f"Run tests: {test_command}", "Check that the change does not regress existing behavior."),
        PlanStep(4, "Show the diff and wait for approval", "Human review is required before any commit or push."),
    )
    return TaskPlan(task, inspection.root, steps)
