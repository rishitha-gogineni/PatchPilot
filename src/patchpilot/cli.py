from __future__ import annotations

import argparse
from pathlib import Path

from .inspector import RepositoryInspector
from .planner import make_plan
from .safety import UnsafeCommand, run_safe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="patchpilot", description="Safety-first coding assistant foundation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect", help="inspect a local repository")
    inspect.add_argument("repository", type=Path)
    plan = subparsers.add_parser("plan", help="create a deterministic review plan")
    plan.add_argument("task")
    plan.add_argument("--repo", type=Path, default=Path.cwd())
    check = subparsers.add_parser("run", help="run one allow-listed command")
    check.add_argument("command")
    check.add_argument("--repo", type=Path, default=Path.cwd())
    check.add_argument("--timeout", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inspector = RepositoryInspector()
    if args.command == "inspect":
        result = inspector.inspect(args.repository)
        print(f"Repository: {result.root}\nProject type: {result.project_type}")
        print(f"Markers: {', '.join(result.markers) or 'none'}")
        print(f"Files ({len(result.files)}):")
        print("\n".join(f"  {name}" for name in result.files) or "  (none)")
        print("Test commands: " + (", ".join(result.test_commands) or "none detected"))
        return 0
    if args.command == "plan":
        inspection = inspector.inspect(args.repo)
        task_plan = make_plan(args.task, inspection)
        print(f"Task: {task_plan.task}\nRepository: {task_plan.repository}\nApproval required: yes")
        for step in task_plan.steps:
            print(f"{step.order}. {step.action} — {step.rationale}")
        return 0
    try:
        result = run_safe(args.command, args.repo, args.timeout)
    except (UnsafeCommand, ValueError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    return result.returncode
