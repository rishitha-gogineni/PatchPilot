from __future__ import annotations

import argparse
import json
from pathlib import Path

from .editor import EditDenied, apply_edit, preview_edit
from .evaluation import evaluate_live_tasks, evaluate_tasks
from .inspector import RepositoryInspector
from .logging import JsonlRunLogger
from .llm import OpenAIPlanner, PlannerError, create_edit_proposal, plan_to_dict, proposal_to_dict
from .orchestrator import run_test_loop
from .planner import make_plan
from .safety import UnsafeCommand, run_safe
from .workflow import apply_proposal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="patchpilot", description="Safety-first coding assistant foundation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect", help="inspect a local repository")
    inspect.add_argument("repository", type=Path)
    plan = subparsers.add_parser("plan", help="create a deterministic review plan")
    plan.add_argument("task")
    plan.add_argument("--repo", type=Path, default=Path.cwd())
    check = subparsers.add_parser("run", help="run one allow-listed command")
    check.add_argument("command_text")
    check.add_argument("--repo", type=Path, default=Path.cwd())
    check.add_argument("--timeout", type=float, default=30.0)
    edit = subparsers.add_parser("edit", help="preview and optionally apply one approved file edit")
    edit.add_argument("file")
    edit.add_argument("--content-file", type=Path, required=True)
    edit.add_argument("--repo", type=Path, default=Path.cwd())
    edit.add_argument("--approve", action="store_true")
    logs = subparsers.add_parser("logs", help="show recent structured run events")
    logs.add_argument("--repo", type=Path, default=Path.cwd())
    logs.add_argument("--limit", type=int, default=20)
    test = subparsers.add_parser("test", help="run a test command with one optional recovery callback")
    test.add_argument("test_command")
    test.add_argument("--repo", type=Path, default=Path.cwd())
    test.add_argument("--timeout", type=float, default=60.0)
    test.add_argument("--recovery-content-file", type=Path)
    test.add_argument("--recovery-path")
    test.add_argument("--approve-recovery", action="store_true")
    ai_plan = subparsers.add_parser("llm-plan", help="generate a structured plan with the configured model")
    ai_plan.add_argument("task")
    ai_plan.add_argument("--repo", type=Path, default=Path.cwd())
    ai_plan.add_argument("--model")
    ai_plan.add_argument("--fallback-model")
    ai_plan.add_argument("--timeout", type=float, default=45.0)
    ai_plan.add_argument("--retries", type=int, default=2)
    propose = subparsers.add_parser("propose", help="generate a review-only edit proposal")
    propose.add_argument("task")
    propose.add_argument("files", nargs="+", help="explicit repository-relative files to send for review")
    propose.add_argument("--repo", type=Path, default=Path.cwd())
    propose.add_argument("--model")
    propose.add_argument("--fallback-model")
    propose.add_argument("--timeout", type=float, default=45.0)
    propose.add_argument("--retries", type=int, default=2)
    propose.add_argument("--json", action="store_true", help="print only the machine-readable proposal JSON")
    apply = subparsers.add_parser("apply-proposal", help="apply a reviewed proposal after explicit approval")
    apply.add_argument("proposal_file", type=Path)
    apply.add_argument("--repo", type=Path, default=Path.cwd())
    apply.add_argument("--approve", action="store_true")
    apply.add_argument("--test-command")
    apply.add_argument("--timeout", type=float, default=60.0)
    apply.add_argument("--recovery-proposal-file", type=Path)
    apply.add_argument("--approve-recovery", action="store_true")
    graph = subparsers.add_parser("graph", help="run the LangGraph approval workflow")
    graph.add_argument("proposal_file", type=Path)
    graph.add_argument("--repo", type=Path, default=Path.cwd())
    graph.add_argument("--test-command")
    graph.add_argument("--timeout", type=float, default=60.0)
    graph.add_argument("--recovery-proposal-file", type=Path)
    graph.add_argument("--approve-recovery", action="store_true")
    graph.add_argument("--thread-id", default="patchpilot-local")
    graph.add_argument("--checkpoint-db", type=Path)
    graph.add_argument("--resume", action="store_true", help="resume an existing SQLite-backed thread")
    decision = graph.add_mutually_exclusive_group()
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", action="store_true")
    multi = subparsers.add_parser("multi-agent", help="run the bounded multi-agent review workflow")
    multi.add_argument("task")
    multi.add_argument("--repo", type=Path, default=Path.cwd())
    multi.add_argument("--test-command")
    multi.add_argument("--timeout", type=float, default=60.0)
    multi.add_argument("--max-revisions", type=int, default=2)
    multi.add_argument("--model")
    multi.add_argument("--fallback-model")
    multi.add_argument("--retries", type=int, default=2)
    multi.add_argument("--thread-id", default="patchpilot-multi-agent")
    multi.add_argument("--checkpoint-db", type=Path)
    multi.add_argument("--resume", action="store_true")
    multi_decision = multi.add_mutually_exclusive_group()
    multi_decision.add_argument("--approve", action="store_true")
    multi_decision.add_argument("--reject", action="store_true")
    evaluate = subparsers.add_parser("evaluate", help="run the deterministic coding-task benchmark")
    evaluate.add_argument("fixture", type=Path)
    evaluate.add_argument("--json", action="store_true")
    live_evaluate = subparsers.add_parser("evaluate-live", help="run the opt-in live-model benchmark")
    live_evaluate.add_argument("fixture", type=Path)
    live_evaluate.add_argument("--model")
    live_evaluate.add_argument("--fallback-model")
    live_evaluate.add_argument("--timeout", type=float, default=45.0)
    live_evaluate.add_argument("--retries", type=int, default=2)
    live_evaluate.add_argument("--input-cost-per-million", type=float, default=0.15)
    live_evaluate.add_argument("--output-cost-per-million", type=float, default=0.60)
    live_evaluate.add_argument("--test-timeout", type=float, default=30.0)
    live_evaluate.add_argument("--json", action="store_true")
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
    if args.command == "llm-plan":
        logger = JsonlRunLogger.for_repository(args.repo)
        try:
            inspection = inspector.inspect(args.repo)
            result = OpenAIPlanner(
                model=args.model,
                fallback_model=args.fallback_model,
                request_timeout=args.timeout,
                max_retries=args.retries,
                logger=logger.record,
            ).create_plan(args.task, inspection)
        except (PlannerError, ValueError) as exc:
            logger.record("model_plan_blocked", reason=str(exc))
            print(f"LLM PLAN BLOCKED: {exc}")
            return 2
        logger.record("model_plan_created", model=result.model, **result.usage)
        print(json.dumps(plan_to_dict(result), indent=2, sort_keys=True))
        print("Approval required before any edit or command execution.")
        return 0
    if args.command == "propose":
        logger = JsonlRunLogger.for_repository(args.repo)
        try:
            inspection = inspector.inspect(args.repo)
            selected = tuple(args.files)
            contents = {path: inspector.read_file(args.repo, path) for path in selected}
            result = create_edit_proposal(
                OpenAIPlanner(
                    model=args.model,
                    fallback_model=args.fallback_model,
                    request_timeout=args.timeout,
                    max_retries=args.retries,
                    logger=logger.record,
                ),
                args.task,
                inspection,
                selected,
                contents,
            )
            diff = preview_edit(args.repo, result.proposal.path, result.proposal.new_content)
            logger.record("edit_proposal_created", model=result.model, path=result.proposal.path, **result.usage)
        except (PlannerError, ValueError, OSError) as exc:
            logger.record("edit_proposal_blocked", reason=str(exc))
            print(f"PROPOSAL BLOCKED: {exc}")
            return 2
        proposal_json = proposal_to_dict(result)
        if args.json:
            print(json.dumps(proposal_json, indent=2, sort_keys=True))
            return 0
        print(json.dumps(proposal_json, indent=2, sort_keys=True))
        print("\nPROPOSED DIFF (review only; no files changed):")
        print(diff or "No changes.")
        print("Approval is required before applying this proposal.")
        return 0
    if args.command == "graph":
        try:
            from .graph_workflow import build_apply_graph, build_sqlite_graph, initial_state, pending_interrupts, resume_approval

            graph_connection = None
            if args.checkpoint_db:
                graph_app, graph_connection = build_sqlite_graph(args.checkpoint_db)
            else:
                graph_app = build_apply_graph()
            config = {"configurable": {"thread_id": args.thread_id}}
            if args.resume:
                if not (args.approve or args.reject):
                    raise ValueError("--resume requires --approve or --reject")
                result = resume_approval(graph_app, config, approved=args.approve)
            else:
                first = graph_app.invoke(
                    initial_state(
                        args.repo,
                        args.proposal_file,
                        test_command=args.test_command,
                        timeout=args.timeout,
                        recovery_proposal_file=args.recovery_proposal_file,
                        approve_recovery=args.approve_recovery,
                    ),
                    config=config,
                )
                pending = pending_interrupts(graph_app, config)
                if pending:
                    if args.approve or args.reject:
                        result = resume_approval(graph_app, config, approved=args.approve)
                    else:
                        print(json.dumps({"state": first, "interrupts": pending}, indent=2, sort_keys=True, default=str))
                        print("Approval required; graph paused without applying files.")
                        return 3
                else:
                    result = first
        except (EditDenied, PlannerError, UnsafeCommand, RuntimeError, ValueError, OSError) as exc:
            print(f"GRAPH BLOCKED: {exc}")
            return 2
        finally:
            if 'graph_connection' in locals() and graph_connection is not None:
                graph_connection.close()
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result.get("status") in {"completed", "rejected"} else 1
    if args.command == "multi-agent":
        graph_connection = None
        logger = JsonlRunLogger.for_repository(args.repo)
        try:
            from .graph_workflow import pending_interrupts
            from .multi_agent import (
                build_multi_agent_graph,
                build_sqlite_multi_agent_graph,
                initial_multi_agent_state,
                openai_agents,
            )
            planner = OpenAIPlanner(
                model=args.model,
                fallback_model=args.fallback_model,
                request_timeout=45.0,
                max_retries=args.retries,
                logger=logger.record,
            )
            agents = openai_agents(planner)
            if args.checkpoint_db:
                graph_app, graph_connection = build_sqlite_multi_agent_graph(args.checkpoint_db, agents)
            else:
                graph_app = build_multi_agent_graph(agents)
            config = {"configurable": {"thread_id": args.thread_id}}
            if args.resume:
                if not (args.approve or args.reject):
                    raise ValueError("--resume requires --approve or --reject")
                from langgraph.types import Command
                result = graph_app.invoke(Command(resume={"approved": args.approve}), config=config)
            else:
                first = graph_app.invoke(
                    initial_multi_agent_state(
                        args.repo,
                        args.task,
                        test_command=args.test_command,
                        timeout=args.timeout,
                        max_revisions=args.max_revisions,
                    ),
                    config=config,
                )
                pending = pending_interrupts(graph_app, config)
                if pending:
                    if args.approve or args.reject:
                        from langgraph.types import Command
                        result = graph_app.invoke(Command(resume={"approved": args.approve}), config=config)
                    else:
                        print(json.dumps({"state": first, "interrupts": pending}, indent=2, sort_keys=True, default=str))
                        print("Approval required; multi-agent graph paused without applying files.")
                        return 3
                else:
                    result = first
        except (PlannerError, UnsafeCommand, RuntimeError, ValueError, OSError) as exc:
            print(f"MULTI-AGENT BLOCKED: {exc}")
            return 2
        finally:
            if graph_connection is not None:
                graph_connection.close()
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result.get("status") in {"completed", "rejected"} else 1
    if args.command == "apply-proposal":
        logger = JsonlRunLogger.for_repository(args.repo)
        try:
            result = apply_proposal(
                args.repo,
                args.proposal_file,
                approved=args.approve,
                test_command=args.test_command,
                timeout=args.timeout,
                recovery_proposal_file=args.recovery_proposal_file,
                recovery_approved=args.approve_recovery,
                logger=logger.record,
            )
        except (EditDenied, PlannerError, UnsafeCommand, ValueError, OSError) as exc:
            logger.record("proposal_workflow_blocked", reason=str(exc))
            print(f"APPLY BLOCKED: {exc}")
            return 2
        print(result.diff or "No changes.")
        if not result.approved:
            print("Approval required; no files changed.")
            return 3
        print(f"Proposal applied: {result.proposal.path}")
        if result.test_summary is None:
            print("Tests: skipped (no test command provided)")
            return 0
        final = result.test_summary.recovery if result.test_summary.recovery_attempted else result.test_summary.initial
        if final.stdout:
            print(final.stdout, end="")
        if final.stderr:
            print(final.stderr, end="")
        print(f"Test success: {'yes' if result.test_summary.success else 'no'}")
        print(f"Recovery attempted: {'yes' if result.test_summary.recovery_attempted else 'no'}")
        return 0 if result.test_summary.success else final.returncode
    if args.command == "evaluate":
        try:
            report = evaluate_tasks(args.fixture)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"EVALUATION BLOCKED: {exc}")
            return 2
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"Fixture: {report['fixture']}")
            print(f"Tasks: {report['tasks']}")
            print(f"Proposal validity: {report['proposal_validity_rate']:.2%}")
            print(f"Test pass rate: {report['test_pass_rate']:.2%}")
            print(f"Task success rate: {report['task_success_rate']:.2%}")
            print(f"Average latency: {report['total_latency_ms'] / report['tasks']:.2f} ms")
            print("Model calls: 0 (deterministic fixture proposals)")
        return 0
    if args.command == "evaluate-live":
        try:
            report = evaluate_live_tasks(
                args.fixture,
                OpenAIPlanner(
                    model=args.model,
                    fallback_model=args.fallback_model,
                    request_timeout=args.timeout,
                    max_retries=args.retries,
                ),
                input_cost_per_million=args.input_cost_per_million,
                output_cost_per_million=args.output_cost_per_million,
                test_timeout=args.test_timeout,
            )
        except (PlannerError, ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"LIVE EVALUATION BLOCKED: {exc}")
            return 2
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"Fixture: {report['fixture']}")
            print(f"Model: {report['model']}")
            print(f"Tasks: {report['tasks']}")
            print(f"Proposal validity: {report['proposal_validity_rate']:.2%}")
            print(f"Test pass rate: {report['test_pass_rate']:.2%}")
            print(f"Task success rate: {report['task_success_rate']:.2%}")
            print(f"Average latency: {report['total_latency_ms'] / report['tasks']:.2f} ms")
            print(f"Model calls: {report['model_calls']}")
            print(f"Tokens: {report['input_tokens']} input / {report['output_tokens']} output")
            print(f"Estimated cost: ${report['estimated_cost']:.6f}")
        return 0
    logger = JsonlRunLogger.for_repository(args.repo) if args.command in {"edit", "test"} else None
    if args.command == "edit":
        try:
            new_content = args.content_file.read_text(encoding="utf-8")
            diff = preview_edit(args.repo, args.file, new_content)
            print(diff or "No changes.")
            if not args.approve:
                print("Approval required; no files changed.")
                return 3
            applied_diff = apply_edit(args.repo, args.file, new_content, approved=True)
            logger.record("edit_applied", path=args.file, changed=bool(applied_diff))
            print("Edit applied.")
            return 0
        except (EditDenied, ValueError, OSError) as exc:
            if logger:
                logger.record("edit_blocked", path=args.file, reason=str(exc))
            print(f"EDIT BLOCKED: {exc}")
            return 2
    if args.command == "logs":
        for event in JsonlRunLogger.for_repository(args.repo).tail(args.limit):
            print(json.dumps(event, sort_keys=True))
        return 0
    if args.command == "test":
        recovery = None
        if args.recovery_content_file or args.recovery_path or args.approve_recovery:
            if not (args.recovery_content_file and args.recovery_path and args.approve_recovery):
                print("TEST BLOCKED: recovery requires --recovery-content-file, --recovery-path, and --approve-recovery")
                logger.record("recovery_blocked", reason="missing recovery approval or target")
                return 2

            def recovery() -> bool:
                content = args.recovery_content_file.read_text(encoding="utf-8")
                diff = apply_edit(args.repo, args.recovery_path, content, approved=True)
                logger.record("recovery_edit_applied", path=args.recovery_path, changed=bool(diff))
                return True

        try:
            summary = run_test_loop(args.repo, args.test_command, timeout=args.timeout, recovery=recovery, logger=logger.record)
        except (UnsafeCommand, ValueError) as exc:
            logger.record("test_blocked", command=args.test_command, reason=str(exc))
            print(f"TEST BLOCKED: {exc}")
            return 2
        final = summary.recovery if summary.recovery_attempted else summary.initial
        print(final.stdout, end="")
        if final.stderr:
            print(final.stderr, end="")
        print(f"Test success: {'yes' if summary.success else 'no'}")
        print(f"Recovery attempted: {'yes' if summary.recovery_attempted else 'no'}")
        return final.returncode
    try:
        result = run_safe(args.command_text, args.repo, args.timeout)
    except (UnsafeCommand, ValueError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    return result.returncode
