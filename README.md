# PatchPilot

PatchPilot is a safety-first coding assistant that turns a software task into
a reviewable plan, inspects a local repository, runs only approved commands,
and leaves the final change under human control.

This project is an original implementation informed by the tool-loop ideas in
[mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent). It does not copy
the upstream repository. The first milestone is deterministic; an LLM planner
and bounded edit/recovery loop will be added only after the safety boundary is
tested.

## Day 1 scope

- Python CLI for repository inspection and task planning
- Repository-root path containment checks
- Explicit allow-list for test and read-only Git commands
- Blocking for destructive commands, shell operators, and network tools
- Test execution with a timeout
- Approval-required plan model
- Unit tests for safety and inspection boundaries

## Architecture

See [docs/architecture.md](docs/architecture.md). The intended workflow is:

```text
task → inspect → propose plan → approve → edit → test → bounded recovery → review diff
```

The Day 1 CLI stops at the plan and safe command runner. It never commits,
pushes, resets, cleans, or deletes files automatically.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

Try it against a repository:

```bash
patchpilot inspect /path/to/repository
patchpilot plan "Fix the failing parser tests" --repo /path/to/repository
patchpilot run "git status --short" --repo /path/to/repository
```

Unsafe commands are rejected before a subprocess starts:

```text
$ patchpilot run "git reset --hard"
BLOCKED: git reset is blocked
```

## Roadmap

1. Add approval-gated file edits and JSONL run logs.
2. Add one bounded test-failure recovery attempt.
3. Add issue input and a small coding-task evaluation set.
4. Add an optional model provider behind the same safety policy.

## Attribution

PatchPilot is an independent project. It is informed by public ideas from
mini-SWE-agent and other SWE-agent research; no upstream source files are
included in this milestone.
