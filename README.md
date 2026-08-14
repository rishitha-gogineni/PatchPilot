# PatchPilot

PatchPilot is a safety-first coding assistant that turns a software task into
a reviewable plan, inspects a local repository, runs only approved commands,
and leaves the final change under human control.

The first milestone is deterministic; an LLM planner and bounded edit/recovery
loop will be added only after the safety boundary is tested.

## Current scope (Days 1–3)

- Python CLI for repository inspection and task planning
- Repository-root path containment checks
- Explicit allow-list for test and read-only Git commands
- Blocking for destructive commands, shell operators, and network tools
- Test execution with a timeout
- Approval-required plan model
- Unit tests for safety and inspection boundaries
- Approval-gated file editing with a unified diff preview
- Repository-local JSONL run logs with no source-content capture
- Test execution with timeout handling and one bounded recovery callback
- Optional structured LLM planning with OpenAI-compatible output validation
- Review-only model-generated edit proposals for explicitly selected files
- Five-task deterministic coding benchmark with validity and test-pass metrics

## Architecture

See [docs/architecture.md](docs/architecture.md). The intended workflow is:

```text
task → inspect → propose plan → approve → edit → test → bounded recovery → review diff
```

PatchPilot never commits, pushes, resets, cleans, or deletes files
automatically. A recovery callback is supplied by a future model planner; the
execution layer itself enforces the one-attempt limit.

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
patchpilot edit src/app.py --content-file /tmp/app.py.new --repo /path/to/repository
patchpilot test "python -m pytest" --repo /path/to/repository
patchpilot logs --repo /path/to/repository
patchpilot llm-plan "Fix the failing parser tests" --repo /path/to/repository
patchpilot propose "Fix the parser tests" src/parser.py --repo /path/to/repository
patchpilot evaluate tests/fixtures/coding_tasks.json
```

For a bounded recovery attempt, provide a reviewed replacement file and an
explicit recovery approval. PatchPilot applies it at most once before rerunning
the test command:

```bash
patchpilot test "python -m pytest" --repo /path/to/repository \
  --recovery-content-file /tmp/repaired.py \
  --recovery-path src/app.py --approve-recovery
```

Unsafe commands are rejected before a subprocess starts:

```text
$ patchpilot run "git reset --hard"
BLOCKED: git reset is blocked
```

An edit without `--approve` prints the diff and changes nothing. Run tests are
captured in `.patchpilot/runs.jsonl`, which is ignored by Git.

### Optional model planner

The model planner receives the task, repository markers, detected test
commands, and a bounded file-name summary. It does not receive source files,
execute tools, or edit the repository. Its JSON plan is validated before it is
shown for approval. Install the optional dependency and configure the model
only when you are ready to make a live API call:

```bash
python -m pip install -e ".[llm]"
export OPENAI_API_KEY="..."
export PATCHPILOT_MODEL="gpt-4o-mini"
patchpilot llm-plan "Fix the failing parser tests" --repo /path/to/repository
```

API-key-free mock tests cover the provider and schema boundary. The key is
needed only for the `llm-plan` command.

The `propose` command is also review-only: it sends only the explicitly named
file contents, validates the returned path and test command, prints a unified
diff, and changes nothing. Applying the diff still requires the existing
approval-gated editor.

The deterministic benchmark uses five small fixture repositories and does not
call a model. It measures proposal validity, test-pass rate, task success, and
latency. A future live mode will use the same fixtures with the configured
model and record token usage separately.

## Roadmap

1. Add issue input and a small coding-task evaluation set.
2. Connect approved model plans to a bounded edit proposal workflow.
3. Add a lightweight interactive UI after the CLI workflow is stable.
