# PatchPilot

PatchPilot is a safety-first coding assistant that turns a software task into
a reviewable plan, inspects a local repository, runs only approved commands,
and leaves the final change under human control.

The core workflow is deterministic and safety-first; optional model proposals
and a LangGraph approval workflow build on the same tested execution boundary.

## Current scope

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
- Explicit `apply-proposal` command with separate recovery approval
- LangGraph inspect → propose → approve → apply → test workflow with checkpointed pause/resume
- Five-task deterministic coding benchmark with validity and test-pass metrics

## Architecture

See [docs/architecture.md](docs/architecture.md). The intended workflow is:

```text
task → inspect → propose → approve → apply → test → bounded recovery → review result
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
patchpilot propose "Fix the parser tests" src/parser.py --repo /path/to/repository --json > /tmp/proposal.json
patchpilot apply-proposal /tmp/proposal.json --repo /path/to/repository
patchpilot apply-proposal /tmp/proposal.json --repo /path/to/repository --approve
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
commands, a bounded file-name summary, and ranked excerpts from relevant safe
text files. Secret files, ignored directories, and binary files are excluded;
the planner still cannot execute tools or edit the repository. Its JSON plan is
validated before it is shown for approval. Install the optional dependency and
configure the model only when you are ready to make a live API call:

```bash
python -m pip install -e ".[llm]"
export OPENAI_API_KEY="..."
export PATCHPILOT_MODEL="gpt-4o-mini"
patchpilot llm-plan "Fix the failing parser tests" --repo /path/to/repository
```

API-key-free mock tests cover the provider and schema boundary. The key is
needed only for the `llm-plan` command.

The `propose` command is review-only: it sends only the explicitly named file
contents, validates the returned path and test command, and prints a unified
diff. Use `--json` to save a proposal for `apply-proposal`; applying it still
requires an explicit approval flag.

### Optional LangGraph workflow

Install the Python 3.9-compatible graph extra when you want the stateful
approval workflow:

```bash
python -m pip install -e ".[graph]"
patchpilot graph /tmp/proposal.json --repo /path/to/repository
patchpilot graph /tmp/proposal.json --repo /path/to/repository --approve
```

The graph pauses before any write and can resume with an approval or rejection
using a thread ID. For restart-safe local workflows, install the SQLite and API
extras:

```bash
python -m pip install -e ".[graph,persistence,api]"
patchpilot graph /tmp/proposal.json --repo /path/to/repository \
  --checkpoint-db .patchpilot/checkpoints.sqlite --thread-id task-1
patchpilot graph /tmp/proposal.json --repo /path/to/repository \
  --checkpoint-db .patchpilot/checkpoints.sqlite --thread-id task-1 \
  --resume --approve
```

PatchPilot also exposes a small local FastAPI service backed by the same SQLite
checkpointer:

```bash
uvicorn patchpilot.api:app_factory --factory --host 127.0.0.1 --port 8000
```

`POST /v1/workflows` starts a thread and returns its approval interrupt,
`GET /v1/workflows/{thread_id}` reads checkpointed state, and
`POST /v1/workflows/{thread_id}/resume` accepts `{ "approved": true }` or
`{ "approved": false }`. SQLite is intended for local or single-process use;
use a server-backed checkpointer before running multiple API workers.

The deterministic benchmark uses five small fixture repositories and does not
call a model. It measures proposal validity, test-pass rate, task success, and
latency. An opt-in live mode uses the same isolated fixtures with the configured
model and records token usage and estimated cost:

```bash
set -a; source .env; set +a
patchpilot evaluate-live tests/fixtures/coding_tasks.json \
  --model gpt-4o-mini --json
```

Live evaluation sends only the selected fixture file to the model and can incur
provider charges. Set `--input-cost-per-million` and
`--output-cost-per-million` to the current provider rates before comparing cost
reports.

### Observability

CLI and graph runs append structured JSONL events to
`.patchpilot/runs.jsonl`. Events include a `run_id` and `trace_id`, but never
capture source-file contents. The local API returns an `X-Request-ID` header and
writes request timing/status events to `api-runs.jsonl` beside its checkpoint
database, so a request can be correlated with its workflow trace.

## Roadmap

1. Add repository-aware retrieval for planner context.
2. Add retry, timeout, and model-fallback policies for live API failures.
3. Add authentication and deployment hardening for public API hosting.
