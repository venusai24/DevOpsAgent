# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

An autonomous Root Cause Analysis (RCA) agent: given `traces.csv` / `metrics.csv` / `logs.csv` /
app-stats CSVs for an incident, it investigates via LLM tool-calling loops orchestrated as a
LangGraph `StateGraph` and produces a root-cause report with supporting evidence. The maintained
source lives entirely under `devops_agent/`; nearly everything else at the repo root (`data/`,
`scratch/`, `incident_data/`, `Corpus/`, `knowledge/`, `prompts/`, `runs/`, `calibration/`, etc.)
is gitignored local/experimental material, not shipped code — check `.gitignore` before assuming
a top-level directory is part of the real system.

## Commands

**Running an investigation** (both support HITL pause/resume — the graph interrupts, prints the
pause reason, and resumes via `Command(resume=...)` once you answer):
```bash
python cli.py --start-time "2021-03-04 10:00:00" --end-time "2021-03-04 10:30:00" \
  --app-metrics <csv> --container-metrics <csv> --logs <csv> --traces <csv>
```
`cli.py` with no args prompts interactively for missing paths. `run-incident.py` is a fixed-config
quick-start already wired to the sample dataset in `incident_data/` — just `python run-incident.py`.
Both read LLM API keys from `.env` (loaded via `python-dotenv`). Set `INVESTIGATION_ID` in the
environment to resume a specific prior investigation from checkpoint rather than starting fresh.

**Tests**
```bash
pytest tests/                          # all tests
pytest tests/unit/                     # unit only
pytest tests/test_rca_convergence.py::test_name   # single test
pytest tests/integration/ --timeout=120            # integration (some expect live infra)
```
`asyncio_mode = "auto"` is set in `pyproject.toml`, so `async def test_...` needs no decorator.
Ignore `tests/integration/run_dry_test.py` — it imports a `src.*` package tree that doesn't exist
in this repo (leftover from an unrelated project layout) and cannot run.

**Lint / typecheck** — the `Makefile` and `[tool.hatch.build.targets.wheel]` in `pyproject.toml`
still reference a `src/` package layout (`src/csv_engine`, `src/taip`, etc.) that no longer
exists; the real package is `devops_agent/`. Don't use `make lint`/`make typecheck`/`make
test-cov` as-is — run ruff/mypy directly against the real path:
```bash
ruff check devops_agent/ tests/
ruff format devops_agent/ tests/
mypy devops_agent/
```
For the same reason, `pip install -e .` / `make install` will fail at the build step. In this
environment `devops_agent` and its dependencies are already importable globally — no editable
install is needed to run or test the code.

**Infra** — `make infra` starts a Postgres+pgvector container via `docker-compose.yml`. It is
**not** required to run an investigation: the live pipeline uses DuckDB in-memory (querying the
CSVs directly) and a local SQLite file for LangGraph checkpointing. Postgres/Redis/Temporal/Qdrant
appear in `pyproject.toml` dependencies and `.env`, but nothing in the actual investigation path
(`devops_agent/orchestrator/**`, `devops_agent/tools/**`) calls them — see "Dead/unwired
subsystems" below before spending time on them.

## Architecture

**Entry → wiring → graph.** `cli.py`/`run-incident.py` get an `AppContainer`
(`devops_agent/core/container.py`), a hand-rolled DI container that builds the tool registry,
LLM factory config, and the compiled LangGraph graph (`devops_agent/orchestrator/graph.py`).
Note: the container builds its own graph with a *synchronous* `SqliteSaver` against
`checkpoints.db`, but both CLI entry points immediately rebuild the graph themselves with an
*async* `AsyncSqliteSaver` against `devops_agent_checkpoints.sqlite` before running it — the
container's own checkpointer is discarded in practice. If an investigation isn't resuming as
expected, check which sqlite file was actually used.

**Control flow is a fixed multi-stage pipeline, not a single agent loop**
(`devops_agent/orchestrator/graph.py`):
```
deduplication → context_assembly (Stage 0, no LLM — pure DuckDB/Python) → save_stage0_artifacts
  → triage (ReAct subgraph: LLM + tools, decides blast radius / T0 / affected components)
  → fan-out: one parallel RCA branch per affected component (LangGraph Send API)
  → rca (ReAct subgraph, once per branch) → merge_rca (deterministic merge of branch results)
  → deterministic_scoring (hand-coded Bayesian scorer + convergence heuristics)
  → critic (conditional; reviews only evidence flagged as arithmetically mismatched)
  → HITL interrupt nodes (AMBIGUOUS_PRE_EVIDENCE / AMBIGUOUS / INCONCLUSIVE) or report_delivery
```
Both `triage` and `rca` are self-contained sub-`StateGraph`s (`orchestrator/nodes/triage_subgraph.py`,
`orchestrator/nodes/rca_subgraph.py`) with the classic `llm_node ⇄ tools_node` shape. `rca` is
dispatched per-component via `orchestrator/nodes/rca_dispatch_node.py`, and results are recombined
by `orchestrator/nodes/merge_rca_results_node.py` before scoring
(`orchestrator/nodes/orchestration_nodes.py::deterministic_scoring_node`, backed by
`core/recovery/scoring_engine.py` + `core/recovery/rca_convergence.py`). HITL pause/resume
(`orchestrator/nodes/hitl_nodes.py`) uses LangGraph's `interrupt()`/`Command(resume=...)`.

**State** is one large `InvestigationState` TypedDict (`orchestrator/state.py`), checkpointed to
SQLite so investigations survive restarts and resume by `investigation_id`. Most fields use a
"last-writer-wins" reducer (`_keep_last`) so parallel RCA branches can all read shared fields
without conflicting; `per_branch_rca_results` and `evidence_log` use append/accumulate reducers.
`orchestrator/agents/schemas.py` defines the Pydantic structured-output schemas the LLMs must
return (`SubmitTriageReport`, `SubmitEvidenceReport`, `SubmitCriticVerdicts`).

**Tools** live in `devops_agent/tools/`: `base.py`/`models.py` define the `BaseTool` contract,
`registry.py` auto-discovers every `BaseTool` subclass under `tools/interfaces/` via
`discover_tools()`, `executor.py` runs a tool with timeout/retry/backoff, and
`langchain_adapter.py` wraps each tool into a LangChain `StructuredTool` (this wrapper also has
its own inline budget/circuit-breaker/dedup logic — but since a fresh `ToolExecutor` is
constructed at the top of every node function, that state does not persist across LLM turns, only
within one). Actual tool implementations sit in `tools/interfaces/*.py`, mostly querying the CSVs
through `core/db/duckdb_client.py` (an in-memory DuckDB singleton using `read_csv_auto` — no
ingestion step needed). Each LLM loop only binds an explicit tool-name allowlist (see the
`tool_names` list inside `rca_llm_node`/`triage_llm_node`) — roughly half the registered tool
classes in `tools/interfaces/` are never bound to either loop and are effectively dead code.
`registry.register()` silently overwrites on a duplicate tool name (logs a warning only) — note
that `tracing_tools.py` and `other_tools.py` both register a class named `query_anomalous_traces`.

**LLM routing** (`core/llm_provider.py::LLMFactory`) is role-based, two-tier: `rca` and
`context_assembly` are "HEAVY" (primary: a Meta Llama endpoint), `triage`/`critic`/`report`/
`evidence` are "LIGHT" (primary: Gemini/Gemma). Every role gets a fallback chain through backup
keys and OpenRouter models via LangChain's `.with_fallbacks(...)`. In practice `context_assembly`
never calls an LLM at all (Stage 0 is deterministic Python), so that HEAVY-tier binding is unused.

**Dead/unwired subsystems** — built out with real logic but not called anywhere on the live
investigation path (confirmed by grep, not just by inspection — check before extending or
debugging them, and check before assuming a guardrail is active):
- `core/recovery/**` (circuit breakers, retry coordinator, failure classifier, output repairer,
  graceful degradation, loop-prevention monitors) and `core/orchestrator/budget/investigation_clock.py`
  — instantiated in `container.py` but never consulted by any orchestrator node. The guardrails
  that *are* actually enforced are inline, ad hoc, and live in three different places with three
  different magic numbers: `rca_subgraph.py`/`triage_subgraph.py` (duplicate-call loop breaker),
  `rca_dispatch_node.py` (`PER_BRANCH_TOTAL_BUDGET`), and `langchain_adapter.py` (its own
  budget/circuit-breaker check, described above).
- `devops_agent/state/repositories/**` and `state/adapters/**` (Postgres/Redis-backed persistence)
  — `orchestration_nodes.py::deduplication_node` and `save_stage0_artifacts_node` are explicit
  stubs ("In a real implementation, this would query the PostgresInvestigationRepository...";
  "Mocking external write") rather than calling into this layer.
- `knowledge/` (diagnostic rules / failure signatures / remediation actions as YAML) — not
  imported by any Python code.
- `orchestrator/agents/evidence_agent.py`, `orchestrator/agents/reasoning_agent.py` — not
  referenced by any node.
- `main.py` — an early example entry point; it calls the graph's sync `.stream()` even though all
  node functions are `async def`. Use `cli.py` or `run-incident.py` instead.

## Agent tooling for the control-loop rewrite

Supports the planned rewrite of the control loop and state schema toward a single hypothesis-driven
investigator + ledger, replacing the per-component fan-out pipeline described above.

### Phase switch — read before touching any of this tooling

`.claude/mode` holds one word, `refactor` (current) or `rewrite`, and is the single source of truth for
which phase the repo is in. `.claude/hooks/_mode.py` resolves it (env `RCA_MODE` overrides for a
one-off command) and owns the two paths that must be updated when the rewrite package is named:
`NEW_PACKAGE_DIRS` and `NEW_TEST_DIR`.

It is a *file*, not an env var, on purpose. An env var's failure mode is "forgot to export in a new
shell → lock silently inert → drift goes unnoticed." A file's is "stale setting over-blocks" — loud and
safe. Prefer the mechanism whose failure is loud.

What the mode changes:

| | `refactor` (hybrid) | `rewrite` (clean slate) |
|---|---|---|
| Write boundary | **Denylist** — legacy reasoning modules read-only, substrate (`tools/`, `core/db/`) editable. Inert unless `RCA_REWRITE_LOCK=1`. | **Allowlist** — only the new package + its tests writable; everything else is reference. Active by default. |
| ruff scope | Changed files only (legacy carries ~74 pre-existing errors) | Whole new package (no inherited debt, so changed-files-only would let rot accumulate in files you stopped touching) |
| pytest gate | Legacy `tests/` | New suite only — running the legacy suite would gate new work on tests for code being deleted |
| `encoding-auditor` | Pre-existing untouched encoding is out of scope | No "pre-existing" excuse; BLOCKERs are must-fix |
| `code-reviewer` | Legacy tree off-limits unless the diff touched it | Only the new package is in scope |

An allowlist is used in rewrite mode specifically because a denylist has to enumerate every legacy
path and will eventually miss one. Flipping the mode file before the new package exists **does not**
brick the repo — the lock degrades to allow-all until `NEW_PACKAGE_DIRS` is on disk.

**Limits, so this isn't re-litigated:**

- Nothing can *prevent* a subagent being dispatched in the wrong phase — hooks cannot intercept Agent
  calls and there is no file-pattern→agent routing. So instead of blocking wrong-phase invocation, each
  subagent reads `.claude/mode` itself and adapts. Wrong-phase calls are harmless and self-announcing
  rather than impossible.
- **The write lock only covers the `Edit|Write|NotebookEdit` tools.** Writes performed through `Bash`
  (`sed -i`, `>` redirection, `tee`, `cp`, a `python3 -c` one-liner) never reach the `PreToolUse` hook
  and are not gated. This is deliberate: detecting writes by parsing arbitrary shell strings is
  unreliable, and an unreliable guard is worse than none because it manufactures false confidence.
  If the boundary needs to be airtight, the thing to reach for is harness-level `permissions.deny`
  rules in `.claude/settings.json` rather than a smarter hook — verify the deny-rule glob syntax
  empirically before relying on it, and treat the hook as defence-in-depth, not the boundary.

### Built

- **`encoding-auditor`** (`.claude/agents/encoding-auditor.md`) — read-only subagent that reviews
  new/changed code for domain logic that should be inference-time LLM reasoning instead: numeric
  thresholds defining "abnormal", keyword lists deciding semantics, hardcoded taxonomies, failure
  modes leaking into prompts, tools returning verdicts instead of data, scaffold-computed confidence.
  Applies the test *would an expert rewrite this when a new failure mode appears?* Runs in a fresh
  context so it can't inherit the main agent's rationalization for a threshold. It is scoped to
  **changed code only** and explicitly told not to audit the legacy tree — that tree is known-encoded
  and auditing it yields hundreds of true-but-useless findings.
- **`code-reviewer`** (`.claude/agents/code-reviewer.md`) — read-only Sonnet subagent for an
  adversarial second look at ordinary correctness. Assumes the code is broken and tries to prove it
  with a concrete failure scenario, labelling each finding CONFIRMED (traced) vs PLAUSIBLE (suspected).
  Calibrated against the defect classes this repo has actually produced — state that silently resets
  because its holder is reconstructed per call, silent overwrite on duplicate registration, two
  mechanisms where one is discarded, config pointing at deleted paths, built-but-unreachable modules,
  one concept with inconsistent constants, stubs that read as implemented, async/sync mismatch,
  LangGraph reducer hazards, and exceptions swallowed into false negative evidence. Deliberately
  disjoint from `encoding-auditor`, which owns encoding concerns and is told not to duplicate them.
- **`Stop` quality gate** (`.claude/hooks/lint_test_gate.py`) — blocks turn-end when changed Python
  fails ruff or the suite is red. Ruff runs on **changed files only**: the legacy tree carries ~74
  pre-existing ruff errors, so gating the whole package would block every turn permanently. pytest
  runs the full suite (~40s, currently green) only when Python outside `.claude/` changed. Has a
  `stop_hook_active` loop guard; `SKIP_QUALITY_GATE=1` bypasses it for knowingly-red refactors.
- **Legacy-tree lock** (`.claude/hooks/legacy_tree_lock.py`, `PreToolUse` on Edit|Write) — makes
  `devops_agent/orchestrator/**`, `core/recovery/**` and `core/orchestrator/**` read-only so the old
  design can be referenced but not incrementally drifted back into. `devops_agent/tools/` and
  `core/db/` are deliberately *not* locked — that substrate is meant to be adapted and kept.
  **Inert until `RCA_REWRITE_LOCK=1`**, so it blocks nothing before the rewrite starts.

**Standing rule — don't wait to be asked:** after editing rewrite code, dispatch `encoding-auditor`
on the changed files before ending the turn; after a non-trivial logic change anywhere, dispatch
`code-reviewer`. Both descriptions carry "use proactively" for the same reason. They are cheap to run
in parallel (one message, two Agent calls) and are scoped not to overlap. Skip `code-reviewer` for
trivial or mechanical edits — firing two subagents on a one-line tweak is pure cost. When the new
package gets a real name, update the path in `encoding-auditor`'s description so its trigger stays
keyed to something observable (a file path) rather than to recognising "this is the rewrite."

Note on what hooks can and cannot do here, so this isn't re-litigated: hooks are shell processes that
**gate** (block a tool call, block turn-end) — they cannot dispatch a subagent or skill. There is also
no `settings.json` mechanism mapping a file pattern to a named subagent. So automatic invocation rests
on the description text plus the standing rule above; hooks can only *catch the miss afterwards*.

### Still to build

- `replay-grader` subagent — runs an investigation against a known-ground-truth incident (seed from
  `tests/integration/inject_causal_anomalies.py`) and grades outcome (correct root cause? calibrated
  confidence?) and process (contrast cited? alternatives refuted with *discriminating* evidence?
  negative evidence backed by coverage?) separately. Needs its own context so the agent that wrote
  the code isn't grading its own work.
- `Stop`: orphan-module check — build the import graph from the entry point, fail if any module in
  the new package isn't transitively reachable. Direct antidote to the current `core/recovery/**`
  pattern (fully built, never called).
- `PostToolUse` on `tools/` edits: assert no duplicate registered tool names (the old registry
  silently overwrote `query_anomalous_traces`) and that every tool's output carries the common
  envelope (`query_id`, `result_handle`, `coverage`, `empty_because`, `provenance`, `error`).
- `Stop`: audit gate — block turn-end when rewrite code changed but `encoding-auditor` never ran on
  it. **Derive that from the harness-written transcript** (`transcript_path` arrives in the hook
  payload), never from a marker file the agent writes about its own behaviour. An agent-written
  "I audited this" record is self-attestation: the same agent that skips the audit is the one that
  writes the record, so the gate would only prove a file makes a claim. Gating on it reproduces
  exactly the pathology documented above in `langchain_adapter.py` — a guardrail whose state is
  controlled by the thing it is supposed to guard. Deferred until the new package exists; a gate
  watching a nonexistent path is dead weight. Confirm the transcript payload shape empirically
  before relying on it.

Deliberately skipped: a magic-number-blocker hook — legitimate constants exist (timeouts, result
caps), so it would be noisy enough to get disabled; that judgment belongs to `encoding-auditor`,
which can weigh context. Prefer a pytest test over a hook wherever the check is expressible as one
(the tool-envelope check really is a test; the Stop gate then covers it for free).

Don't build a legacy-research subagent — the built-in `Explore` agent already covers "how did the
old code handle X" without polluting main context.
