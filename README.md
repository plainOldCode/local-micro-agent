# Local Micro Agent

A spec-driven implementation loop for local models on M2 Max 64GB class
machines. Give it a request or a spec, and a deterministic controller drives
swappable LLMs (Ollama/MLX, llama-server, vLLM, LM Studio, or any
OpenAI-compatible endpoint) through as many CODE/TEST loops as it takes.

It is not a general-purpose interactive agent. The controller owns scheduling,
acceptance decisions, budgets, and safety gates; models only plan, decompose,
and generate patches. This keeps weak local models productive: hallucinated
test verdicts cannot override real exit codes, repeated failures are gated, and
runaway thinking is treated as a failed call.

Runtime dependencies: none outside the Python standard library.

## State Machine

Classic mode (the default flat loop):

```text
PLAN ──► READ ──► CODE ◄──► TEST ──► DONE / FAILED
                   ▲          │
                   └─ REFLECT ◄┘   retry path; BRAINSTORM runs from REFLECT
                                   when the search is stuck
```

Spec mode (`workflow.spec_mode=true` or `preset: "spec"`): a version 2
`run_spec.json` task graph becomes the loop driver.

```text
PLAN ──► READ ──► SCHEDULE ──► TASK_READ ──► ACCEPT_SYNTH ──► CODE ◄──► TEST
                   │   ▲                                       ▲          │
                   │   └─ SPEC_SYNTH (cold start, once)        └ REFLECT ◄┤
                   │   ▲                                                  │
                   │   └──────────── task closed / deferred ◄─────────────┘
                   │
                   ├──► DONE    all tasks closed
                   └──► FAILED  global loop cap / no recovery possible
```

| State | Role | Model call |
|---|---|---|
| `PLAN` | project-instructions-first action plan | planner |
| `SPEC_SYNTH` | request → v2 task graph (cold start only, once) | reasoner |
| `SCHEDULE` | pick next runnable task, recovery, termination | **none** |
| `TASK_READ` | read only the active task's scope | none |
| `ACCEPT_SYNTH` | synthesize and freeze task acceptance tests | coder |
| `READ` | select minimum files for the plan | planner |
| `CODE` | strict patch/search-replace candidates | coder |
| `TEST` | run commands; deterministic pass/fail | none/tester |
| `REFLECT` | short failure analysis before retry (conditional) | reflector |

## Quick Start

```bash
local-micro-agent --config config/config.qwen36-35b-a3b-coding-mxfp8-ollama.json \
  --repo . --request "..."
```

Copy `config.example.json` or a tuned profile from `config/`, point
`models.default` at your model server, then pick a preset:

```json
{
  "workflow": {
    "preset": "search",
    "writable_files": ["src/target.py"],
    "test_commands": ["python3 -m pytest -q"],
    "max_code_test_loops": 100
  }
}
```

Keep `workflow.writable_files` narrow enough to protect tests, fixtures, and
generated files. Smoke check: `python3 -m compileall src`.

## Workflow Presets

The workflow section has ~80 flags, and most long-run failures come from
invalid flag combinations rather than from the model. Set `workflow.preset` to
one of four vetted bundles from `src/local_micro_agent/presets.py`:

| Preset | For | Enables |
|---|---|---|
| `minimal` | conservative fix-the-tests loop | deterministic test decisions, retries, target-not-found repair; all exploration machinery off |
| `search` | long-running metric search | conditional reflect, brainstorm after repeated rejections, novelty/axis/region gates, adaptive gate controller, candidate history + artifacts, continue-after-improvement, profiling |
| `structural` | multi-step refactors | `search` + run-spec task graph, semantic analysis, structural scaffold/probe/expand lifecycle, structural checkpoints |
| `spec` | build a spec through a task graph | `structural` + strict spec-mode scheduling: task-scoped READ, synthesized acceptance, dependency gates, recovery rounds, progress/report artifacts |

Preset values are defaults, not a mode switch: any key set explicitly in
`workflow` wins. The expanded workflow records preset-supplied keys in
`preset_defaulted_keys` so the controller can tell preset defaults apart from
caller-supplied values (for example when deriving the loop budget).

Without a preset, the shipped `config/*.json` profiles run the plain
`PLAN -> READ -> (CODE -> TEST)*` path: `reflect_before_retry` is false,
`brainstorm_after_rejections` is 0, and the exploration machinery stays
dormant. Use `search`, `structural`, or `spec` when a run should actually
exercise it.

## Spec Mode

`workflow.spec_mode=true` promotes the run-local spec from advisory context to
the controller's task scheduler.

### Scheduler

`SCHEDULE` is deterministic and never calls a model. Each visit it persists
`run_spec.json` and progress, then:

1. all tasks closed → `DONE`; global CODE/TEST cap reached → `FAILED`
   (`stop_reason=max_code_test_loops`).
2. picks the next `open` task whose `depends_on` are all closed. Selection
   order: revisited tasks first, then most-depended-on, then declaration
   order. `depends_on` edges are hard gates — the controller never relaxes
   business dependencies to spend budget.
3. if nothing is runnable: restores `deferred` tasks once, then reopens failed
   prerequisite tasks that block downstream work for up to
   `spec_task_recovery_rounds` (default 2) fresh recovery rounds (per-round
   `attempts_used=0`, prior totals preserved for reporting). Metric-search
   portfolios can also relax failed dependencies
   (`spec_relax_failed_dependencies_with_budget=true`) and recycle exhausted
   failed tactics (`spec_reopen_failed_portfolio_tasks=true`) while loop budget
   remains. Only if neither applies does it exit `FAILED`
   (`stop_reason=no_recovery_possible`) with a blocked-task diagnosis such as
   `task-003 waiting on task-002`.

`max_code_test_loops` counts only CODE/TEST attempts; PLAN, READ, SPEC_SYNTH,
SCHEDULE, TASK_READ, and ACCEPT_SYNTH do not consume it. Tasks with no
deliverables and no acceptance commands are treated as context-only and close
automatically after `TASK_READ` without consuming budget.

On cold start (no v2 `run_spec.json`), SCHEDULE routes through `SPEC_SYNTH`
once: the configured spec-synthesis role emits the graph from the request,
plan, read source, and semantic facts, with a fallback model role
(`spec_synth_fallback_model_role`) after model, validation, or parse failures.
With `spec_resume=true` (default) an existing v2 graph is resumed instead, so
interrupted runs skip already-closed tasks.

For metric optimization runs, enable `spec_tactic_portfolio=true` with
`spec_force_metric_acceptance=true`. The controller then treats generated tasks
as independent measurable hypotheses, strips waterfall dependencies from metric
tasks, forces `acceptance.kind="metric"`, skips synthesized task-local tests,
and keeps trying sibling/reopened tactics until the global loop budget is
exhausted or the metric improves.

### Task schema (run_spec v2)

```jsonc
{
  "version": 2,
  "spec_id": "feature-or-optimization",
  "objective": "...",
  "invariants": ["..."],
  "task_graph": [
    {
      "task_id": "task-001",
      "title": "Implement parser",
      "depends_on": [],
      "deliverables": ["src/parser.py"],      // task-scoped writable files
      "read_hints": ["src/parser.py", "docs/spec.md"],
      "target_symbols": ["parse_item"],
      "target_regions": ["src/parser.py::parse_item"],
      "preserved_invariants": ["existing accepted inputs keep their parsed shape"],
      "edit_scope": "Change only the parse_item branch that handles tagged items.",
      "risk_level": "local",
      "tactic_stage": "local_edit",
      "risk_evidence": {
        "field": "edit_scope",
        "quote": "Change only the parse_item branch",
        "explanation": "single-region local branch edit"
      },
      "validator": {
        "kind": "command",
        "failure_condition": "configured tests fail or expected metric is missing"
      },
      "correctness_rationale": "The edit is limited to one branch and keeps fallback behavior.",
      "fallback_plan": "Revert to the previous branch and add a narrower guard.",
      "acceptance": { "kind": "synthesized", "commands": [] },
      "budget": { "attempts_max": 8, "attempts_used": 0 },
      "status": "open"                         // open|needs_design|in_progress|closed|deferred|failed|failed_design
    }
  ]
}
```

`deliverables` become the active task's writable set, intersected with the
global `workflow.writable_files` upper bound. `TASK_READ` loads `read_hints`,
the deliverables of dependency tasks, and the task's own deliverables.

When `spec_design_contract_gate=true`, implementation tasks must be executable
design contracts before CODE can run. The scheduler rejects tasks that only
name broad ideas, or that omit target symbols/regions, invariants, edit scope,
validator failure condition, correctness rationale, or fallback plan. Rejected
tasks are marked `needs_design` and routed back through `SPEC_SYNTH` with the
contract issues as rewrite focus. After repeated correctness failures, the
current task is also marked `needs_design` so the next attempt rewrites the
design instead of retrying the same tactic family. Correctness-preserving
`last_correct` survivor artifacts are summarized into SPEC context as safe
composition evidence.

Design rewrite budgets are task-scoped. If a task exhausts its design rewrite
budget before it becomes a bounded, verifiable, independently executable unit,
it is marked `failed_design` and isolated from scheduling. Other dependency-free
open tasks can continue. The run stops with `spec_design_contract_incomplete`
only when no schedulable task remains because the remaining work is design-
invalid.

Recent `design_rejected` and `failed_design` shapes are summarized back into
the next SPEC call as negative design memory. This memory records the rejected
task shape and contract issues, not benchmark-specific answers. SPEC should not
regenerate the same shape unless the replacement has a materially narrower
target region, clearer validator/failure condition, and a risk contract that
addresses the rejection.

`spec_grounding_gate=true` adds a deterministic target-grounding floor for
SPEC. After READ, the controller writes
`.local_micro_agent/spec_grounding_facts.json` with the current writable files,
read-only files, Python AST symbol/region spans, imported symbol origins,
allowed writable target regions, configured test commands, metric regex, and
baseline metric. SPEC sees a compact copy of those facts and must choose
implementation `target_regions`, `deliverables`, and structural probe
`expected_changed_regions` from the writable, resolvable target set. Read-only
or imported symbols may still appear in `read_hints`, invariants, hazards, or
correctness rationale; they are rejected only when used as changed targets or
deliverables. Grounding failures are design-contract issues such as
`unresolvable_target_region`, `non_writable_target_region`,
`imported_symbol_targeted`, `read_only_deliverable`, and
`probe_contract_region_mismatch`, so an impossible SPEC is rewritten before it
can spend CODE/TEST loops.

Candidate failure memory is also scoped before it reaches CODE or SPEC. Records
carry `failure_origin`, `issue_scope`, `repo_valid_after_restore`,
`repair_task_eligible`, and `memory_use`. Only `current_repo` issues may become
repair tasks. Failures from rejected candidate deltas, pre-apply contracts,
patch misses, or metric gates are negative lessons for retargeting or avoiding
the same shape; they are not treated as proof that the current source is broken.

At the loop cap, pending design rewrites are not sent back through `SPEC_SYNTH`.
The current run spec is preserved with `last_stop_reason=max_code_test_loops`
and a `pending_spec_rewrite_reason`. The terminal report also writes
`.local_micro_agent/terminal_state.json` with candidate/status distributions,
spec-progress distributions, and the last task snapshots so final analysis is
not confused by a late spec rewrite.

Spec-scheduled active todos can be made hard even before the first metric
improvement with `spec_hard_active_todo_contract=true`. In that mode, the
active design contract stays in the CODE prompt and candidate records, even
when generic durable todos would otherwise be soft until the first improvement.
`todo_enforce_active_change_scope=true` also rejects single-candidate edits
before apply when their target/search block is outside the active todo's named
target symbol or file region.

`spec_structural_risk_gate=true` adds a domain-neutral check for structural
changes such as rewrites, reordering, batching, scheduling, parallelization,
state lifecycle changes, data/control-flow changes, loop structure changes, and
side-effect movement. Those tasks must declare `risk_level=structural`,
start with `tactic_stage=structural_probe`, and provide `risk_evidence`,
`probe_plan`, `probe_diff_contract`, `invariant_evidence`, and
`rollback_or_shrink_plan`.
`risk_evidence` must quote an actionable task field such as `title` or
`edit_scope`; safety explanations in `correctness_rationale`, `fallback_plan`,
or invariant fields are not used as risk triggers. Active structural probes are
also constrained to a small single-region edit before apply, using
`structural_probe_max_changes` and `structural_probe_max_changed_lines`.
`probe_diff_contract_gate=true` adds an after-apply, before-test check: the
controller computes the actual snapshot diff and rejects structural probe
candidates whose changed files, hunks, line count, or Python symbol/region
touches exceed the active task's `probe_diff_contract`. Rejected probe diff
mismatches are recorded as candidate-delta lessons, not as current-repo repair
tasks.

### Acceptance

- `synthesized` (default in the `spec` preset): `ACCEPT_SYNTH` asks the coder
  lane for test files only — the model cannot emit shell commands. Files are
  syntax-preflighted, executed red-first (a brand-new task's tests must fail;
  zero-test suites are rejected), then frozen by SHA-256. The execution
  command always comes from the controller template
  `spec_acceptance_command_template` (default
  `{quoted_python} -m unittest discover -s {quoted_dir} -p 'test*.py'`, where
  `{quoted_python}` is the running interpreter). CODE can never write under
  `spec_acceptance_dir` (default `.lma_acceptance`), and frozen hashes are
  re-verified before every TEST and inside the regression gate. After
  `spec_acceptance_synth_retries` failures the task downgrades to `command`
  acceptance. `spec_force_default_acceptance_kind=true` overrides
  model-declared kinds.
- `command`: human-supplied shell commands (task-level or `test_commands`).
- `metric`: existing metric rules drive acceptance for performance tasks. In
  strict spec mode, metric tasks require a proven improvement over the current
  baseline by default (`spec_metric_requires_improvement=true`); an unchanged
  metric is treated as an inert/no-signal edit and the task stays open.

### Closing a task

A task closes only when its acceptance is green, the spec regression gate
passes (re-runs the acceptance of already-closed tasks —
`spec_regression_scope`: `all` (default) | `dependents` | `none` — plus
optional `spec_invariant_commands`), and frozen acceptance hashes still match.
A regression failure keeps the current task's changes for in-task repair; if
the task's budget is then exhausted, the task-boundary snapshot is restored
and the task is deferred. Patch-application misses (stale search blocks) do
not consume the task budget, mirroring
`todo_ignore_patch_failures_for_budget`.
Metric task failures record the candidate metric, baseline, improvement flag,
and no-improvement hint in the task observation so the next CODE attempt can
steer toward changes that actually execute on the benchmark path.

### Progress and reports

Spec mode appends every scheduling event to
`.local_micro_agent/spec_progress.jsonl` and writes
`.local_micro_agent/spec_report.md` at DONE/FAILED with graph progress,
`code_test_loop_count` vs `max_code_test_loops`, `stop_reason`, per-task
attempts and recovery rounds, deliverables, and acceptance state. This
distinguishes a true loop-cap exit from dependency blocking or no-recovery
termination.

## Search Machinery (classic mode)

These features drive long metric-search runs; the `search` preset enables the
validated combination.

**Retry feedback.** Each retry carries compact notes (`target not found`,
`no-op`, `comment-only`, failed metrics, restore events) into the next CODE
prompt. With `reflect_before_retry=true`, `reflect_conditionally=true`
(default) keeps REFLECT off the hot path: structured simple failures
(`patch_miss`, `duplicate_variant`, axis/family/contract mismatches) go
straight back to CODE, and REFLECT runs once per
`reflect_after_repeated_failure_class` (default 3) repeats of the same class.

**Novelty and cooldowns.** `candidate_novelty_gate=true` rejects byte-identical
retries of rejected candidates before tests run. `adaptive_search_memory=true`
tags candidates with domain-neutral strategy axes, tracks per-axis and
per-region statistics (`file::symbol` via AST, 50-line buckets as fallback,
combined with axis+family), and cools down axes/regions that keep failing.
`adaptive_search_reject_cooled_axes` / `adaptive_search_reject_cooled_regions`
turn cooldowns into pre-test gates (`rejected_cooled_axis`,
`rejected_cooled_region`). `adaptive_search_force_strategy_axis` makes the
controller choose a required axis per CODE loop. Domain-specific axis
vocabulary belongs in the request, not the harness.

**Adaptive gate controller.** `adaptive_gate_controller=true` makes failed
tactic-family gates evidence-aware: weakly evidenced gates run in `shadow`
mode, repeated all-skipped brainstorm pressure reopens families in `soft`
mode, and well-evidenced gates stay `hard`. Decisions are logged to
`.local_micro_agent/gate_decisions.jsonl` and summarized into CODE prompts.

**Durable todos.** An active todo keeps its tactic in the CODE prompt until
`todo_attempt_budget` is exhausted. Patch-application misses do not consume
the budget (`todo_ignore_patch_failures_for_budget=true`). Active-todo
contracts are controller-enforced (`todo_enforce_active_contract=true`):
candidates that drift from the declared axis/family are rejected pre-test;
same-todo duplicate variants are rejected
(`todo_reject_duplicate_variants=true`). Todos carry an observation chain of
recent attempts, failure classes, and recovery hints into CODE and REFLECT
(`observation_backed_todo_continuation=true`).

**Brainstorm.** Runs from REFLECT when the search is stuck. Tactics are scored
on current-run evidence (`brainstorm_score_tactics=true`), can be steered with
`brainstorm_open_novelty_lanes`, and selected tactics become active todos with
`spec_task_id` lineage when an advisory run spec is enabled
(`run_spec_after_read=true`).

**Structural tactics.** `structural_tactic_lifecycle=true` gives
scaffold/probe/expand tactics non-budgeted structural learning failures
(`scope_too_broad`, `invariant_broken`) before normal budgeting resumes.
`structural_state_checkpoint=true` retains correctness-preserving structural
patches separately from the metric-winning best state so long refactors can
continue from validated intermediate structure.

**Improvement loop.** `continue_after_improvement=true` persists
`.local_micro_agent/best_state.json` and `best.patch` on every metric
improvement and keeps searching until `max_code_test_loops`.
`validated_pattern_followup=true` first tries a narrow extension of the latest
validated pattern before exploring unrelated families.

## Model Lanes And Token Budgets

**Role lanes.** Map `models.{planner,coder,tester,reasoner,reflector,brainstorm}`
to providers with different sampling/thinking settings. Two routing layers:

- `model_role_overrides_by_call_site`: pin a call site to a role, e.g.
  `{"reflect": "reflector", "brainstorm": "brainstorm", "spec_synth": "spec_synth"}`.
- `deep_reasoning_enabled=true`: escalate selected call sites (default
  `reflect`) from their fast lane to `deep_reasoning_model_role` only when
  triggered — same failure class repeated
  (`deep_reasoning_after_same_failure_class`), no improvement for N loops
  (`deep_reasoning_after_no_improvement_loops`), or consecutive invariant
  failures (`deep_reasoning_after_invariant_failures`). This keeps reflect and
  brainstorm on cheap no-think lanes by default.

The legacy `reasoning_lane_enabled` routing for `plan`/`semantic_analysis`
call sites still applies; exact JSON/patch roles (`coder`, `brainstorm`,
`tester`) are excluded from both layers unless reconfigured.

**Reasoning-only responses are failures.** If a thinking provider returns
reasoning with empty final content, the call fails and the node takes its
normal fallback path. Keep iterative output budgets small: the tuned configs
cap JSON/patch and reflect-escalation lanes at 8K tokens. Artifact-producing
controller calls such as PLAN Markdown and run-spec JSON should use no-think
finalizer lanes, not large thinking lanes. A large `max_tokens` on a short
controller call invites a full reasoning-only burn (`num_predict` includes
thinking tokens on Ollama).

**A3B MXFP8 Ollama lane split.** The
`config/config.qwen36-35b-a3b-coding-mxfp8-ollama.json` profile keeps every
lane at `num_ctx=64000` to avoid runner reloads. Its default coder/tester lane
uses `think=false`, `max_tokens=8192`, and low sampling (`temperature=0.15`,
`top_p=0.9`, `top_k=20`, `min_p=0`) to reduce deterministic duplicate variants
without making JSON/search-replace output too loose. `reflector` and
`brainstorm` are fast no-think lanes. PLAN is routed to a no-think
`plan_final` lane (`temperature=0.7`, `top_p=0.8`, `max_tokens=12288`) so it
must emit usable Markdown instead of hidden reasoning. `spec_synth` is also a
no-think finalizer lane (`max_tokens=16384`) and requests Ollama JSON mode
(`format="json"`) before deterministic schema normalization. `reasoner` remains
the protected 8K thinking lane for semantic analysis and triggered deep-reflect
only.

For A3B model-tuning changes, run a short 10-loop smoke before long searches
and inspect `profile_events.jsonl`: reasoning-only calls should be zero,
full-`max_tokens` burns should be zero, patch misses should stay under 20%,
duplicate variants under 10%, and coder token/sec should be recorded against
the previous run. Also confirm `ollama ps` shows the expected MLX-loaded A3B
model; if it silently falls back to another backend, speed assumptions are not
valid.

**Input token budget.** For providers that declare `num_ctx`, the usable input
budget is `num_ctx - max_tokens`. Calls near the budget
(`prompt_token_budget_warn_ratio`, default 0.9) are flagged in profile records
with a once-per-loop note, and `auto_shrink_dynamic_context=true` (default)
shrinks oversized CODE dynamic suffixes to
`prompt_token_budget_target_ratio` (default 0.85) so silent `num_ctx`
truncation cannot eat the stable prefix instructions.

**OpenAI-compatible passthrough.** `think` adds common thinking-control keys
to the request; `extra_body` merges custom fields (useful for LM Studio). For
servers that ignore per-request thinking flags, set
`disable_thinking_with_assistant_prefill=true` with `think=false`. Verify with
a smoke request before trusting thinking-off for benchmarks.

**Ollama structured output.** Native Ollama providers can set top-level
`format` (for example `"json"` or a JSON schema) to constrain artifact output.
Use it for strict JSON controller nodes such as spec synthesis, while keeping
free-form Markdown nodes constrained by prompt contract and validation.

## Grounding And Context

**Project instructions first.** `PLAN` auto-detects `AGENTS.md`, `CLAUDE.md`,
`INSTRUCTIONS.md`, then `README.md` variants, and receives the workflow
constraints (`writable_files`, `test_commands`, metric settings) next to the
project context. Override with `project_instruction_files` /
`project_context_files`, or disable with `readme_first=false`.

**Semantic analysis.** `semantic_analysis_after_read=true` writes a
domain-neutral analysis artifact after READ (data visibility, read/write
hazards, API contracts, lifecycle ordering, safe hooks), filters it, and feeds
only the curated copy into CODE/BRAINSTORM prompts.

**External context packs.** `external_context_paths` injects read-only
advisory documents with source/hash/trust metadata, labeled as advisory in all
prompts; local source and tests stay authoritative. Bound with
`external_context_char_limit` / `external_context_item_char_limit`.

**Focused symbols.** `context_symbols` replaces full-file CODE context with
exact function/class excerpts for narrow Python edits. Separately,
`symbol_source_context_before_code=true` (default) scans the request, plan,
active spec task, active todo, and reflection for dotted Python symbols such as
`KernelBuilder.build`, then injects the current exact unnumbered source span
before CODE so search/target blocks can be copied from the live method body.

**Prompt-cache friendly layout.** `prompt_cache_friendly_layout=true` keeps a
stable prefix (system instruction, request, plan, source) and merges all
runtime feedback into one trailing dynamic message, so serving-layer prefix
caches (vLLM/SGLang, provider implicit caching) stay warm.

**Fresh, line-numbered source.** `current_source_context_before_code=true`
rereads writable files before each CODE attempt and appends a line-numbered
excerpt to the dynamic suffix (the prompt states that `N: ` prefixes are not
file content).

**Candidate preflight.** Before tests run, changed Python files are
`ast.parse`d and scanned for HTML-entity escaping
(`candidate_syntax_preflight`, `candidate_html_entity_preflight`); failures
become structured test results without spending a real test run.

**Patch-miss recovery.** After `Replacement target not found` or comment-only
edits, an exact context refresh
(`exact_context_refresh_after_patch_miss=true`) injects the current source of
the missed region into the next CODE attempt, and
`repair_target_not_found=true` runs a narrow same-candidate repair: the prompt
quotes the stale search text next to the best-matching current-source region
(`repair_anchor_context_lines`, default 18), and repaired candidates are
preflighted for existence and uniqueness before evaluation. Replacement edits
may also carry optional `target_region`, `start_line`, `end_line`,
`anchor_before`, and `anchor_after` hints. The controller treats line numbers
as hints, not authority: it first applies a unique exact match, then retargets
through line/anchor-bounded windows, then falls back to the legacy stripped-line
whitespace match. A supplied `target_hash` is stored as diagnostic metadata only.
Ambiguous, missing, and no-op targets are recorded as structured `patch_miss_*`
fields in candidate history. Whitespace misses with a single stripped-line match
are still retargeted automatically. Apply is all-or-repair for generated
candidates: if a multi-change candidate applies one edit but another
replacement, patch, out-of-plan, or empty change fails, the controller restores
the baseline before repair or test so partial diffs are never evaluated as
successful candidates. Patch failures preserve the actual touched and rejected
files (`patch_touched_files`, `patch_rejected_files`) plus a short
`patch_failure_detail`, so structured history points to the real failed file
even when the model's declared `path` was stale or misleading.

**Output format.** For models that struggle to JSON-escape multi-line code,
`code_output_format="xml"` switches CODE to raw `<search>`/`<replace>` blocks.
`log_raw_model_outputs=true` saves malformed outputs for diagnosis.

## Telemetry

`profile_agent=true` writes `.local_micro_agent/profile_events.jsonl`: phase
spans, model-call spans (prompt/completion tokens, tok/s, provider timings,
token-budget fields), and command spans. Streaming providers also write
`.local_micro_agent/model_streams/*.txt` with separate `.reasoning.txt`
artifacts and `reasoning_only_response` markers.

`record_candidate_artifacts=true` persists per-candidate metadata, unified
diffs, and test transcripts under `.local_micro_agent/candidate_artifacts`,
including concrete no-change reasons. `diagnostic_commands` attaches advisory
observation commands (e.g. an `EXPLAIN` summary or generated-IR stats) to each
evaluated candidate; outputs feed back into REFLECT/CODE without changing
pass/fail.

Candidate history (`candidate_history_path`) persists accepted/rejected
directions as JSONL across runs. Adaptive search rebuilds domain-neutral
episode memory from those records: strategy axes, edit regions, tactic
families, failure classes, metrics, and cooldown signals. Correct candidates
that do not improve the metric are also preserved when
`preserve_correct_survivors=true` as `.local_micro_agent/last_correct_state.json`
and `.local_micro_agent/last_correct.patch`; they remain rollback/learning
artifacts, not accepted performance wins.

## Safety And Clean Evaluation

- `TEST` trusts exit codes (`deterministic_test_decision=true` in all shipped
  configs; always enforced in spec mode) — an LLM tester cannot reject green
  tests.
- Every edit is checked against the writable set; failed candidates are rolled
  back from snapshots; `.lma_acceptance` is never writable by CODE.
- Seeded options (`plan_markdown`, `seed_files`, `seed_changes`) are for
  resume/harness experiments only.
- For clean model evaluation, do not inject prior-run winning patches or
  human-discovered ladders into prompts; advisory context must carry
  provenance (`external_context_paths`).

## Files

- `config.example.json`: provider and workflow configuration.
- `config/`: ready-to-use local model provider configs.
- `src/local_micro_agent/orchestrator.py`: FSM core, snapshot/patch
  application, command execution, CLI.
- `src/local_micro_agent/presets.py`: named workflow flag bundles.
- `src/local_micro_agent/state.py`: state enum and the single state bag.
- `src/local_micro_agent/models.py`: model-manager abstraction.
- `src/local_micro_agent/prompts.py`: micro system prompts per state.
- `src/local_micro_agent/validators.py`: JSON/XML validation and repair.
- `src/local_micro_agent/mcp_client.py`: async tool boundary.
- `src/local_micro_agent/mixins/`: stateless domain mixins composed into
  `MicroAgent` —
  `model_runtime` (model calls, JSON repair, token budgets),
  `telemetry` (profiling, streams, logging),
  `tactics` (brainstorm generation/scoring/gates),
  `search_memory` (axis/region cooldowns, contracts, failure memory),
  `todos` (todo lifecycle, run-spec graph, spec scheduler, structural
  checkpoints),
  `context` (project/external context, excerpts, slicing),
  `candidates` (history, observations, artifacts, repair).

The public entry point remains `local_micro_agent.orchestrator.MicroAgent`.
`tests/test_mixin_modules.py` guards the structure: independent imports, no
method collisions, no shadowing by the FSM core, no mutable mixin state.
