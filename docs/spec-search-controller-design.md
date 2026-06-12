# Spec Search Controller Design

## Problem

The current spec-mode controller is still a mostly linear repair loop:

1. `SPEC_SYNTH` creates one `run_spec.json`.
2. `SCHEDULE` executes one runnable task from that graph.
3. `TEST` observes failure and routes to retry, rewrite, defer, or terminal.
4. Targeted rewrites mutate the same graph.

Recent clean runs show that this removes one loop class at a time, but then
exposes the next one:

- repeated `active_task_drift` now routes to contract rewrite/defer;
- repeated portfolio failures now cap at `deferred_portfolio_exhausted`;
- graph rewrite rejection can still collapse the search frontier into
  `deferred_contract_drift + failed_design`, with terminal
  `no_recovery_possible`.

The underlying issue is not the specific stop reason. It is that the controller
treats the spec graph as the plan, not as one candidate in a bounded search.
When that graph loses its runnable frontier, the controller has no first-class
backtrack/reseed mechanism.

## Design Principles

- Treat each spec graph as a search candidate, not as the only plan.
- Let LLMs generate and summarize candidates; let Python own transitions,
  budgets, gates, cooldowns, and terminal decisions.
- Store external feedback as typed observations, not only prose memory.
- Prefer sibling/backtrack/reseed over same-task rewrite after design-invalid
  graph rewrites.
- Terminate only when every runnable frontier is exhausted under explicit
  budgets.

These map to the research lessons already used in the investigation:

- Tree of Thoughts / LATS: keep multiple trajectories and backtrack instead of
  repairing one path indefinitely.
- Reflexion / workflow memory: preserve failure feedback, but make it reusable
  and typed.
- SWE-agent ACI: narrow the action surface and make feedback machine-readable.
- Self-correction cautions: do not trust ungrounded "reflect and fix"; require
  deterministic external gates.

## New Artifacts

### `.local_micro_agent/failure_signatures.jsonl`

One record per meaningful failed transition.

```json
{
  "schema": "failure_signature.v1",
  "created_loop": 7,
  "graph_id": "graph-0001",
  "phase": "graph_rewrite",
  "task_id": "task-002",
  "status": "graph_rewrite_rejected",
  "failure_class": "design_rewrite_invalid",
  "issue_code": "single_broad_structural_task",
  "issue_scope": "spec_graph",
  "target_regions": ["perf_takehome.py::KernelBuilder.build"],
  "tactic_stage": "structural_probe",
  "episode_fingerprint": "graph-0001:graph_rewrite:task-002:structural_probe:single_broad_structural_task:9d4a3f31",
  "cooldown_key": "9d4a3f31:structural_probe:single_broad_structural_task",
  "summary": "Targeted rewrite collapsed portfolio to one broad structural task."
}
```

### `.local_micro_agent/spec_graph_candidates.jsonl`

Append-only event ledger for generated graph candidates. Status changes are
recorded as new events rather than mutating prior records, so resume can rebuild
the current graph index deterministically.

```json
{
  "schema": "spec_graph_candidate.v1",
  "event": "candidate_created",
  "graph_id": "graph-0003",
  "parent_graph_id": "graph-0001",
  "origin": "reseed_after_graph_rewrite_rejected",
  "status": "backtrackable",
  "created_loop": 7,
  "selected_loop": null,
  "rejected_loop": null,
  "score": {
    "runnable_tasks": 2,
    "quality_issues": 0,
    "design_issues": 0,
    "cooldown_hits": 0,
    "duplicate_hits": 0
  },
  "graph_signature": [
    "perf_takehome.py::KernelBuilder.build:structural_probe"
  ],
  "cooldown_keys": [
    "9d4a3f31:structural_probe:single_broad_structural_task"
  ],
  "spec_sidecar_path": ".local_micro_agent/spec_graph_candidates/graph-0003.json"
}
```

The currently selected graph is still persisted to `run_spec.json` for
compatibility. The candidate ledger is the search memory around it.
Full graph JSON should live in sidecar files under
`.local_micro_agent/spec_graph_candidates/<graph_id>.json`; the jsonl ledger
should stay small enough to scan frequently.

### `run_spec.json` Additions

```json
{
  "search": {
    "graph_id": "graph-0003",
    "parent_graph_id": "graph-0001",
    "generation": 1,
    "reseed_attempts": 1,
    "reseed_attempts_max": 2,
    "cooldown_keys": [],
    "spec_synth_calls_used_at_selection": 9
  }
}
```

## Failure Signature Rules

Initial MVP signatures:

| Trigger | `failure_class` | `issue_code` | Next policy |
| --- | --- | --- | --- |
| active task scope/file/shape drift streak | `active_task_drift` | candidate drift status | contract rewrite once, then defer |
| targeted graph rewrite rejected | `design_rewrite_invalid` | graph gate issue | store signature, backtrack/reseed |
| correctness failure streak | `correctness_failure` | repeated correctness class | design rewrite or tactic cooldown |
| metric OK but no improvement | `valid_no_improvement` | no metric gain | preserve survivor, sibling expansion |
| portfolio revisit cap reached | `portfolio_exhausted` | recovery budget exhausted | defer tactic |
| patch target/apply miss | `patch_miss` | patch miss reason | one retarget repair, then lesson |

Two keys are required:

```text
episode_fingerprint = {graph_id}:{phase}:{task_id}:{tactic_stage}:{issue_code}:{target_region_hash}
cooldown_key = {target_region_hash}:{tactic_stage}:{issue_code}
```

`episode_fingerprint` is graph-local and can include `task_id` for deduping
within one execution path. `cooldown_key` intentionally excludes `task_id`
because task ids are graph-local and cannot reliably match across sibling or
reseeded graphs. `target_region_hash` should be stable but compact. It should
use the declared target region string first, not the raw patch body.

For active-task drift, signatures and candidate records also carry drift
target telemetry:

- `drift_declared_regions` / `drift_declared_symbols`: the active contract;
- `drift_attempted_regions`: structured regions from candidate metadata or
  post-apply diff summaries;
- `drift_region_pairs`: compact `declared -> attempted` pairs for later
  retarget/reseed analysis;
- `drift_cooldown_key`: the same cooldown-key namespace used by signatures.

This turns repeated drift into a search signal without making policy decisions
from prose. Backoff and reseed policy should consume these structured fields,
not natural-language edit-scope similarity.

## Controller Loop

### Phase A: Spec Search

Cold start defaults to lazy search:

1. Generate one candidate graph and select it if gates pass.
2. Store rejected candidates and the selected candidate in the graph ledger.
3. Defer sibling generation until the selected graph loses its runnable
   frontier.

Lazy generation avoids paying `N - 1` local model calls on runs where the first
graph succeeds. This matters because local model calls are serial on the M2 Max
host. Eager generation remains an optional later mode via
`spec_graph_candidate_count`, but it should not be the default.

Every SPEC_IDEA, finalizer, fallback finalizer, sibling, and reseed call must
consume the existing `_consume_spec_synth_call_budget` path. Prefer sharing one
idea pass and running bounded finalizer variants when multiple candidates are
needed.

When multiple candidates exist, run existing grounding, quality, design, and
graph gates on each. Score deterministic properties only:

   - runnable task count;
   - quality/design issue count;
   - cooldown hits;
   - duplicate hits;
   - task count within configured range;
   - first task is executable and bounded.

Selection order:

1. valid graph with zero quality/design issues;
2. no cooldown hits;
3. first task is bounded local or a materially narrow structural probe;
4. task count within configured range;
5. fewer duplicate hits;
6. generation order as the final tie-breaker.

Do not maximize runnable task count blindly. A graph with many shallow tasks
should not outrank a smaller graph whose first task is more executable.

For MVP, candidate generation can be sequential calls to the existing
SPEC_IDEA/SPEC_FINALIZE path. Later, one model call may emit several graph
ideas, but the controller should still store them as separate candidates.

Candidate diversity check:

- Compute a graph signature from each task's `(target_region, tactic_stage)`.
- If a new candidate is too similar to an existing candidate, record it as
  `duplicate_variant` and exclude it from sibling selection.
- When generating candidate `i + 1`, include compact signatures from prior
  candidates as "already proposed" context.

### Phase B: Execute One Task

Keep the current task execution path:

`SCHEDULE -> TASK_READ -> ACCEPT_SYNTH -> CODE -> TEST`.

The difference is that the active graph has a `search.graph_id`, and each task
observation can create a typed failure signature.

### Phase C: Observe

After TEST or graph gate rejection, normalize feedback into:

- candidate history;
- spec progress event;
- task `last_observation`;
- optional `failure_signature`.

The signature is the input to transition policy and future cooldowns.

### Phase D: Transition Policy

The controller must choose from a closed set of actions:

| Observation | Action |
| --- | --- |
| `active_task_drift` under rewrite budget | `contract_rewrite` |
| `active_task_drift` after rewrite budget | `defer_contract_drift` |
| `graph_rewrite_rejected` | `reject_graph_or_task_then_backtrack` |
| `correctness_failure` streak | `design_rewrite_or_tactic_cooldown` |
| `valid_no_improvement` | `preserve_survivor_then_expand_sibling` |
| `portfolio_exhausted` | `defer_portfolio_tactic` |
| no runnable tasks, sibling graph exists | `select_sibling_graph` |
| no runnable tasks, reseed budget remains | `reseed_graph` |
| no runnable tasks, no search budget remains | `terminate_search_frontier_exhausted` |

No transition should call an LLM unless it is explicitly `contract_rewrite`,
`design_rewrite`, or `reseed_graph`.

### Phase E: Backtrack / Reseed

When the selected graph has no runnable frontier:

1. Try a `backtrackable` sibling from `spec_graph_candidates.jsonl`.
2. Before selecting a sibling, re-extract current grounding facts from the
   mutated repo and rerun grounding/design/quality gates.
3. If a sibling no longer gates cleanly, append `candidate_rejected` with
   `status=rejected_stale`, then try the next sibling.
4. If no valid sibling exists, run graph reseed if
   `spec_graph_reseed_attempts` remains.
5. Reseed prompt receives current failure signatures as cooldowns:
   - same `target_region + tactic_stage + issue_code` is banned;
   - the new graph must include at least one runnable local probe or a
     materially narrower structural probe;
   - the reseed must preserve useful closed/survivor evidence as facts, not as
     stale graph nodes.
6. If reseed candidate fails gates, record it as `rejected`.
7. If no candidate survives, terminate diagnostically.

Closed tasks from a prior graph should not be copied into a new graph as
already-closed nodes. They should be summarized into survivor/grounding facts
and the new graph should be gated against the current repo state.

### Phase F: Terminal

Replace generic no-runnable terminal with frontier-aware reasons:

- `search_frontier_exhausted_after_drift_deferred`
- `search_frontier_exhausted_after_design_invalid`
- `search_frontier_exhausted_after_portfolio_exhausted`
- `search_frontier_exhausted_after_graph_reseed_exhausted`
- `partial_success_search_frontier_exhausted`

Terminal artifacts should include:

- `selected_graph_id`;
- `graph_candidate_counts` by status;
- `failure_signature_counts` by `failure_class` and `issue_code`;
- `graph_reseed_attempts`;
- `cooldown_keys`;
- `spec_synth_calls_used`;
- `spec_synth_call_budget`;
- existing drift/portfolio counters.

## Relationship to Existing Memory

The new signature layer is an index over existing observations, not a parallel
source of truth.

| Existing artifact | Current role | Signature relationship |
| --- | --- | --- |
| `candidates.jsonl` | raw CODE/TEST candidate episodes | source for candidate failure signatures |
| `spec_progress.jsonl` | scheduler and SPEC transition events | source for graph/design/recovery signatures |
| adaptive search memory | tactic/family/region cooldown hints | consumes cooldown keys after MVP 3 |
| gate decision records | detailed validator outcomes | provide `issue_code` and `issue_scope` |
| `failure_signatures.jsonl` | normalized transition dataset | compact typed index for policy/backtrack/reseed |

Patch miss, correctness failure, no-improvement, and active-drift handling
should not be reimplemented twice. The signature should describe the existing
decision and make it reusable by later graph search policy.

## MVP Implementation Plan

### MVP 1: Typed Failure Signatures

Scope:

- Add signature writer/loader helpers.
- Emit signatures for:
  - `active_task_drift` defer/rewrite;
  - `graph_rewrite_rejected`;
  - portfolio exhaustion.
- Add terminal/report summaries, including `spec_synth_calls_used` and
  `spec_synth_call_budget`.

Why first:

This makes the next patches testable without changing scheduling behavior.

Tests:

- graph rewrite rejection writes `design_rewrite_invalid` signature;
- drift defer writes `active_task_drift` signature with stable fingerprint;
- cooldown key excludes graph-local `task_id`;
- terminal_state includes signature counts.

### MVP 2: Graph Candidate Ledger

Scope:

- Add graph id/search metadata to persisted specs.
- On cold start, store selected graph as `selected`.
- On failed quality/design graph candidates, store them as `rejected`.
- Store graph specs in sidecar files and graph ledger events in jsonl.
- Keep `run_spec.json` compatibility.

Why second:

This establishes the search abstraction without requiring multi-candidate
generation yet.

Tests:

- cold start selected spec appends `spec_graph_candidate`.
- quality-rejected spec appends rejected graph candidate with issue codes.
- resumed spec keeps the existing `search.graph_id`.
- candidate status changes append new ledger events.

### MVP 3: Backtrack / Reseed After Graph Rewrite Rejection

Scope:

- Change `graph_rewrite_rejected` from terminal-prone `failed_design` only to:
  1. emit failure signature;
  2. mark the rejected target `deferred_design_invalid` or
     `failed_design_backtrack`;
  3. try sibling candidate;
  4. else reseed up to `spec_graph_reseed_attempts=2`.
- Add cooldown context to SPEC_IDEA/SPEC_FINALIZE prompts.
- Re-gate any sibling candidate against current grounding facts before
  selection.

Tests:

- `deferred_contract_drift + failed_design` no longer stops as
  `no_recovery_possible` when reseed budget remains.
- graph rewrite rejection schedules a valid sibling graph when available.
- stale sibling is marked `rejected_stale` and skipped.
- reseed prompt contains rejected fingerprint/cooldown.
- reseed exhaustion stops with
  `search_frontier_exhausted_after_graph_reseed_exhausted`.

### MVP 4: Multi-Graph Search Expansion

Scope:

- Add optional `spec_graph_candidate_count`.
- Generate extra candidates lazily on frontier collapse by default.
- Allow eager 2-4 graph generation only when explicitly configured.
- Score and select deterministically.
- Keep unsuccessful valid graphs as `backtrackable`.

Tests:

- best valid graph selected over quality-invalid graph.
- sibling graph selected after selected graph exhausts.
- candidate budget is bounded by `spec_synth_call_budget`.
- duplicate graph variants are excluded from sibling selection.

## Non-Goals

- Do not encode benchmark-specific optimization tactics.
- Do not let the model directly decide terminal states.
- Do not remove existing `run_spec.json` compatibility.
- Do not parallelize local model calls on the M2 Max host; candidate graph
  generation should respect the existing local-model serial policy.
- Do not widen CODE's writable surface to fix search failures.

## Open Design Questions

Resolved from review:

- Store compact graph ledger records plus sidecar graph JSON files.
- Split design-invalid states:
  - `deferred_design_invalid`: recoverable via backtrack/reseed.
  - `failed_design`: current graph cannot use this task anymore.
- `valid_no_improvement` should not immediately select a sibling. Use the
  existing task-level portfolio budget first, preserve the survivor, then
  expand sibling/reseed after exhaustion.
- Closed partial-success tasks should be preserved as survivor/grounding facts
  and re-gated against the current repo, not copied as stale closed nodes.

Remaining:

- Exact similarity threshold for `duplicate_variant`.
- Whether eager multi-graph cold start should ever be enabled by preset, or
  only by explicit workflow config.

## Recommended Next Patch

Start with MVP 1 and the minimal terminal reason fix:

1. Add failure signature artifact helpers.
2. Emit `design_rewrite_invalid` for `graph_rewrite_rejected`.
3. Emit signature summaries into `terminal_state.json`.
4. Add a diagnostic stop reason for the current observed mixed case:
   `search_frontier_exhausted_after_design_invalid`.
5. Include `spec_synth_calls_used` / `spec_synth_call_budget` in terminal
   output while touching terminal summaries.

This does not solve search yet, but it turns the latest failure into a typed
transition dataset. MVP 2 and MVP 3 can then use that dataset to backtrack and
reseed.
