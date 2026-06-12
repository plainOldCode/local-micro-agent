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
- targeted rewrite quality failures now defer the target and move to siblings,
  but drift recovery can still let a bad targeted rewrite collapse the sibling
  portfolio before graph reseed is tried.

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

Drift backoff policy uses the same cooldown-key namespace. Once
`spec_drift_saturation_threshold` active-task drift records repeat the same
`drift_cooldown_key`, targeted rewrite is treated as saturated: the task is
deferred, the saved rewrite call is recorded, and sibling/backtrack/reseed can
advance the frontier. Targeted rewrites for drifted tasks must also change at
least one structured material axis: `target_regions`, `tactic_stage`,
`validator.kind`, or `deliverables`. Free-form `edit_scope` text may be
advisory later, but it is not a hard-reject axis.

Reseed prompts may receive model-suggested regions derived from repeated
`declared -> attempted` drift pairs. These are advisory only and must still pass
the deterministic writable/grounding gates. The controller can also reserve
`spec_reseed_reserved_synth_calls` so targeted rewrites cannot consume the last
SPEC calls needed for graph reseed.

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
| targeted rewrite `quality_rejected` | `defer_target_then_backtrack_or_reseed` |
| `graph_rewrite_rejected` | `reject_graph_or_task_then_backtrack` |
| `correctness_failure` streak | `design_rewrite_or_tactic_cooldown` |
| `valid_no_improvement` | `preserve_survivor_then_expand_sibling` |
| `portfolio_exhausted` | `defer_portfolio_tactic` |
| no runnable tasks, sibling graph exists | `select_sibling_graph` |
| no runnable tasks, reseed budget remains | `reseed_graph` |
| no runnable tasks, no search budget remains | `terminate_search_frontier_exhausted` |

No transition should call an LLM unless it is explicitly `contract_rewrite`,
`design_rewrite`, or `reseed_graph`.

### Phase D.1: Target-Node Transactions

Targeted rewrites must be treated as transactions on one task node, not as
authorization to replace the selected graph.

This applies especially to `active_task_drift` recovery. A drifted task is
evidence that the active contract may be wrong or too broad. It is not evidence
that unrelated sibling tasks should be deleted, reset, or made dependent on the
new target. The controller should therefore apply targeted rewrite output using
these postconditions:

1. Only the target task may be replaced, deferred, or retired.
2. Runnable sibling task ids, statuses, dependencies, budgets, and observations
   are restored from the previous spec by default.
3. New non-target tasks from the model are ignored unless a workflow flag later
   enables explicit local expansion and they pass graph gates.
4. Schedulable sibling count must not decrease because of the model output.
5. For drift recovery, the replacement target must be materially different from
   the failed drift shape using structured fields only:
   `target_regions`, `tactic_stage`, `validator.kind`, and `deliverables`.
6. If the replacement fails quality, graph, or material-diversity gates, the
   controller defers only the target task and returns to `SCHEDULE`.

In other words, a targeted rewrite can improve or retire the target node, but it
cannot spend the sibling frontier. `portfolio_collapsed_below_min_runnable`
should become a target-local failure outcome, not a graph-wide frontier loss.

Current implementation warning:

- `_merge_targeted_spec_rewrite()` currently promotes `additional_tasks` to
  replacements when no explicit replacement exists, and appends additional
  tasks after merge. MVP 5 must disable both behaviors in target-transaction
  mode. A model output that omits the target but proposes unrelated new tasks is
  a target rewrite failure, not a graph expansion.
- Transaction telemetry must be computed from the raw model output before
  sibling restoration. The merged spec intentionally hides omissions by
  restoring siblings, so raw output and merged output must stay separate.

Suggested helper shape:

```python
@dataclass
class TargetedRewriteTransaction:
    merged_spec: dict[str, Any]
    raw_rewrite_spec: dict[str, Any]
    target_task_id: str
    target_outcome: str
    drift_related: bool
    preserved_sibling_task_ids: list[str]
    ignored_non_target_task_ids: list[str]
    schedulable_sibling_count_before: int
    schedulable_sibling_count_after: int
    replacement_task_ids: list[str]
    issues: list[str]
```

The transaction helper should be the single place that:

- splits raw rewrite tasks into target replacements and ignored non-target
  proposals;
- restores siblings from `previous_spec`;
- inherits design/contract rewrite attempt counters;
- computes graph/material-diversity issues from raw replacement tasks plus the
  restored sibling frontier;
- prepares telemetry for progress events, signatures, and terminal/report
  summaries.

Suggested status mapping:

| Target state | Rewrite failure | Target outcome | Sibling outcome |
| --- | --- | --- | --- |
| `needs_design` | quality or graph gate rejected | `deferred_design_invalid` | restored |
| `needs_contract_rewrite` | quality or graph gate rejected | `deferred_contract_drift` | restored |
| drift-saturated task | duplicate material axes | `deferred_contract_drift` | restored |

Suggested progress/signature fields:

- progress event: reuse existing event names for report compatibility:
  - `drift_recovery` for drift/contract targets;
  - `design_rejected` for design targets.
- action: `defer_target_preserve_siblings`
- issue_scope: `target_task`
- preserve evidence:
  - `preserved_sibling_task_ids`;
  - `schedulable_sibling_count_before`;
  - `schedulable_sibling_count_after`;
  - `ignored_non_target_task_ids`.

Using existing event names keeps `_append_spec_progress_event()` and current
reports useful without losing task detail fields. If a new event name is added
later, the report/terminal summarizers must be updated in the same patch.

Graph reseed should run only after the transaction returns to `SCHEDULE` and no
runnable sibling or backtrackable graph remains.

### Phase D.2: Metric-Neutral Plateau Control

Correctness-preserving candidates with no metric gain are useful search evidence,
but they should not be treated as a reason to keep sampling the same task,
region, and tactic indefinitely.

The `424bc36` 20-loop clean smoke showed the new target-node transaction policy
working: the run reached 11 CODE loops instead of stopping after 2, preserved
sibling frontier, and recorded target-local rewrite recovery. The next dominant
failure was a metric-neutral plateau:

- 11 candidates total;
- 9 `no_improvement` candidates;
- direct validation stayed `CYCLES 147734`;
- `task-002` was reopened until portfolio recovery was exhausted;
- graph reseed still converged near the same `build_kernel` / `build_hash`
  `local_edit` shapes.

This is not primarily a code-generation failure. It is a controller failure to
turn valid negative performance evidence into a stronger transition decision.
The controller should record metric-neutral attempts as plateau signatures and
use them in scheduling, portfolio reopen, graph scoring, and reseed prompts.

Plateau signature:

```text
target_region_hash + tactic_stage + edit_shape_fingerprint + metric_delta_bucket
```

Where:

- `target_region_hash` is the same task-id-free hash used by failure signatures;
- `tactic_stage` comes from the active task / candidate record;
- `edit_shape_fingerprint` should prefer an existing candidate fingerprint when
  present, falling back to a compact structured edit shape;
- `metric_delta_bucket` is one of `improved`, `neutral`, or `regressed`.

For this MVP, only `neutral` plateau signatures need policy effects. Improved
and regressed candidates already have existing paths.

Policy:

1. A correctness-passing, metric-neutral candidate writes a
   `metric_neutral_plateau` failure signature when the active acceptance requires
   metric improvement.
2. The signature is scoped as `candidate_delta`, but the controller also records
   the best available `spec_task_id` / `todo_id` so the plateau can be charged to
   the active task.
3. Repeating the same plateau signature on the same task closes that task as
   `deferred_no_improvement_plateau`.
4. Repeating the same region/tactic plateau across reopened variants blocks
   portfolio reopen for that task.
5. Graph scoring penalizes candidate graphs whose runnable tasks overlap recent
   plateau cooldown keys.
6. Graph reseed prompt receives plateau cooldown keys and must produce at least
   one materially different runnable task: different `target_regions`,
   different `tactic_stage`, or different `validator.kind` / deliverable shape.

Suggested thresholds:

| Counter | Default | Effect |
| --- | ---: | --- |
| same plateau fingerprint per task | 1 repeat | defer task as plateau |
| same region/tactic neutral attempts | 2 attempts | block portfolio reopen |
| graph plateau cooldown overlap | any | graph score penalty |
| reseed all tasks overlap plateau cooldowns | any | quality/graph reject |

The low thresholds are intentional. A metric-neutral candidate is already
correctness-preserving evidence. Retrying the same plateau shape is unlikely to
be informative unless the new task is materially different.

Important implementation warning:

The `424bc36` run produced later `no_improvement` candidate records where
`spec_task_id` and `todo_id` were empty even though logs showed `task-002` was
active. MVP 6 must fix attribution before relying on plateau counters. The
candidate recorder should recover task identity from, in order:

1. candidate record `spec_task_id` / `todo_id`;
2. current `active_todo`;
3. persisted `active_todo.json`;
4. in-memory `current_spec_task_id`;
5. `run_spec.active_task_id`;
6. the latest `scheduled` / `retry` spec progress event for the current loop.

If no task id can be recovered, record the plateau signature as graph-level
evidence but do not charge a task-local threshold.

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
   - recent metric-neutral plateau cooldown keys are banned unless the new graph
     is materially different in structured fields;
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

For no-improvement, the existing candidate record remains the raw observation.
MVP 6 adds only a typed plateau index and transition policy. It must not add
benchmark-specific tactic knowledge or hidden optimization hints.

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

### MVP 5: Portfolio-Preserving Targeted Recovery

Scope:

- Split targeted rewrite application into a target-node transaction helper.
- Preserve sibling tasks by controller rule after quality, graph, and material
  diversity failures.
- Convert drift-target graph rewrite rejection into
  `deferred_contract_drift`, not generic `deferred_design_invalid`.
- Do not let `portfolio_collapsed_below_min_runnable` consume the whole graph
  frontier when runnable siblings existed before the targeted rewrite.
- Add terminal/report counters for target-local rewrite rejections and restored
  siblings.

Why next:

The `9130b84` 20-loop clean run confirmed that bounded targeted quality retry
works: `task-001` was deferred after final quality failure and the scheduler
moved to siblings. The next failure was downstream: `task-002` and `task-003`
hit active-task drift, then their targeted graph rewrites were rejected for
portfolio collapse, consuming both graph reseeds after only two CODE loops.

This is an error-propagation shape. A local drift recovery attempted to repair
one active task, but the rejected rewrite still damaged the graph-level search
frontier. MVP 5 keeps the recovery local so sibling and backtrack frontiers stay
available.

Test matrix:

| Case | Scenario | Expected |
| --- | --- | --- |
| normal | drift target rewrites into a materially different replacement | replacement merged; sibling status/dependencies/budget/observation preserved |
| edge | graph contract issue such as `portfolio_collapsed_below_min_runnable` | only target deferred; runnable sibling schedules next; no reseed while sibling is runnable |
| edge | repeated drift material axes fail diversity gate | target becomes `deferred_contract_drift`; sibling frontier preserved |
| error | raw model output has no target replacement and only unrelated new tasks | unrelated tasks are ignored, not promoted to replacement |
| error | target id missing or malformed `task_graph` | no artifact/state pollution, or target-local failure with explicit telemetry |
| report | target-local rejection occurs | terminal/report include target-local rejection count, preserved sibling ids, ignored non-target ids |

Additional assertions:

- `additional_tasks` are never appended during target-transaction mode unless an
  explicit future config enables local expansion.
- raw-output omissions are recorded even though the persisted merged spec keeps
  restored siblings.
- `graph_reseed_requested` is not emitted from a target-local rejection while
  at least one sibling remains schedulable.

### MVP 6: Metric-Neutral Plateau Control

Scope:

- Add `metric_neutral_plateau` signatures for correctness-passing candidates that
  fail only because the metric did not improve.
- Fix task attribution for single-candidate `no_improvement` records so
  `spec_task_id` / `todo_id` survive through restore, retarget, and metric-gate
  paths.
- Add plateau counters by:
  - exact candidate/edit fingerprint;
  - task id;
  - target-region hash;
  - tactic stage.
- Defer a task as `deferred_no_improvement_plateau` when it repeats the same
  plateau signature.
- Block portfolio reopen when the only available reopen path repeats recent
  plateau cooldown keys.
- Add graph score plateau penalties and reseed material-difference gates.
- Report plateau counts, plateau task ids, and plateau cooldown keys in
  terminal/report artifacts.

Why next:

The `424bc36` run proved MVP 5 works and shifted the bottleneck. The run reached
11 CODE loops and preserved sibling frontier, but spent most of those loops on
correctness-preserving, metric-neutral edits:

- `no_improvement=9`;
- `task-002` reopened until `portfolio_recovery_budget_exhausted`;
- final reseed still sampled graph shapes near recent plateau cooldowns.

The next controller improvement is to convert metric-neutral attempts into
search evidence that closes redundant branches earlier and forces novelty in
the next frontier.

Test matrix:

| Case | Scenario | Expected |
| --- | --- | --- |
| normal | first correctness-passing metric-neutral candidate | survivor preserved; plateau signature written; task remains retryable |
| normal | same plateau fingerprint repeats on same task | task becomes `deferred_no_improvement_plateau`; sibling schedules next |
| edge | no-improvement candidate record lacks `spec_task_id` but active todo exists | plateau is attributed to active task |
| edge | portfolio reopen would select only a plateau-cooled task | reopen blocked; task stays deferred; backtrack/reseed considered |
| edge | reseed graph repeats plateau-cooled region/tactic | graph rejected or strongly penalized unless materially different |
| error | no task identity can be recovered | graph-level plateau signature written; no task-local defer threshold fires |
| report | plateau defer occurs | terminal/report include plateau counts, task ids, and cooldown keys |

Additional assertions:

- `deferred_no_improvement_plateau` is counted as deferred progress.
- Plateau cooldown keys exclude graph-local task ids.
- Improved candidates never write plateau signatures.
- Correctness failures do not write plateau signatures; they keep existing
  correctness/design paths.

## Non-Goals

- Do not encode benchmark-specific optimization tactics.
- Do not let the model directly decide terminal states.
- Do not remove existing `run_spec.json` compatibility.
- Do not parallelize local model calls on the M2 Max host; candidate graph
  generation should respect the existing local-model serial policy.
- Do not widen CODE's writable surface to fix search failures.
- Do not solve metric-neutral loops by increasing portfolio recovery rounds or
  graph reseed attempts.
- Do not punish all no-improvement candidates equally; only repeated or
  region/tactic-overlapping neutral plateaus should change control flow.

## Open Design Questions

Resolved from review:

- Store compact graph ledger records plus sidecar graph JSON files.
- Split design-invalid states:
  - `deferred_design_invalid`: recoverable via backtrack/reseed.
  - `failed_design`: current graph cannot use this task anymore.
- A single `valid_no_improvement` should not immediately select a sibling. Use
  the existing task-level budget first, preserve the survivor, then apply plateau
  control only when the same fingerprint or region/tactic repeats.
- Closed partial-success tasks should be preserved as survivor/grounding facts
  and re-gated against the current repo, not copied as stale closed nodes.

Remaining:

- Exact similarity threshold for `duplicate_variant`.
- Whether eager multi-graph cold start should ever be enabled by preset, or
  only by explicit workflow config.
- Exact plateau fingerprint fallback when the candidate record lacks a stable
  fingerprint but has multiple edits.

## Recommended Next Patch

Implement MVP 6.1 next:

1. Bind metric-neutral plateau evidence to the current candidate/test
   transition.
2. Ignore stale metric observations when the latest candidate is an active-task
   drift, pre-apply reject, graph/design reject, or any non-metric-neutral
   transition.
3. Require plateau records to carry a current `candidate_id`, `loop`,
   recovered task identity, metric, baseline, and `improved=false`.
4. Record ignored stale observations as controller diagnostics, not as
   `metric_neutral_plateau` signatures.
5. Add normal, edge, and error tests before any M2 Max smoke.

Do not increase graph reseed attempts or portfolio recovery rounds to hide this
failure. The goal is to spend fewer loops on known metric-neutral shapes and make
the next frontier materially different.

### MVP 6.1: Transition-Bound Plateau Evidence

The `d857fcb` 20-loop smoke showed a controller bookkeeping bug: the terminal
report contained `metric_neutral_plateau_count=1` even though the only CODE
candidate was `active_task_drift` and had no metric value. The root cause is a
credit-assignment error: `_handle_spec_task_test_result()` reads
`metric_acceptance` and `last_candidate_observation` from shared scratch and can
combine a stale `no_improvement` metric observation with the current non-metric
candidate transition.

Metric-neutral plateau evidence must therefore be transition-bound:

- `_record_single_candidate_observation()` should bind `metric_acceptance` to
  the final candidate history record after `_append_candidate_history()` applies
  any pre-apply drift/contract metadata. Binding fields:
  `candidate_transition_bound=true`, `candidate_id`, `loop`, `todo_id`,
  `spec_task_id`, `candidate_status`, and `candidate_failure_class`.
- `_record_metric_neutral_plateau_signature()` may write a plateau signature only
  when both sides of the transition agree:
  - metric observation says `failure_class=no_improvement`,
    `requires_improvement=true`, numeric `metric`, numeric `baseline`, and
    `improved=false`;
  - candidate observation is current-loop and current-candidate bound;
  - candidate failure class is metric-neutral (`no_improvement`,
    `probe_no_signal`, or `scaffold_validated`);
  - candidate status is a rejected/no-signal metric result, not an active-task
    drift, pre-apply contract reject, graph/design reject, patch miss, or
    correctness failure.
- If the transition does not bind, the controller should not write a failure
  signature. It may append a note or compact diagnostic such as
  `stale_metric_observation_ignored` for report/debugging.

Required tests:

| kind | case | expected |
|---|---|---|
| normal | current candidate and current metric observation are both no-improvement | one plateau signature is written |
| edge | metric observation is no-improvement but candidate id or loop differs | no plateau signature; ignored diagnostic recorded |
| error | active-task drift follows a stale no-improvement metric observation | no plateau signature and no plateau terminal count |
