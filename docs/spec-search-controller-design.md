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

Implement MVP 8 next:

1. Turn `SPEC_THINK_BRIEF` from free-form advisory Markdown into a typed
   hypothesis-option brief. This is prompt design plus a controller protocol,
   not prompt wording alone.
2. Add a deterministic parser/normalizer for hypothesis options:
   `hypothesis`, `change_boundary`, `causal_evidence`, `expected_signal`,
   `invariants`, `fallback`, and `why_not_smaller`.
3. Gate those options with domain-neutral structural checks before the no-think
   finalizer can use them. The controller must not invent domain-specific
   subtasks or split an idea into a local probe on its own.
4. Feed only accepted hypothesis options plus controller-owned constraints to
   `SPEC_FINALIZE`; rejected options become feedback for the next brief/rewrite.
5. Add normal, edge, and error tests before any M2 Max smoke.

Do not increase graph reseed attempts or portfolio recovery rounds to hide SPEC
artifact failures. Do not encode benchmark-specific optimization tactics. The
goal is to let the model analyze domain-specific code, while the controller
checks whether the analysis contains enough causal evidence to become an
independently testable task.

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

### MVP 7: Runner-Agnostic SPEC Thinking Brief

The `3fbfa5a` 20-loop smoke validated plateau attribution but still exhausted
the SPEC synthesis budget (`24/24`) in quality/reseed paths. The remaining
failure is not that the controller lacks prompts or failure memory; it is that
analysis, generation, and strict JSON finalization are still too tightly coupled.
When Qwen thinking is enabled, several local serving stacks can return useful
reasoning in a provider-specific field while leaving final `content` empty.
When thinking is disabled, the strict JSON finalizer is cleaner but loses the
analysis needed to choose a different exploration direction.

The controller should split SPEC synthesis into three lanes:

1. **Thinking brief lane**: analysis-only, provider-aware, non-JSON.
2. **Constraint lane**: deterministic controller extraction from brief,
   grounding, config, and recent failures.
3. **No-think finalizer lane**: strict run-spec JSON only.

#### Runner Response Abstraction

Add a small response type at the model-runtime boundary:

```python
@dataclass
class ModelTextParts:
    content: str
    reasoning: str
    usage: dict[str, Any]
    source: Literal["content", "reasoning", "mixed", "empty"]
```

Provider mapping:

- Ollama native:
  - final text: `message.content`
  - thinking text: `message.thinking`
  - request control: `think: true|false`
- OpenAI-compatible / LM Studio / reasoning proxy:
  - final text: `message.content`
  - thinking text: `message.reasoning_content`, `message.thinking`, or
    `message.reasoning`
  - request control: provider-specific `think`, `enable_thinking`,
    `enableThinking`, or `extra_body.chat_template_kwargs.enable_thinking`
- Content-only providers:
  - final text: `content`
  - thinking text: empty
  - brief lane falls back to `content`

Streaming must preserve the same distinction. Today Ollama stream chunks can
carry `{"kind": "reasoning"}`, while OpenAI-compatible reasoning chunks are
passed through the callback as plain strings. MVP 7 should make stream chunks
provider-neutral before the brief lane relies on them:

- `{"kind": "reasoning", "content": "..."}`
- `{"kind": "content", "content": "..."}`
- `{"kind": "meta", ...}` when token/finish data is available

Existing `_model_chat()` should keep returning strict final `content` for normal
call sites and should still reject reasoning-only responses by default. Add a
separate `_model_thinking_brief()` path that accepts reasoning-only output for
explicit analysis call sites and returns the best available brief text:

1. prefer non-empty final `content`;
2. otherwise use non-empty `reasoning`;
3. otherwise return an empty brief and let finalizer continue with deterministic
   facts only.

This keeps the existing JSON and CODE paths safe while allowing Qwen thinking to
be used where empty final content is not fatal.

#### SPEC_THINK_BRIEF Call Site

Add `SPEC_THINK_BRIEF_SYSTEM` as a replacement for the current advisory
`SPEC_IDEA` role when `workflow.spec_thinking_brief_enabled=true`.

Rules:

- Do not emit run-spec JSON.
- Do not use tool calls or JSON mode.
- Use deterministic context only:
  - user request and plan;
  - focused source/semantic facts;
  - grounding facts;
  - recent failure signatures/cooldown keys;
  - current rewrite/reseed focus;
  - optional externally supplied advisory evidence.
- Output compact Markdown sections:
  - Objective interpretation;
  - likely writable target regions;
  - rejected/recently failed shapes;
  - required material-difference axes;
  - smallest guarded probe candidates;
  - finalizer constraints to enforce.

Persist artifacts separately:

- `.local_micro_agent/spec_think_brief.md`
- `.local_micro_agent/spec_think_brief_meta.json`
- optional `.local_micro_agent/spec_think_raw_reasoning.md` when raw reasoning
  must be retained for debugging and `workflow.log_raw_model_outputs=true`

Meta fields:

- `provider_kind`, `provider_model`, `call_site`, `role`;
- `content_chars`, `reasoning_chars`, `selected_source`;
- `reasoning_only_response`;
- `thinking_enabled`;
- optional `preserve_thinking_enabled`;
- prompt/output token usage when available.

Raw reasoning can be large and provider-dependent. The finalizer should consume
a compact brief or extracted constraints, not an unbounded reasoning trace. If
the selected source is `reasoning`, the controller should cap the persisted
brief to `workflow.spec_thinking_brief_char_limit` and keep the raw trace only
under the existing raw-output/debug logging policy.

#### Controller-Owned Constraints

The brief is advisory. The controller must convert it into a compact
`SpecSynthesisConstraints` object before finalization:

```json
{
  "allowed_target_regions": ["perf_takehome.py::fn"],
  "banned_cooldown_keys": ["region_hash:structural_probe:issue"],
  "banned_issue_codes": ["design_contract_rollback_or_shrink_plan_missing"],
  "required_material_difference": {
    "target_region": true,
    "tactic_stage": true,
    "validator_kind": false,
    "deliverables": false
  },
  "probe_contract": {
    "max_files_changed": 1,
    "max_hunks": 2,
    "max_changed_lines": 20,
    "max_changed_functions": 1
  },
  "minimum_runnable_local_edit_tasks": 1,
  "forbidden_task_shapes": ["broad_structural_probe_without_guard"],
  "must_preserve_sibling_frontier": true
}
```

Constraint sources, in priority order:

1. deterministic workflow config;
2. writable/grounding facts;
3. current target-node transaction/reseed focus;
4. recent failure signatures and plateau/drift cooldown keys;
5. thinking brief suggestions that are consistent with 1-4.

The model may suggest constraints, but it does not own them. If a suggested
target is not writable/resolvable, the controller drops it and records why.

#### Finalizer Contract

`SPEC_FINALIZE` remains no-think strict JSON. Its prompt receives:

- deterministic source/semantic/grounding context;
- the thinking brief as advisory Markdown;
- `SpecSynthesisConstraints` as strict JSON;
- quality feedback from the previous rejected finalizer output.

The finalizer must:

- satisfy every controller constraint;
- echo only allowed target regions in tasks/probe contracts;
- avoid banned cooldown keys and issue codes;
- produce at least one runnable local-edit or guarded structural-probe task when
  the search frontier is not terminal;
- include an explicit `constraint_satisfaction` known fact or decision rule when
  it intentionally rejects the brief's first suggested target.

Quality gates remain authoritative. However, repeated finalizer rejection for
the same issue should not simply ask the same model to try again. After the
configured retry cap, the controller should either:

- apply a deterministic shrink/repair template for the failed fields; or
- defer the target locally and move to sibling/backtrack/reseed with updated
  constraints.

#### Optional External Evidence

External search should be a controller capability, not a hidden model behavior.
Use it only in the thinking brief lane and only for general grounding:

Allowed:

- official API/library/language documentation;
- error messages and known framework semantics;
- general algorithm or performance technique references.

Disallowed:

- benchmark-specific solution search;
- take-home problem title search;
- copying answer code;
- hidden test or leaderboard probing.

External evidence must be stored as compact citations/snippets in a separate
brief input section and must not widen CODE writable files.

#### Config

Suggested workflow keys:

```json
{
  "spec_thinking_brief_enabled": true,
  "spec_thinking_brief_model_role": "reasoner",
  "spec_thinking_brief_call_sites": [
    "spec_think_brief",
    "spec_think_brief_rewrite",
    "spec_think_brief_reseed"
  ],
  "spec_thinking_brief_accept_reasoning_only": true,
  "spec_thinking_brief_preserve_thinking": false,
  "spec_thinking_brief_external_search": false,
  "spec_finalizer_no_think": true
}
```

Provider-level thinking should remain declarative:

- Ollama roles can set `think=true` only for the brief model.
- OpenAI-compatible Qwen roles can set thinking through explicit provider fields
  or `extra_body.chat_template_kwargs.enable_thinking=true`.
- Finalizer/coder/tester roles should set thinking off through the runner's
  supported mechanism.

`preserve_thinking` should be treated as an experiment flag for the brief lane
only. It may improve agentic consistency, but it also consumes context and must
not leak hidden reasoning into final JSON prompts.

#### Tests

| kind | case | expected |
|---|---|---|
| normal | Ollama response has `message.thinking` and empty `message.content` for `spec_think_brief` | brief is accepted, persisted, and marked `selected_source=reasoning`; finalizer still receives no-think JSON prompt |
| normal | OpenAI-compatible response has `reasoning_content` plus final `content` | brief uses final content, records reasoning chars in meta, and preserves current `_model_chat()` behavior for JSON call sites |
| edge | content-only provider lacks reasoning fields | brief falls back to content and meta says `selected_source=content` |
| edge | thinking brief is empty but finalizer can run from deterministic grounding facts | no terminal failure; note says finalizer used facts only |
| edge | provider-specific `enable_thinking` is unsupported or ignored | brief path degrades to content-only; finalizer remains no-think |
| error | reasoning-only response occurs on `run_spec`, `code`, or JSON repair call site | existing rejection remains active |
| error | thinking brief suggests a non-writable target or benchmark-specific search result | controller drops it from constraints and records rejection reason |
| report | OpenAI-compatible streaming emits reasoning chunks | stream artifacts keep reasoning/content chunks distinguishable |
| report | run uses thinking brief | terminal/spec report include brief source, reasoning chars, content chars, and whether reasoning-only was accepted |

#### Non-Goals

- Do not expose hidden reasoning to CODE as instructions.
- Do not require every provider to support thinking.
- Do not parse final JSON out of thinking text.
- Do not solve artifact quality by increasing SPEC call budget.
- Do not make external search mandatory for ordinary local coding tasks.
- Do not allow external search to change writable files, metric commands, or task
  objectives.

### MVP 8: Hypothesis-Option Brief Protocol

The `db2381c` M2 Max smoke validated the runner plumbing: Ollama
`message.thinking` was captured, reasoning-only output became an 8KB
`spec_think_brief.md`, rewrite/reseed thinking briefs were invoked, controller
constraints were written, and no-think finalizers produced a `run_spec.json`.
The quality result did not improve. The selected graph still collapsed into a
single broad structural task, then produced active-task drift and exhausted the
graph reseed frontier.

That does not mean the controller should hard-code smaller local probes. The
thinking brief is allowed to use domain-specific symbols, performance terms,
API names, failure modes, and code structure. The controller should not
interpret those terms semantically or split them into arbitrary subtasks. Its
job is narrower: require the model's own analysis to expose a causal,
testable hypothesis before the finalizer can turn it into a task graph.

MVP 8 makes the brief a typed protocol. This includes prompt design, but the
important boundary is the controller verifier that consumes the protocol.

#### Prompt vs Protocol

Plain prompt design says:

> Analyze carefully and produce a smaller, well-grounded task.

That is not enough. The next finalizer can still ignore the request or turn a
domain-specific insight into a broad task shape.

Protocol design says the brief must emit hypothesis options with stable fields,
and the controller checks those fields without understanding domain-specific
terminology:

```json
{
  "hypothesis_id": "h1",
  "hypothesis": "Changing X should improve Y because Z.",
  "change_boundary": {
    "regions": ["relative.py::symbol"],
    "kind": "function|class|callsite|dataflow|schema|test|unknown",
    "minimality_claim": "Why this is the smallest meaningful boundary."
  },
  "causal_evidence": [
    {
      "source": "code|test|metric|failure|grounding|external",
      "reference": "relative.py::symbol or artifact id",
      "claim": "What this evidence supports."
    }
  ],
  "expected_signal": {
    "validator_kind": "command|metric|unit|manual|unknown",
    "command_or_metric": "configured command or metric name",
    "success_condition": "Observable outcome for this hypothesis."
  },
  "invariants": ["Behavior or contract that must stay true."],
  "fallback": {
    "on_failure": "What axis should be abandoned or revised.",
    "preserve": ["Evidence or sibling options to keep."]
  },
  "why_not_smaller": "Why a narrower boundary would not test the hypothesis."
}
```

The schema can be emitted as Markdown sections or JSON-like blocks, but the
controller should normalize it into a typed `SpecHypothesisOption` before the
finalizer sees it. Free-form prose that cannot be normalized remains advisory
only and must not become a task by itself.

#### Controller Contract

The controller must verify each hypothesis option using domain-neutral checks:

- `hypothesis` is non-empty and names a causal claim, not only a task label.
- `change_boundary.regions` are writable/resolvable according to grounding
  facts, or the option is marked unusable with a reason.
- `change_boundary.minimality_claim` exists. The controller does not decide
  whether the claim is semantically true, but it rejects options with no
  boundary/minimality argument.
- `causal_evidence` has at least one non-empty record tied to code, tests,
  metrics, failures, grounding facts, or allowed external evidence.
- `expected_signal` maps to configured validators or metric commands. A task
  must say how success or failure will be observed.
- `invariants` are present for implementation tasks.
- `fallback.on_failure` states what axis to abandon or revise. Repeating the
  same failed cooldown key without a new axis is rejected.
- `why_not_smaller` exists for broad or structural boundaries. This prevents
  the model from using a broad task without explaining why the boundary cannot
  be reduced.

The controller does not:

- invent `change_boundary` fields from domain terms;
- split a broad option into smaller tasks on its own;
- infer domain semantics from names such as "slot packer", "parser", or
  "schema migration";
- accept a hypothesis only because the prose sounds plausible.

Accepted options become `SpecSynthesisConstraints` input. Rejected options are
recorded as brief-contract feedback and can be shown to the next
`SPEC_THINK_BRIEF` rewrite.

#### Finalizer Input

`SPEC_FINALIZE` receives:

- the compact original thinking brief as advisory text;
- `accepted_hypothesis_options`;
- `rejected_hypothesis_options` with issue codes;
- existing deterministic `SpecSynthesisConstraints`;
- grounding facts and recent failure signatures.

The finalizer must build tasks from accepted options only. Each task should
carry an explicit link back to the hypothesis option:

```json
{
  "task_id": "task-001",
  "hypothesis_id": "h1",
  "hypothesis_claim": "...",
  "expected_signal": {...},
  "why_this_task_boundary": "..."
}
```

If no hypothesis option passes, the controller should either:

- ask for another thinking brief with the contract failures as focus, when
  SPEC budget remains; or
- stop/reseed with a diagnostic such as
  `spec_think_brief_contract_incomplete`.

It should not let the finalizer synthesize a task graph from rejected prose.

#### External Evidence

External search remains optional and controller-owned. If enabled, search
results can appear as `causal_evidence` with `source=external`, but they must
obey the same general rules from MVP 7:

- official docs and general technical references are allowed;
- benchmark-specific solution search, hidden-answer search, and copied answer
  code are disallowed;
- external evidence cannot widen writable files, objectives, validators, or
  acceptance commands.

#### Config

Suggested workflow keys:

```json
{
  "spec_hypothesis_brief_enabled": true,
  "spec_hypothesis_option_min_evidence": 1,
  "spec_hypothesis_require_expected_signal": true,
  "spec_hypothesis_require_why_not_smaller_for_structural": true,
  "spec_hypothesis_max_options": 5,
  "spec_hypothesis_rewrite_on_contract_failure": true
}
```

MVP 8 should keep the existing `spec_thinking_brief_enabled` flag. The new
flags only control how the selected brief text is normalized and gated before
finalization.

#### Implementation Seam

MVP 8 fits the current code path without changing normal `_model_chat()`
behavior:

1. `SPEC_THINK_BRIEF_SYSTEM` / `spec_think_brief_prompt()` should request
   hypothesis-option sections instead of only "smallest guarded probe
   candidates".
2. `_maybe_build_spec_thinking_brief_context()` should still collect
   `ModelTextParts`, persist `spec_think_brief.md`, and record provider meta.
   After that, it should call a new normalizer such as
   `_spec_hypothesis_options_from_brief(brief)`.
3. The normalizer should persist:
   - `.local_micro_agent/spec_hypothesis_options.json`
   - `.local_micro_agent/spec_hypothesis_option_rejections.jsonl`
4. `_spec_synthesis_constraints_context()` should include accepted options and
   rejection feedback alongside the existing controller constraints.
5. `spec_prompt()` / finalizer focus should tell the no-think JSON model to
   create tasks only from accepted options and to include `hypothesis_id` links.
6. Existing `_spec_quality_report()` and target-node transaction gates remain
   authoritative after finalization. MVP 8 adds a pre-finalizer contract; it
   does not replace post-finalizer quality gates.

#### Tests

| kind | case | expected |
|---|---|---|
| normal | thinking brief emits two valid hypothesis options with writable regions and expected signals | both options are normalized, accepted, persisted, and passed to finalizer |
| normal | finalizer emits tasks linked to accepted `hypothesis_id` values | tasks pass the hypothesis-link gate and persist in `run_spec.json` |
| edge | option uses domain-specific terminology but has evidence, boundary, expected signal, and fallback | accepted without the controller understanding the terms |
| edge | option has a broad structural boundary plus a non-empty `why_not_smaller` | allowed to proceed to existing quality/probe gates; not rewritten by controller |
| edge | brief is free-form Markdown with no parseable options | advisory text is kept, but no task graph is finalized from it |
| error | option targets non-writable or unresolved regions | rejected with a structured issue; finalizer does not receive it as accepted |
| error | option has a hypothesis but no expected signal | rejected before finalizer |
| error | option repeats a banned cooldown key without a new boundary or fallback axis | rejected as repeated failed shape |
| report | no hypothesis option passes and budget remains | next `SPEC_THINK_BRIEF` focus includes brief-contract failures |
| report | terminal occurs from incomplete brief contract | terminal/spec report includes accepted/rejected option counts and issue codes |

#### Non-Goals

- Do not make the controller a domain-specific planner.
- Do not force every hypothesis into a local-edit probe.
- Do not require all valid tasks to be small; require them to be causally
  justified, observable, and bounded.
- Do not let unstructured thinking prose bypass the existing spec quality gate.
- Do not increase model-call budgets to compensate for malformed brief options.

### MVP 9-D''' - Option-To-Task Finalizer Repair

#### Problem

MVP 9-D'' fixed missing typed hypothesis options by adding a bounded brief
repair. The next smoke showed a later failure: accepted hypothesis options
existed, but the no-think finalizer still failed to translate them into a
runnable task graph. The representative quality issues were:

- `design_contract_structural_edit_scope_too_broad_start_with_one_reversible_probe`
- `design_contract_rollback_or_shrink_plan_must_describe_a_smaller_guarded_probe`
- `hypothesis_boundary_shrink_plan_missing`

That is not a reason to weaken D' hard provenance gates. It is a narrower
finalizer repair problem: the controller has valid option ids and boundaries,
but the finalizer needs one bounded chance to repair the option-to-task mapping.

#### Design

After normal SPEC quality rewrites are exhausted, run at most one
`hypothesis_task_repair` finalizer call when all of these hold:

- `spec_hypothesis_brief_enabled` is true;
- at least one accepted hypothesis option is available;
- the latest quality report contains option-boundary or shrink/probe contract
  failures;
- SPEC synthesis budget remains.

The repair prompt receives:

- the authoritative focus, grounding facts, and synthesis constraints;
- accepted hypothesis options and rejected-option summary;
- the exact quality report;
- a compact excerpt of the failed task graph.

The repair output is still ordinary run-spec JSON and must pass the same
quality gate before persistence. If it passes, persist the repaired graph and
record `quality_repaired`. If it fails, record the rejected graph candidate and
continue the existing terminal/soft-fallback logic. D' hard hypothesis
provenance failures still block soft fallback into CODE.

#### Tests

| kind | case | expected |
|---|---|---|
| normal | accepted multi-region option is narrowed to one guarded local task after first finalizer misses shrink text | repair prompt runs once and persists the repaired spec |
| edge | quality failure is unrelated to hypothesis boundary/shrink | no repair call is attempted |
| error | repair output repeats the invalid shrink plan | no `run_spec.json` is persisted and hard fallback remains blocked |
