# Local Micro Agent

Ultra-light local coding-agent skeleton for M2 Max 64GB class machines.

The design goal is to avoid the prompt bloat of full agent frameworks by
running a small finite-state workflow:

1. `PLAN`: produce a compact file/action plan.
2. `READ`: load only planned source files through the tool boundary.
3. `CODE`: generate strict patch/write operations from the plan and source.
4. `TEST`: run configured commands with strict time/output limits and loop back on failure.

The implementation is intentionally small. It is a scaffold for experiments
with local models such as Qwen 3.6 35B/27B, Ollama, llama-server, vLLM, or
commercial APIs behind an OpenAI-compatible endpoint.

Runtime dependencies: none outside the Python standard library.

## How A Run Starts

The default path is project-instructions-first, then README:

1. `PLAN` reads repo-local instruction/context files before asking the model to
   plan. By default this auto-detects `AGENTS.md`, `CLAUDE.md`,
   `INSTRUCTIONS.md`, then `README.md`, `Readme.md`, `readme.md`, `README`, or
   `README.txt`.
2. `PLAN` receives workflow constraints such as `writable_files`,
   `test_commands`, and metric settings next to the project context.
3. `PLAN` turns the user request plus project context into a compact action
   plan that respects those constraints.
4. `READ` selects the minimum files needed for that plan.
5. `CODE` may only modify files allowed by `workflow.writable_files` or the
   planned file list.
6. `TEST` runs configured commands with timeout and output limits, then accepts,
   rejects, or retries. If `workflow.reflect_before_retry=true`, rejected
   retries pass through `REFLECT` before returning to `CODE`.

Each retry carries compact feedback forward. Recent agent notes such as
`target not found`, `no-op`, `comment-only`, failed metrics, and restore events
are added to the next `CODE` prompt. `REFLECT` adds a short no-code failure
analysis to the retry prompt, which helps local models avoid repeating the same
invalid or no-op candidate. Candidate history can also be persisted as JSONL
with `workflow.candidate_history_path` so accepted/rejected directions survive
across runs.

For tasks where the source has a subtle execution model, enable
`workflow.semantic_analysis_after_read=true`. After `READ`, the agent writes a
domain-neutral semantic analysis artifact to
`workflow.semantic_analysis_path`, writes the controller-filtered prompt copy to
`workflow.semantic_analysis_curated_path`, and feeds only the curated copy into
later `CODE` and `BRAINSTORM` prompts. The artifact should capture facts such as
data visibility, read/write hazards, API contracts, lifecycle ordering, current
metric constraints, and safe implementation hooks. Background benchmark notes or
other non-constraints are kept out of the curated prompt context. It is generated
from the current request and source files, not from hidden benchmark-specific
rules; existing artifacts at the same path are filtered again before resumed
runs load them into the prompt.

Use `workflow.external_context_paths` to inject read-only advisory context packs
such as Markdown notes from a human, prior high-quality analysis, repo docs, or
later fetched web references. These packs are stored separately from source file
context, carry source/hash/trust metadata, and are labeled as advisory in
`PLAN`, `READ`, `SEMANTIC_ANALYSIS`, `REFLECT`, `BRAINSTORM`, and `CODE`
prompts. Local source files, tests, and the current user request remain
authoritative. Bound the total and per-item prompt size with
`workflow.external_context_char_limit` and
`workflow.external_context_item_char_limit`.

For non-code freeform reasoning, enable `workflow.reasoning_lane_enabled=true`
and map `workflow.reasoning_lane_model_role` to a low-temperature provider with
thinking enabled, for example `models.reasoner = "qwen_reason"`. By default the
lane only routes `plan`, `semantic_analysis`, and `reflect` call sites. Exact
JSON/search-replace roles such as `coder`, `brainstorm`, and `tester` stay on
their configured providers unless you explicitly change the call-site and
excluded-role lists. This keeps planner-style reasoning separate from strict
patch generation and JSON repair paths.

Thinking providers are expected to return final content, not only hidden or
side-channel reasoning. Ollama native and OpenAI-compatible backends normalize
provider reasoning fields into model usage metadata. If a response contains
reasoning but its final content is empty or only whitespace, the controller
treats it as a failed model call by default. `PLAN`, `READ`, `CODE`, `TEST`,
`REFLECT`, and `BRAINSTORM` then use their normal failure or fallback path
instead of silently accepting an empty plan, JSON object, or retry analysis. Use
`workflow.allow_reasoning_only_response=true` only for diagnostics, or
`workflow.reasoning_only_allowed_call_sites` to allow a specific experimental
call site.

Keep iterative thinking budgets small. The tuned local configs cap exact
JSON/patch lanes at 8K tokens, reasoning lanes at 8K tokens, and exploration
lanes at 16K tokens, with non-greedy sampling for `think=true` roles. Larger
context windows are useful for source grounding, but large generation budgets on
short controller calls can turn a retry into a full reasoning-only burn with no
actionable final content.

For long exploratory runs that otherwise drift between unrelated brainstorm
ideas, enable `workflow.run_spec_after_read=true`. After `PLAN` and `READ`, the
agent asks the model to synthesize a run-local spec from the current request,
plan, read source, and semantic facts, then persists it to
`workflow.run_spec_path` (default `.local_micro_agent/run_spec.json`). The spec
contains objective/invariant facts plus a small `task_graph` of unit tasks with
`task_id`, `strategy_axis`, `family_key`, expected signal, and status.
`BRAINSTORM` is asked to derive tactics from open or repairable spec tasks, and
new active todos keep `spec_task_id` lineage when a tactic maps to the graph.
Candidate observations update the spec task state with controller hints such as
`needs_repair`, `stale_variant`, `needs_guard_or_smaller_scope`, or
`validated_no_metric_signal`, giving later loops an explicit repair/pivot/deepen
signal instead of relying only on unstructured recent history.

For exploration-heavy runs, set `workflow.candidate_novelty_gate=true`.
Rejected candidate fingerprints are then remembered inside the current run,
and an identical later candidate is rejected before tests run. The rejection is
fed back as `forbidden repeated pattern`, which makes retry loops spend budget
on new search directions instead of repeatedly testing the same failed patch.

Set `workflow.adaptive_search_memory=true` when a long run should manage its
own search budget. The agent tags each candidate with coarse strategy axes
such as `correctness`, `api_contract`, `data_flow`, `state_management`,
`error_handling`, `parsing`, `performance`, or `runtime_control`, records
per-axis success/failure statistics, and feeds a compact search-memory summary
into later `CODE` prompts. Axes that fail repeatedly in a recent window enter a
temporary cooldown, so the model is steered toward under-explored directions
without hard-coding task-specific blacklists. The same axes are written to
`workflow.candidate_history_path` records when candidate history is enabled.

If the model ignores the cooled axes, set
`workflow.adaptive_search_reject_cooled_axes=true`. That turns cooldowns into a
controller-side pre-test gate: any candidate whose extracted axes are still in
cooldown is rejected as `rejected_cooled_axis` before file edits or tests run.
For stronger control, set `workflow.adaptive_search_force_strategy_axis=true`
and optionally provide `workflow.adaptive_search_axis_pool` as a prompt
vocabulary. The controller then chooses a required axis for each `CODE` loop,
injects it into the prompt, and rejects candidates with missing, cooled, or
wrong declared `strategy_axis` values before applying edits. Set
`workflow.adaptive_search_strict_axis_pool=true` only when you intentionally
want to reject explicit axes that are not in `adaptive_search_axis_pool`.

The built-in axis set is intentionally domain-neutral. If a benchmark or
project needs specialized axes or tactic families, put that domain vocabulary in
the task request, for example in `request.txt`, and ask the agent to emit
explicit `strategy_axis` and optional `family_key` labels. The orchestrator does
not infer problem-specific tactic families from keywords. `family_key` is a
free-form label supplied by the model and is used only as an explicit
current-run grouping signal.

For v0.2-style adaptive gate control, set
`workflow.adaptive_gate_controller=true` together with
`workflow.adaptive_search_memory=true`. Failed tactic family gates then become
evidence-aware instead of permanently static:

- weakly evidenced failed families run in `shadow` mode and are allowed through
  while the gate decision is recorded;
- repeated all-skipped brainstorm pressure reopens families in `soft` mode so
  the search can recover from overblocking;
- sufficiently evidenced gates remain `hard` and still protect test budget.

Gate decisions are written to `.local_micro_agent/gate_decisions.jsonl` by
default and summarized back into later `CODE` prompts. Useful knobs include
`adaptive_gate_min_family_attempts_for_hard`,
`adaptive_gate_all_skipped_relax_streak`, `adaptive_gate_recent_limit`, and
`adaptive_gate_decisions_path`.

Structural tactics such as schedulers, parser rewrites, cache layers, lifecycle
refactors, and vectorization often need a scaffold/probe/expand lifecycle rather
than a single metric-winning patch. With
`workflow.structural_tactic_lifecycle=true`, active todos whose tactic text
looks structural are tagged with `tactic_stage` such as `structural_probe`.
Their candidate records include `stage_result`, and early correctness failures
are recorded as structural learning classes like `scope_too_broad` or
`invariant_broken` instead of immediately exhausting the todo. The controller
allows up to `workflow.structural_tactic_soft_failures` non-budgeted structural
learning failures before normal todo budgeting resumes.

Correctness-preserving structural scaffold/probe patches can also be retained
separately from metric-winning best state. With
`workflow.structural_state_checkpoint=true`, tests-passing structural candidates
that do not improve the metric are written to `workflow.structural_state_path`
and `workflow.structural_checkpoint_dir`. Later CODE prompts receive a compact
checkpoint summary plus patch excerpt, so long refactors, parser rewrites, cache
layers, migrations, and scheduler changes can continue from validated
intermediate structure without treating that checkpoint as the final best patch.

Use `workflow.brainstorm_open_novelty_lanes` to give `BRAINSTORM` a compact
menu of still-open exploration lanes. These lanes are included whenever
brainstorming runs by default, even before the all-skipped/new-family gate
fires, so clean-start searches can see coarse structural routes early. Set
`workflow.brainstorm_include_open_novelty_lanes=false` to suppress that prompt
section.

Durable todos honor `workflow.todo_attempt_budget` before moving on. A rejected
candidate, failed test, or no-change patch keeps the same active todo in the
next `CODE` prompt until the budget is exhausted, so the model can use the
error signal to repair or narrow the probe instead of treating every tactic as a
one-shot attempt. Patch-application misses such as stale search blocks are
separated from idea failures by default
(`workflow.todo_ignore_patch_failures_for_budget=true`), so a tactic is not
discarded just because generated patch/search text did not match the current
source.
Before the first metric improvement, `workflow.todo_soft_until_first_improvement=true`
keeps active todo contracts advisory so exploratory candidates are not hard
rejected for axis or family drift. That soft contract no longer disables
brainstorm throttling: by default
`workflow.pre_improvement_todo_blocks_brainstorm=true` lets an active todo with
remaining attempt budget block repeated brainstorms until the tactic gets its
budgeted tries. Set it to `false` to restore the older free-brainstorm behavior.
By default, `workflow.todo_enforce_active_contract=true` also makes active todo
contracts controller-enforced: a queued candidate whose declared
`strategy_axis` or detected family drifts away from the active todo is rejected
before edits or tests and counted against that todo's retry budget. The
controller trusts a matching declared `strategy_axis` as structured intent;
natural-language reason axis matching is used as supporting evidence, not as a
separate hard reject, so dynamic axes from the request are not overblocked by
lexical wording misses.
`workflow.todo_reject_duplicate_variants=true` also rejects same-todo retries
whose candidate/change reasons are effectively the same as a recent rejected
attempt, preventing retry budget from being spent on retesting the same
micro-variant.
With `workflow.observation_backed_todo_continuation=true` (default), active
todos also carry an observation chain into CODE and REFLECT even when
pre-first-improvement contracts are soft. The chain includes recent attempts,
structured failure classes, recovery hints, and diagnostic summaries. This keeps
a todo from becoming a small isolated loop: the next candidate is asked to
continue from the latest evidence, repair the named invariant, or move the edit
to a measurably relevant location instead of resetting to a generic tactic.

By default, `workflow.brainstorm_score_tactics=true` scores selectable
BRAINSTORM tactics instead of accepting the first valid block. The score uses
only current-run harness evidence: recent validated pattern aliases,
failed/patch-failure aliases, tactic specificity, novelty lane, hook detail, and
original order as a tie-breaker. `workflow.brainstorm_reject_axis_family_mismatch=true`
also skips tactics only when the explicit `family_key` is itself the same as a
known axis but the tactic declares a different axis. The controller no longer
uses domain-specific family-to-axis maps or keyword rules. This keeps selection
logic in the harness and benchmark/domain hints in the request.

Set `workflow.validated_pattern_followup=true` with
`workflow.continue_after_improvement=true` to create a follow-up todo from the
latest current-run improvement before exploring unrelated families. The follow-up
todo keeps the same axis/family and asks for a narrow nearby extension of the
validated local pattern. It is derived only from current-run candidate history
and artifacts, so clean evaluation does not receive prior-run answer hints.

Set `workflow.continue_after_improvement=true` for long-running search. When a
candidate improves the metric, the agent persists `.local_micro_agent/best_state.json`
and `.local_micro_agent/best.patch`, updates the in-memory best metric, and
continues to the next `CODE` loop until `max_code_test_loops` is reached.

Set `workflow.deterministic_test_decision=true` when configured shell commands
and metric rules should be the source of truth. In this mode `TEST` does not ask
a model to reinterpret the command result, so passing tests cannot be rejected by
tester hallucination and failing commands do not spend another model call. Pair
it with `workflow.retry_rejected_candidates=true` when `max_code_test_loops`
should allow repair attempts after a failing command, metric miss, or no-op CODE
attempt. The shipped local-model configs enable both flags together.

For local models that struggle to JSON-escape multi-line code snippets, set
`workflow.code_output_format="xml"`. In XML mode the CODE node emits raw
`<search>` and `<replace>` blocks inside `<candidates>` instead of putting
multi-line code inside JSON strings. Set `workflow.log_raw_model_outputs=true`
to save malformed model outputs under `.local_micro_agent/raw_model_outputs`
when parsing or repair fails.

By default, `workflow.prompt_cache_friendly_layout=true` splits CODE prompts
into a stable prefix and a dynamic suffix. The stable prefix keeps the CODE
system instruction, user request, plan, and source context at the front. Runtime
feedback such as test output, retry reflection, active todo contracts, adaptive
search memory, gate telemetry, tactic libraries, and recent candidate history is
merged into one trailing dynamic message. This does not implement prompt/KV
caching inside the agent; it only keeps prompt layout friendly to provider or
self-hosted serving-layer prefix caches such as OpenAI/Gemini implicit prompt
caching, Anthropic cache breakpoints, Gemini cached content, or vLLM/SGLang
automatic prefix caching. Avoid placing volatile timestamps, request IDs, or
tool-output snippets before stable repo context if cache hit rate matters.

By default, `workflow.current_source_context_before_code=true` rereads writable
files immediately before each CODE attempt and appends the current source excerpt
to the dynamic suffix. This keeps the stable prefix cache-friendly while giving
the model fresh target/search text after accepted candidates have changed the
repo. Use `workflow.current_source_context_char_limit` to bound that refreshed
context.

Set `workflow.record_candidate_artifacts=true` to persist candidate-level
provenance under `.local_micro_agent/candidate_artifacts`. Each candidate gets a
metadata JSON file, and candidates that apply edits also get a unified diff; test
runs get a compact stdout/stderr transcript. `rejected_no_changes` records now
store the concrete no-change reason, such as target-not-found, no-op replacement,
comment-only edits, out-of-plan paths, or patch rejection. Recent candidate
history includes those details so later CODE calls can repair the actual miss
instead of only seeing a generic rejection status.

Set `workflow.diagnostic_commands` to attach user-defined observation commands
to each evaluated candidate after tests run. Diagnostics are advisory feedback:
they do not change pass/fail or metric acceptance. Each command can be a string
or an object with `name`, `command`, `when` (`after_test`), `timeout_seconds`,
and `output_limit`. Outputs are stored in candidate history/artifacts and fed
back to REFLECT/CODE so the next attempt can see what the edit actually changed
or failed to change. For example, a performance task might record generated IR
or bundle counts; a web task might record bundle size or accessibility output;
a database task might record an `EXPLAIN` summary.

Set `workflow.repair_target_not_found=true` to turn a stale search block into a
narrow same-candidate repair pass. When a candidate has no applied edits because
`Replacement target not found` was recorded, the controller rereads the current
writable source excerpt, asks the CODE model to regenerate exactly one candidate
with a verbatim current-source search block, and then evaluates that repaired
candidate inside the same todo attempt. The repair is recorded with
`repair_parent_id` so later analysis can distinguish normal candidates from
search-block repairs.

Set `workflow.profile_agent=true` for structured controller profiling. The
agent writes `.local_micro_agent/profile_events.jsonl` by default, with phase
spans for `PLAN`/`READ`/`CODE`/`TEST`/`REFLECT`, model-call spans, and test
command spans. Each record includes `elapsed_ms`, loop/state metadata, success
or error information, and compact call metadata such as role, prompt/output
character counts, command exit code, and stdout/stderr sizes. This is intended
for comparing bottlenecks across local serving backends such as Ollama, LM
Studio, vLLM, or SGLang; it is diagnostic logging, not prompt/KV caching.
When provider usage metadata is available, model-call records also include
`prompt_tokens`, `completion_tokens`, `total_tokens`, token-per-second rates,
and provider timing fields such as Ollama prompt/eval/total durations. Providers
that do not expose token usage continue to record character counts and elapsed
wall time only.
When profiling is enabled, providers with native streaming support may also
stream model output into `.local_micro_agent/model_streams/*.txt`; model-call
profile records include `stream_path`, `stream_chunks`, and `stream_chars`.
Reasoning chunks are written separately to matching `.reasoning.txt` artifacts
and recorded with `reasoning_stream_path`, `reasoning_stream_chunks`,
`reasoning_content_chars`, and `reasoning_only_response` fields when the
provider exposes that metadata. Ollama native and OpenAI-compatible providers
support this path. Set `workflow.profile_model_stream=false` to disable
streaming artifacts, or tune `workflow.profile_model_stream_log_interval_chars`
to control the compact progress lines written to `agent.log`.

OpenAI-compatible providers also accept optional request passthrough fields:
`think` adds common thinking-control keys (`think`, `enable_thinking`, and
`enableThinking`) to the chat request, and `extra_body` is merged into the
request body after the standard fields. This is useful for local servers such
as LM Studio that expose model-specific custom fields in their chat templates.
Some servers may ignore these keys on the OpenAI-compatible endpoint; verify
with a smoke request before treating thinking as disabled for benchmark runs.
For LM Studio models where thinking is enabled globally and per-request custom
fields are ignored, set `disable_thinking_with_assistant_prefill=true` together
with `think=false` to append an empty assistant `<think>` prefill for that
provider only.

For clean model-evaluation runs, do not inject prior-run winning patches or
human-discovered transformation ladders into the prompt. Candidate ladders used
by CODE should come from the current run's own PLAN, BRAINSTORM, READ, or
future RESEARCH artifacts. Solver-oriented runs may enable explicit research or
external context gathering, but that context should carry provenance instead of
being silently mixed into clean-eval prompts.

Use `workflow.project_instruction_files` to name instruction files explicitly.
Use `workflow.project_context_files` to fully override the auto-detected context
set, or set `workflow.readme_first=false` for controlled experiments.
Use `workflow.external_context_paths` for explicit advisory context that should
be visible to prompts without being treated as local source code.

Seeded workflow options are for resume and harness experiments, not the normal
first look at a repository:

- `workflow.plan_markdown`: bypasses README-first planning with a known plan.
- `workflow.seed_files`: bypasses model file selection in `READ`.
- `workflow.seed_changes`: bypasses model code generation in `CODE`.

For general-purpose agent behavior, prefer README-first planning and keep
`workflow.writable_files` narrow enough to protect tests, fixtures, generated
files, and other out-of-scope surfaces.

```json
{
  "workflow": {
    "readme_first": true,
    "project_instruction_files": [],
    "project_context_files": [],
    "writable_files": ["src/target.py"],
    "test_commands": ["python3 -m pytest -q"]
  }
}
```

## Focused Source Context

For narrow Python edits, `workflow.context_symbols` can replace full-file
CODE context with exact function/class excerpts:

```json
{
  "workflow": {
    "seed_files": ["src/target.py"],
    "context_symbols": {
      "src/target.py": ["parse_request", "TargetService.apply"]
    }
  }
}
```

## Files

- `config.example.json`: provider and workflow configuration.
- `config/`: ready-to-use local model provider configs.
- `src/local_micro_agent/orchestrator.py`: FSM runner.
- `src/local_micro_agent/state.py`: single global state bag.
- `src/local_micro_agent/models.py`: model-manager abstraction.
- `src/local_micro_agent/mcp_client.py`: async tool boundary.
- `src/local_micro_agent/prompts.py`: micro system prompts per state.
- `src/local_micro_agent/validators.py`: JSON validation/retry helpers.

## Smoke

```bash
python3 -m compileall src
```

## Next Practical Step

Copy `config.example.json`, point `models.default` at the preferred local
model server, and set `workflow.seed_files` / `workflow.writable_files` for
narrow experiments. Existing tuned profiles live under `config/`, for example:

```bash
local-micro-agent --config config/config.qwen36-27b-mlx-ollama.json --repo . --request "..."
```
