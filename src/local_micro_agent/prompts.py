from __future__ import annotations

import json

from .state import AgentState

DEFAULT_CHAR_LIMIT = 15000


PLAN_SYSTEM = """You are the PLAN node in a local coding-agent FSM.
Output only concise Markdown with:
1. Files to read or modify
2. Ordered implementation steps
3. Test commands
Do not write code. Do not include unrelated architecture discussion.
Respect project instructions and workflow constraints before giving generic
advice. If writable files are constrained, do not plan modifications outside
that set. Do not modify tests unless the user explicitly asks for test changes.
Prefer reading source entrypoints named by the README or task text before
choosing implementation changes.
Always emit final Markdown content. Make the plan concrete enough for a
downstream spec node: name known target files, invariant constraints, measurable
signals, and the next read needed when evidence is missing."""

READ_SYSTEM = """You are the READ node in a local coding-agent FSM.
Select the minimum source files needed for the plan.
Output strict JSON:
{"files":["relative/path.py"],"reason":"short reason"}
Do not include markdown or prose outside JSON."""

SEMANTIC_ANALYSIS_SYSTEM = """You are the SEMANTIC_ANALYSIS node in a local coding-agent FSM.
Do not write code. Extract durable facts the next CODE attempts must obey.
Output concise Markdown with these sections:
- Code-usable facts
- Hazards and ordering constraints
- Current task metric constraints
- Safe implementation hooks
- Background / non-constraints
Keep it domain-neutral and grounded only in the supplied request, plan, source,
and clearly labeled external advisory context.
Prefer concrete read/write, ordering, lifecycle, API, or metric facts over generic advice."""

SPEC_SYSTEM = """You are the SPEC node in a local coding-agent FSM.
Do not write code. Convert the request, plan, read source, and semantic facts into
one run-local execution spec that the deterministic spec scheduler can execute.
For optimization or metric-search requests, treat tasks as an agentic tactic
portfolio, not a waterfall implementation plan.
Every implementation task must be a bounded, verifiable, independently
executable unit: one intent, one primary target region, one edit boundary, one
validator, and one rollback/shrink path. A broad idea is not a task until it is
decomposed into such a unit.
Output strict JSON with:
{
  "version": 2,
  "spec_id": "short-lowercase-id",
  "objective": "one sentence",
  "invariants": ["must preserve..."],
  "known_facts": ["grounded fact from current read/source only"],
  "task_graph": [
    {
      "task_id": "task-001",
      "hypothesis_id": "accepted-hypothesis-id",
      "title": "short task",
      "strategy_axis": "axis_or_general_edit",
      "family_key": "lowercase_snake_case_or_empty",
      "expected_signal": "observable test/metric/diagnostic signal",
      "target_symbols": ["symbol_or_function_name"],
      "target_regions": ["relative/path.py::symbol_or_local_region"],
      "preserved_invariants": ["specific behavior, API, ordering, or data invariant"],
      "edit_scope": "one-sentence maximum edit boundary",
      "risk_level": "local|structural",
      "tactic_stage": "local_edit|structural_probe|structural_expand",
      "risk_evidence": {
        "field": "title|edit_scope|strategy_axis|family_key|expected_signal",
        "quote": "exact source phrase from that field supporting the risk label",
        "explanation": "why this task is local or structural"
      },
      "probe_plan": "first smallest reversible probe when risk_level is structural",
      "probe_diff_contract": {
        "allowed_files": ["relative/path.py"],
        "allowed_regions": ["relative/path.py::symbol_or_local_region"],
        "expected_changed_regions": ["relative/path.py::symbol_or_local_region"],
        "target_symbols": ["symbol_or_function_name"],
        "max_files_changed": 1,
        "max_hunks": 2,
        "max_changed_lines": 40,
        "max_changed_functions": 1,
        "forbidden_symbols": ["symbol_or_function_name"],
        "forbidden_regions": ["relative/path.py::symbol_or_region"],
        "required_unchanged_regions": ["relative/path.py::symbol_or_region"],
        "allowed_change_kinds": ["add_guard|add_observation|single_callsite_probe|local_refactor"],
        "allowed_edit_shape": "add_guard|add_observation|instrument_local_decision|single_branch_reorder|single_callsite_probe|one_hunk_local_refactor",
        "forbidden_edit_shapes": ["whole_function_refactor|full_rewrite|cross_region_redesign|multi_function_coordination"],
        "must_include_guard": true,
        "observation": "what the diff-level probe is allowed to observe"
      },
      "invariant_evidence": ["observable invariant or diagnostic proving the probe is safe"],
      "validator": {
        "kind": "metric|command|synthesized",
        "failure_condition": "observable condition that rejects the attempt"
      },
      "correctness_rationale": "why this task can preserve behavior",
      "fallback_plan": "how to shrink, repair, or abandon this task after failure",
      "rollback_or_shrink_plan": "how to revert or reduce the probe after failure",
      "status": "open",
      "depends_on": [],
      "deliverables": ["relative/path.py"],
      "read_hints": ["relative/path.py"],
      "acceptance": {
        "kind": "synthesized",
        "commands": []
      },
      "budget": {
        "attempts_max": 3,
        "attempts_used": 0
      }
    }
  ],
  "decision_rules": ["when patch_miss then repair with fresh source", "..."]
}
Rules:
- Ground every task in the supplied request, plan, source, and semantic facts.
- Prefer small measurable unit tasks over broad ideas.
- Always set version to 2.
- Output one JSON object only. Do not include markdown fences, comments, prose,
  or private reasoning tags.
- For performance/metric search tasks, use depends_on: [] unless a task truly
  consumes a concrete artifact produced by another task. Do not create a linear
  chain just because tactics are listed in an order.
- Make each implementation task one independent optimization hypothesis that
  can fail without blocking sibling hypotheses.
- Make each implementation task an executable design contract, not a broad
  idea label. It must name the target symbol/region, the invariants CODE must
  preserve, a small edit scope, a validator/failure condition, and why the
  task should preserve correctness.
- If an idea cannot be expressed as one target region, one intent, one edit
  boundary, and one validator, split it into smaller tasks or emit a context
  task/read hint instead of sending it to CODE.
- Set deliverables to the smallest writable file paths or globs the task may change.
- Set read_hints to the source paths the task needs before CODE.
- Set expected_signal to a concrete command, metric, diagnostic, or source-level
  observation the controller can use as feedback.
- Do not emit abstract tasks such as "optimize the system", "reduce overhead",
  "rewrite the hot path", or "refactor the pipeline" unless the task also
  provides concrete target_symbols/target_regions, preserved_invariants, a
  narrow edit_scope, and a validator.failure_condition.
- If a task changes behavior ordering, data/control flow, state lifecycle,
  execution structure, scheduling, batching, parallelism, resource lifetime,
  API/schema contracts, or side-effect placement,
  set risk_level to "structural" and tactic_stage to "structural_probe" for
  the first attempt. Provide probe_plan, probe_diff_contract,
  invariant_evidence, and a rollback_or_shrink_plan. A structural task must
  start as a small reversible probe, not a full rewrite.
- For structural_probe tasks, make probe_diff_contract a diff-level execution
  contract: allowed files/regions, expected changed region, max files/hunks/
  changed lines/functions, forbidden regions, required unchanged regions, and
  the observation. When rewriting after active-task drift, also include one
  concrete allowed_edit_shape and any forbidden_edit_shapes so CODE has a
  micro edit shape to execute instead of inferring the shrink from prose.
  CODE diffs outside this contract are rejected before tests.
- Set risk_level to "local" and tactic_stage to "local_edit" only when the
  change is a narrow local edit that does not reorder behavior, state, data
  flow, control flow, or side effects.
- Always provide risk_evidence for local and structural tasks. Its field must
  be one of title, edit_scope, strategy_axis, family_key, or expected_signal,
  and its quote must be copied from that field. Do not cite
  correctness_rationale, fallback_plan, preserved_invariants, or
  invariant_evidence as risk evidence; those fields explain safety, not the
  requested edit's risk class.
- Use acceptance.kind "synthesized" for implementation tasks unless the request supplies
  an explicit command or metric acceptance.
- For command acceptance, include only human-supplied commands from the request or config.
- For metric acceptance, include the measurable command or leave commands empty when it
  should use the configured workflow metric command.
- Do not invent files, commands, benchmarks, or constraints that are absent from
  the current request/config/source context.
- When deterministic Spec grounding facts are supplied, choose implementation
  target_regions, deliverables, and probe_diff_contract expected_changed_regions
  only from writable allowed_target_regions. Read-only or imported symbols may
  be cited as context, hazards, invariants, or read_hints, but must not be
  changed targets.
- When accepted SPEC hypothesis options are supplied, create runnable
  implementation tasks only from those options. Every runnable task must copy
  one accepted `hypothesis_id`, and its target_regions must stay within that
  option's change_boundary.regions. If no hypothesis option was accepted, do
  not invent implementation tasks from free-form analysis or rejected prose.
- When a SPEC_IDEA advisory brief is supplied, do not silently drift away from
  its first feasible writable target. Either make that target the first runnable
  task, or add an `idea_rejection_reason:` entry to known_facts or
  decision_rules that cites deterministic grounding facts or recent failure
  memory. Do not reject a SPEC_IDEA target just because another idea "seems
  better"; performance quality is decided only by the metric gate.
- Do not include historical prior-run winners unless they are present in this run's input.
- When Spec focus includes an existing task graph and one rewrite target, treat
  the call as a targeted graph rewrite. Preserve sibling tasks by default; do
  not collapse the portfolio to a single replacement task. If you replace the
  rejected target, keep its task_id or set replaces_task_id to that task.
- When failure memory is supplied by issue scope, only entries labeled as
  current_repo_issues may become repair/syntax-fix tasks. Rejected candidate
  lessons are transient candidate-delta, patch, contract, metric, or design
  failures; use them only to avoid, shrink, or retarget candidate shapes. Do
  not infer that the current source has a syntax/test bug from a rejected
  candidate lesson.
- Keep task_graph to 3-8 tasks."""

SPEC_IDEA_SYSTEM = """You are the SPEC_IDEA node in a local coding-agent FSM.
Do not write code and do not emit run_spec JSON.
Think through the request, plan, source, semantic facts, grounding facts, and
recent failures. Output concise Markdown only:
- feasible writable targets and why they are writable/resolvable
- read-only or imported symbols that are context only and must not be changed
- 3-6 candidate task ideas, each with one target region, risk, smallest probe,
  validator signal, and reason it avoids recent failed shapes
- designs to reject before SPEC_FINALIZE

This is advisory input for a no-think JSON finalizer. Always emit final
Markdown content. Do not end after hidden reasoning."""


SPEC_THINK_BRIEF_SYSTEM = """You are the SPEC_THINK_BRIEF node in a local coding-agent FSM.
Do not write code and do not emit run_spec JSON.
Analyze the request, plan, source, semantic facts, grounding facts, and recent
failures so a separate no-think JSON finalizer can produce a bounded run spec.
Output compact Markdown only. Domain-specific symbols, APIs, algorithms, and
failure terms are allowed inside fields, but do not rely on prose paragraphs.

Emit 1-5 typed hypothesis options in this exact block form:

BEGIN_HYPOTHESIS_OPTION hyp-short-id
hypothesis: What causal claim this option tests and why it helps the objective.
change_boundary.regions: relative/path.py::symbol
change_boundary.kind: local_edit|structural_probe|structural_expand|other
change_boundary.minimality_claim: Why this is the smallest resolvable boundary.
causal_evidence: Source, test, metric, or failure-log evidence for the claim.
expected_signal.validator_kind: metric|command|synthesized|diagnostic
expected_signal.command_or_metric: Which configured command, metric, or signal observes it.
expected_signal.success_condition: What concrete observation counts as improvement or safety.
invariants: Behavior, API, data-flow, or correctness constraints to preserve.
fallback.on_failure: Which axis to abandon, shrink, or retarget if this fails.
fallback.preserve: Which siblings, invariants, or facts must remain usable.
why_not_smaller: Why the controller should not split this boundary further.
END_HYPOTHESIS_OPTION

Rules:
- The controller validates structure only; it will not interpret domain terms
  or split your option into smaller tasks for you.
- Use change_boundary.regions from deterministic writable grounding facts when
  they are supplied.
- If an idea is broad, explain why it cannot be made smaller in why_not_smaller
  or leave it out.
- Do not emit run_spec JSON, task_graph JSON, code, patches, or shell commands.
- Options that omit evidence, observable signal, fallback, boundary, or
  why_not_smaller may be rejected before the finalizer runs.

This is an analysis-only brief. It may use thinking internally, but the visible
brief must be concise and must not contain JSON run-spec artifacts."""

ACCEPTANCE_SYNTH_SYSTEM = """You are the ACCEPT_SYNTH node in a local coding-agent FSM.
Write task-local acceptance tests before implementation.
Output strict JSON:
{
  "files": [
    {"path": "test_task.py", "content": "test code"}
  ]
}
Rules:
- Paths must be relative filenames, not absolute paths.
- Write only test files for the current task.
- Use Python stdlib unittest-compatible tests unless the task explicitly requires another format.
- Each concrete requirement needs at least one specific assertion or input/output pair.
- Tests must fail before the task implementation exists or is completed.
- Do not output shell commands; the controller will build the acceptance command.
- Do not test private model reasoning or unrelated behavior.
- No markdown fences, no commentary outside JSON."""

REFLECT_SYSTEM = """You are the REFLECT node in a local coding-agent FSM.
Do not write code. Analyze only the latest rejected attempt and feedback.
Output exactly 1-3 concise Markdown bullets:
- why the previous attempt failed
- what must change in the next CODE attempt
- what pattern must not be repeated
Always emit final content. Do not end after hidden reasoning."""

BRAINSTORM_SYSTEM = """You are the BRAINSTORM node in a local coding-agent FSM.
The search is stuck in a local minimum. Do not write code.
Output exactly 3 numbered tactics in Markdown.
Each tactic must:
- be a different algorithmic or architectural paradigm
- avoid repeating the rejected patterns
- derive from an open or repairable Run-local spec task when one is supplied
- name exactly one stable strategy_axis; prefer any domain axes explicitly
  requested by the task, otherwise use the supplied Known strategy axes
- if Open novelty lanes are provided, include one novelty_lane line copied from
  those lanes before choosing the family_key
- include one family_key line with a concise lowercase_snake_case tactic-family
  label, such as input_validation, data_flow_cleanup, api_contract_alignment,
  error_recovery, performance_hot_path, parser_variant, or state_lifecycle
- if New family required is true, every tactic must use a family_key that is not
  listed in Forbidden family aliases and must avoid the same idea under a new name
- put only secondary or uncertain category names after "new_axis_suggestion:"
- include one concrete implementation hook in the supplied source
- include one spec_task_id line when the tactic maps to a Run-local spec task
Keep each tactic to 2 short sentences.
Always emit final content. Do not end after hidden reasoning."""

CODE_SYSTEM = """You are the CODE node in a local coding-agent FSM.
Use only the supplied plan, source files, external advisory context, and latest
test failure.
Output strict JSON:
{
  "changes": [{
    "path": "relative/path.py",
    "target": "exact existing text",
    "replacement": "new text",
    "reason": "why",
    "target_region": "relative/path.py::symbol",
    "start_line": 10,
    "end_line": 14,
    "anchor_before": "nearby exact text before target",
    "anchor_after": "nearby exact text after target"
  }]
}
Rules:
- Modify only listed files.
- Prefer exact target/replacement snippets.
- For every replacement edit, copy the target verbatim from the current supplied source.
- For every replacement edit, include start_line/end_line and small anchor_before/anchor_after
  hints when available. Line numbers are hints only; never include line-number prefixes
  inside target or replacement.
- Do not invent or paraphrase a replacement target; stale targets will be rejected.
- Use "patch" only if target/replacement is impossible.
- Use full-file "content" only for very small files.
- When the active task names one target_region, emit one change for that target
  unless the prompt explicitly asks for a queued multi-candidate format.
- Preserve existing public behavior unless the plan says otherwise.
- Do not output comment-only, formatting-only, or explanatory placeholder changes.
- No markdown fences, no commentary outside JSON."""

CODE_XML_SYSTEM = """You are the CODE node in a local coding-agent FSM.
Use only the supplied plan, source files, external advisory context, and latest
test failure.
Do not output JSON. Output exactly one small candidate in this XML-like format:
<candidates>
<candidate id="1">
<strategy_axis>one_known_axis</strategy_axis>
<reason>one short sentence</reason>
<change>
<path>relative/path.py</path>
<search>
exact existing code, copied verbatim; 1-40 lines only
</search>
<replace>
new code, copied verbatim; 1-40 lines only
</replace>
</change>
</candidate>
</candidates>
Rules:
- Modify only listed files.
- Emit exactly one <candidate> and exactly one <change>.
- Include exactly one <strategy_axis> tag inside <candidate>.
- Keep <reason> to one sentence.
- Keep <search> and <replace> under 40 lines each.
- Never replace an entire function or class.
- Prefer a tiny local edit around the immediate bottleneck.
- The <search> block must match existing code exactly, including whitespace.
- Copy <search> verbatim from the current supplied source; do not paraphrase stale code.
- Put raw code inside <search> and <replace>; do not JSON-escape quotes or newlines.
- Do not add markdown fences or prose outside <candidates>.
- Do not output comment-only, formatting-only, or explanatory placeholder changes."""

TEST_SYSTEM = """You are the TEST node in a local coding-agent FSM.
Given test output, decide whether the work is complete or needs another CODE loop.
Output strict JSON:
{"status":"pass|retry|fail","reason":"short reason","next_focus":"specific fix target"}
Do not include markdown or prose outside JSON."""


def plan_prompt(state: AgentState, project_context: str = "") -> list[dict[str, str]]:
    parts = [state.user_request]
    if project_context:
        parts.append(f"Project context:\n{project_context}")
    external_context = external_context_block(state)
    if external_context:
        parts.append(external_context)
    user_content = "\n\n".join(parts)
    return [
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def read_prompt(state: AgentState) -> list[dict[str, str]]:
    user_content = f"Plan:\n{state.plan_markdown}"
    focused = state.scratch.get("focused_read_context")
    if isinstance(focused, str) and focused.strip():
        user_content = f"{user_content}\n\nFocused read context:\n{focused}"
    external_context = external_context_block(state)
    if external_context:
        user_content = f"{user_content}\n\n{external_context}"
    return [
        {"role": "system", "content": READ_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def semantic_analysis_prompt(state: AgentState, focus: str = "") -> list[dict[str, str]]:
    source_blocks = "\n\n".join(
        f"### {snap.path}\n```text\n{slice_text(snap.content)}\n```" for snap in state.file_context
    )
    focus_block = f"\n\nAnalysis focus:\n{focus}" if focus.strip() else ""
    external_context = external_context_block(state)
    external_block = f"\n\n{external_context}" if external_context else ""
    return [
        {"role": "system", "content": SEMANTIC_ANALYSIS_SYSTEM},
        {
            "role": "user",
            "content": (
                f"User request:\n{state.user_request}\n\n"
                f"Plan:\n{state.plan_markdown}\n\n"
                f"Source files:\n{source_blocks}"
                f"{external_block}"
                f"{focus_block}"
            ),
        },
    ]


def spec_idea_prompt(state: AgentState, focus: str = "") -> list[dict[str, str]]:
    source_blocks = "\n\n".join(
        f"### {snap.path}\n```text\n{slice_text(snap.content)}\n```" for snap in state.file_context
    )
    semantic_analysis = state.scratch.get("semantic_analysis")
    semantic_block = (
        f"\n\nSemantic analysis:\n{semantic_analysis}"
        if isinstance(semantic_analysis, str) and semantic_analysis.strip()
        else ""
    )
    focus_block = f"\n\nSpec focus and grounding facts:\n{focus}" if focus.strip() else ""
    return [
        {"role": "system", "content": SPEC_IDEA_SYSTEM},
        {
            "role": "user",
            "content": (
                f"User request:\n{state.user_request}\n\n"
                f"Plan:\n{state.plan_markdown}\n\n"
                f"Source files:\n{source_blocks}"
                f"{semantic_block}"
                f"{focus_block}"
            ),
        },
    ]


def spec_think_brief_prompt(state: AgentState, focus: str = "") -> list[dict[str, str]]:
    source_blocks = "\n\n".join(
        f"### {snap.path}\n```text\n{slice_text(snap.content)}\n```" for snap in state.file_context
    )
    semantic_analysis = state.scratch.get("semantic_analysis")
    semantic_block = (
        f"\n\nSemantic analysis:\n{semantic_analysis}"
        if isinstance(semantic_analysis, str) and semantic_analysis.strip()
        else ""
    )
    focus_block = f"\n\nSpec focus and deterministic constraints:\n{focus}" if focus.strip() else ""
    return [
        {"role": "system", "content": SPEC_THINK_BRIEF_SYSTEM},
        {
            "role": "user",
            "content": (
                f"User request:\n{state.user_request}\n\n"
                f"Plan:\n{state.plan_markdown}\n\n"
                f"Source files:\n{source_blocks}"
                f"{semantic_block}"
                f"{focus_block}"
            ),
        },
    ]


def spec_hypothesis_repair_prompt(
    state: AgentState,
    *,
    brief: str,
    options_payload: dict,
    focus: str = "",
) -> list[dict[str, str]]:
    focus_block = f"\n\nSpec focus and deterministic constraints:\n{focus}" if focus.strip() else ""
    user = (
        "The previous SPEC_THINK_BRIEF did not produce accepted typed "
        "hypothesis options. Repair the typed hypothesis blocks so they address "
        "the controller validation issues below.\n\n"
        f"User request:\n{state.user_request}\n\n"
        f"Plan:\n{state.plan_markdown}\n\n"
        f"Previous brief preview:\n{brief[:4000]}\n\n"
        "Controller validation summary:\n"
        f"{json.dumps(options_payload, ensure_ascii=False, indent=2)[:4000]}"
        f"{focus_block}\n\n"
        "Repair rules:\n"
        "- Fix every validation issue listed above; do not repeat rejected shapes.\n"
        "- For structural_hypothesis_boundary_kind_mismatch, either use "
        "change_boundary.kind=structural_probe|structural_expand, or shrink "
        "the hypothesis to a genuinely local guarded edit that no longer claims "
        "broad structural behavior.\n"
        "- Do not return change_boundary.kind=local_edit for options that mention "
        "broad data transformation, dispatch or scheduling changes, pipeline-wide "
        "behavior, whole-loop changes, cross-region coordination, or global "
        "state/ordering changes. Those shapes must be structural_probe/"
        "structural_expand unless rewritten as one guarded statement-level "
        "local edit.\n"
        "- change_boundary.regions must be exact deterministic grounding regions; "
        "do not add parenthetical notes, explanations, or prose inside region "
        "strings.\n\n"
        "Return only 1-3 BEGIN_HYPOTHESIS_OPTION blocks. Do not emit run_spec "
        "JSON, prose outside blocks, code, shell commands, or markdown fences."
    )
    return [
        {"role": "system", "content": SPEC_THINK_BRIEF_SYSTEM},
        {"role": "user", "content": user},
    ]


def spec_hypothesis_task_repair_prompt(
    state: AgentState,
    *,
    focus: str,
    repair_context: str,
) -> list[dict[str, str]]:
    source_blocks = "\n\n".join(
        f"### {snap.path}\n```text\n{slice_text(snap.content)}\n```" for snap in state.file_context
    )
    semantic_analysis = state.scratch.get("semantic_analysis")
    semantic_block = (
        f"\n\nSemantic analysis:\n{semantic_analysis}"
        if isinstance(semantic_analysis, str) and semantic_analysis.strip()
        else ""
    )
    focus_block = f"\n\nSpec focus:\n{focus}" if focus.strip() else ""
    return [
        {"role": "system", "content": SPEC_SYSTEM},
        {
            "role": "user",
            "content": (
                "Repair the previous SPEC_FINALIZE output by converting accepted "
                "typed hypothesis options into the smallest runnable task graph "
                "that passes the deterministic quality gate.\n\n"
                f"User request:\n{state.user_request}\n\n"
                f"Plan:\n{state.plan_markdown}\n\n"
                f"Source files:\n{source_blocks}"
                f"{semantic_block}"
                f"{focus_block}\n\n"
                f"Repair context:\n{repair_context}\n\n"
                "Return one JSON run_spec object only. Copy accepted hypothesis_id "
                "values exactly. For each copied hypothesis_id, preserve its accepted "
                "change_boundary.regions and boundary kind as the task contract: target_regions "
                "must be inside the accepted regions, and structural_probe/structural_expand "
                "boundaries must remain risk_level=structural and tactic_stage=structural_probe "
                "unless the task explicitly narrows to one guarded local probe inside the "
                "accepted regions. Do not emit markdown, prose, code, or shell commands."
            ),
        },
    ]


def spec_prompt(state: AgentState, focus: str = "") -> list[dict[str, str]]:
    source_blocks = "\n\n".join(
        f"### {snap.path}\n```text\n{slice_text(snap.content)}\n```" for snap in state.file_context
    )
    semantic_analysis = state.scratch.get("semantic_analysis")
    semantic_block = (
        f"\n\nSemantic analysis:\n{semantic_analysis}"
        if isinstance(semantic_analysis, str) and semantic_analysis.strip()
        else ""
    )
    focus_block = f"\n\nSpec focus:\n{focus}" if focus.strip() else ""
    return [
        {"role": "system", "content": SPEC_SYSTEM},
        {
            "role": "user",
            "content": (
                f"User request:\n{state.user_request}\n\n"
                f"Plan:\n{state.plan_markdown}\n\n"
                f"Source files:\n{source_blocks}"
                f"{semantic_block}"
                f"{focus_block}"
            ),
        },
    ]


def acceptance_synth_prompt(state: AgentState, task: dict, acceptance_dir: str) -> list[dict[str, str]]:
    source_blocks = "\n\n".join(
        f"### {snap.path}\n```text\n{slice_text(snap.content)}\n```" for snap in state.file_context
    )
    run_spec = state.scratch.get("run_spec")
    spec_block = (
        json.dumps(run_spec, ensure_ascii=False, indent=2)
        if isinstance(run_spec, dict) and run_spec
        else str(run_spec or "")
    )
    return [
        {"role": "system", "content": ACCEPTANCE_SYNTH_SYSTEM},
        {
            "role": "user",
            "content": (
                f"User request:\n{state.user_request}\n\n"
                f"Run-local spec:\n{spec_block}\n\n"
                f"Current task:\n{json.dumps(task, ensure_ascii=False, indent=2)}\n\n"
                f"Acceptance directory:\n{acceptance_dir}\n\n"
                f"Task-scoped source context:\n{source_blocks or '<none>'}"
            ),
        },
    ]


def reflect_prompt(state: AgentState, feedback_notes_limit: int = 12) -> list[dict[str, str]]:
    external_context = external_context_block(state)
    external_block = f"\n\n{external_context}" if external_context else ""
    todo_chain = state.scratch.get("todo_observation_chain")
    todo_chain_block = (
        f"\n\nObservation-backed todo continuation:\n{todo_chain}"
        if isinstance(todo_chain, str) and todo_chain.strip()
        else ""
    )
    return [
        {"role": "system", "content": REFLECT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"User request:\n{state.user_request}\n\n"
                f"Plan:\n{state.plan_markdown}\n\n"
                f"{external_block}"
                f"{todo_chain_block}\n\n"
                f"Latest test summary:\n{state.latest_test_summary()}\n\n"
                f"Recent agent feedback:\n{state.recent_notes_summary(feedback_notes_limit)}"
            ),
        },
    ]


def brainstorm_prompt(
    state: AgentState,
    reject_summary: str,
    cooled_axes: list[str],
    known_axes: list[str],
    todo_ledger_summary: str = "",
    forbidden_family_aliases: list[str] | None = None,
    open_novelty_lanes: list[str] | None = None,
    new_family_required: bool = False,
    feedback_notes_limit: int = 8,
) -> list[dict[str, str]]:
    source_blocks = "\n\n".join(
        f"### {snap.path}\n```text\n{slice_text(snap.content)}\n```" for snap in state.file_context
    )
    external_context = external_context_block(state)
    external_block = f"{external_context}\n\n" if external_context else ""
    semantic_analysis = state.scratch.get("semantic_analysis")
    semantic_block = (
        f"Semantic analysis:\n{semantic_analysis}\n\n"
        if isinstance(semantic_analysis, str) and semantic_analysis.strip()
        else ""
    )
    run_spec = state.scratch.get("run_spec")
    spec_block = (
        f"Run-local spec:\n{run_spec}\n\n"
        if isinstance(run_spec, str) and run_spec.strip()
        else (
            "Run-local spec:\n"
            f"{json.dumps(run_spec, ensure_ascii=False, indent=2)}\n\n"
            if isinstance(run_spec, dict) and run_spec
            else ""
        )
    )
    return [
        {"role": "system", "content": BRAINSTORM_SYSTEM},
        {
            "role": "user",
            "content": (
                f"User request:\n{state.user_request}\n\n"
                f"Plan:\n{state.plan_markdown}\n\n"
                f"Source files:\n{source_blocks}\n\n"
                f"{external_block}"
                f"{semantic_block}"
                f"{spec_block}"
                f"Current best/test summary:\n{state.latest_test_summary()}\n\n"
                f"Known strategy axes:\n{', '.join(known_axes)}\n\n"
                f"Cooled axes:\n{', '.join(cooled_axes) if cooled_axes else 'none'}\n\n"
                "Forbidden family aliases:\n"
                f"{', '.join(forbidden_family_aliases or []) if forbidden_family_aliases else 'none'}\n\n"
                "Open novelty lanes:\n"
                f"{chr(10).join(f'- {lane}' for lane in (open_novelty_lanes or [])) if open_novelty_lanes else 'none'}\n\n"
                f"New family required:\n{str(new_family_required).lower()}\n\n"
                f"Recent reject summary:\n{reject_summary}\n\n"
                f"Durable todo ledger summary:\n{todo_ledger_summary or 'none'}\n\n"
                f"Recent agent feedback:\n{state.recent_notes_summary(feedback_notes_limit)}"
            ),
        },
    ]


def code_prompt(
    state: AgentState,
    feedback_notes_limit: int = 12,
    output_format: str = "json",
    cache_friendly_layout: bool = True,
) -> list[dict[str, str]]:
    source_blocks = "\n\n".join(
        f"### {snap.path}\n```text\n{slice_text(snap.content)}\n```" for snap in state.file_context
    )
    external_context = external_context_block(state)
    external_block = f"\n\n{external_context}" if external_context else ""
    semantic_analysis = state.scratch.get("semantic_analysis")
    semantic_block = (
        f"\n\nSemantic analysis:\n{semantic_analysis}"
        if isinstance(semantic_analysis, str) and semantic_analysis.strip()
        else ""
    )
    run_spec = state.scratch.get("run_spec")
    spec_block = (
        f"\n\nRun-local spec:\n{run_spec}"
        if isinstance(run_spec, str) and run_spec.strip()
        else (
            "\n\nRun-local spec:\n"
            f"{json.dumps(run_spec, ensure_ascii=False, indent=2)}"
            if isinstance(run_spec, dict) and run_spec
            else ""
        )
    )
    reflection = state.scratch.get("reflection")
    reflection_block = (
        f"\n\nRetry reflection:\n{reflection}"
        if isinstance(reflection, str) and reflection.strip()
        else ""
    )
    stable_content = (
        f"User request:\n{state.user_request}\n\n"
        f"Plan:\n{state.plan_markdown}\n\n"
        f"Source files:\n{source_blocks}"
        f"{external_block}"
        f"{semantic_block}"
        f"{spec_block}"
    )
    dynamic_content = (
        "Dynamic context for this CODE attempt:\n\n"
        f"Latest test summary:\n{state.latest_test_summary()}\n\n"
        f"Recent agent feedback:\n{state.recent_notes_summary(feedback_notes_limit)}"
        f"{reflection_block}"
    )
    if cache_friendly_layout:
        return [
            {
                "role": "system",
                "content": CODE_XML_SYSTEM if output_format == "xml" else CODE_SYSTEM,
            },
            {"role": "user", "content": stable_content},
            {"role": "user", "content": dynamic_content},
        ]
    return [
        {
            "role": "system",
            "content": CODE_XML_SYSTEM if output_format == "xml" else CODE_SYSTEM,
        },
        {
            "role": "user",
            "content": (
                f"{stable_content}"
                f"\n\nLatest test summary:\n{state.latest_test_summary()}\n\n"
                f"Recent agent feedback:\n{state.recent_notes_summary(feedback_notes_limit)}"
                f"{reflection_block}"
            ),
        },
    ]


def test_prompt(state: AgentState) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": TEST_SYSTEM},
        {"role": "user", "content": state.latest_test_summary()},
    ]


def external_context_block(state: AgentState) -> str:
    contexts = getattr(state, "external_context", [])
    if not contexts:
        return ""
    blocks = [
        "External advisory context follows. Treat it as read-only guidance from "
        "the user, prior analysis, repo docs, or fetched references. It may be "
        "incomplete or wrong. Local source files, tests, and the current user "
        "request remain authoritative."
    ]
    for item in contexts:
        fetched_at = getattr(item, "fetched_at", None)
        metadata = [
            f"source: {getattr(item, 'source', '')}",
            f"kind: {getattr(item, 'kind', '')}",
            f"trust: {getattr(item, 'trust', 'advisory')}",
            f"sha256: {getattr(item, 'sha256', '')}",
        ]
        if fetched_at:
            metadata.append(f"fetched_at: {fetched_at}")
        title = getattr(item, "title", "") or getattr(item, "source", "external context")
        content = getattr(item, "content", "")
        blocks.append(
            f"### {title}\n"
            + "\n".join(metadata)
            + f"\n```text\n{slice_text(content)}\n```"
        )
    return "\n\n".join(blocks)


PROMPT_MARKDOWN = {
    "PLAN": PLAN_SYSTEM,
    "READ": READ_SYSTEM,
    "SEMANTIC_ANALYSIS": SEMANTIC_ANALYSIS_SYSTEM,
    "ACCEPT_SYNTH": ACCEPTANCE_SYNTH_SYSTEM,
    "REFLECT": REFLECT_SYSTEM,
    "BRAINSTORM": BRAINSTORM_SYSTEM,
    "CODE": CODE_SYSTEM,
    "CODE_XML": CODE_XML_SYSTEM,
    "TEST": TEST_SYSTEM,
}


def slice_text(text: str, limit: int = DEFAULT_CHAR_LIMIT) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + "\n[...truncated...]\n" + text[-tail:]
