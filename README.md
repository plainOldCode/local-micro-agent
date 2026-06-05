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

For local models that struggle to JSON-escape multi-line code snippets, set
`workflow.code_output_format="xml"`. In XML mode the CODE node emits raw
`<search>` and `<replace>` blocks inside `<candidates>` instead of putting
multi-line code inside JSON strings. Set `workflow.log_raw_model_outputs=true`
to save malformed model outputs under `.local_micro_agent/raw_model_outputs`
when parsing or repair fails.

Use `workflow.project_instruction_files` to name instruction files explicitly.
Use `workflow.project_context_files` to fully override the auto-detected context
set, or set `workflow.readme_first=false` for controlled experiments.

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
    "writable_files": ["perf_takehome.py"],
    "test_commands": ["python tests/submission_tests.py"]
  }
}
```

## Focused Source Context

For narrow Python edits, `workflow.context_symbols` can replace full-file
CODE context with exact function/class excerpts:

```json
{
  "workflow": {
    "seed_files": ["perf_takehome.py"],
    "context_symbols": {
      "perf_takehome.py": ["build_kernel", "Builder.emit"]
    }
  }
}
```

## Files

- `config.example.json`: provider and workflow configuration.
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
narrow experiments.
