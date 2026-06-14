# Local LLM Autonomy Experiment - 2026-06-14

## Summary

The 2026-06-14 M2 Max experiments tried to turn a local Qwen 3.6 35B A3B
model into an autonomous optimization loop for the Anthropic performance
take-home. The effort produced useful controller improvements, but did not
achieve reliable local-LLM self-driving optimization.

Final verdict: the local model can follow strong steering well enough to choose
the right broad optimization family, and the controller can now land patches
reliably. It still cannot reliably select and execute the metric-bearing code
change without repeated no-op or inert variants. In practice, the run remains a
guided stochastic search rather than autonomous engineering.

## Goal

The target was a small local loop that could:

1. Read the benchmark and current source.
2. Choose an optimization direction.
3. Emit patch candidates.
4. Validate correctness and cycle count.
5. Learn from rejected candidates.
6. Continue until it found a lower `CYCLES` result.

The practical aspiration was local-model "autonomous driving": the controller
would provide deterministic apply/test guardrails, while the local model would
handle the optimization reasoning and next-candidate selection.

## What Changed

The main work happened across issues #37, #38, #39, #40, and #41.

### Output and Repair Plumbing

- XML candidate output was made the simple-mode path.
- Candidate output reminders were split into independent helpers.
- Malformed candidate repair became output-format-aware, so XML output no
  longer falls back to JSON repair prompts.
- Patch-miss repair was extended to no-op/reapply cases.
- Mixed `patch_noop + target_not_found` failures now prefer the current-source
  no-op repair path.
- XML candidates now preserve location hints:
  - `start_line`
  - `end_line`
  - `anchor_before`
  - `anchor_after`

The important result: patch landing improved substantially. Later smoke runs
had `patch_miss=0`.

### GPT-5.5 Request and Hints

Issue #38 added visible experiment artifacts:

- `gpt-5.5-request.txt`
- `gpt-5.5-hints.md`
- `gpt-5.5-agent-config.json`

These artifacts strongly instructed the model to focus on conservative VLIW
packing in `KernelBuilder.build` rather than broad `build_kernel` rewrites.

This worked: the model stopped drifting into broad vectorization and repeatedly
targeted the intended family.

### Simple Mode Cleanup

Issue #40 removed a custom simple-mode plateau guard that was duplicating the
shared novelty system. The simple preset now uses the shared
`candidate_novelty_gate` and leaves simple reports opt-in.

Issue #41 defined `simple` as a raw-ish measurement loop rather than a practical
middle scaffold:

- `simple_thinking_brief_enabled` defaults to `false`.
- Reasoning-only thinking briefs are rejected unless explicitly enabled.
- The GPT-5.5 experiment config keeps advisory artifacts through explicit
  opt-in.
- A deterministic e2e test now verifies that a scripted XML model can fix a real
  Python bug and reach `DONE`.

## Run Results

### Baseline

All runs used:

- Baseline metric: `CYCLES 147734`
- M2 Max host: `skshim-m2max`
- Model: `qwen3.6:35b-a3b-coding-mxfp8` through Ollama
- Primary benchmark command: `/opt/homebrew/bin/python3 perf_takehome.py`

### Successful Improvement Runs

Two runs found real improvements:

- `b6a35ab` run: best `CYCLES 123158`
- `7db93ee` run: best `CYCLES 119062`

The common feature was not just the chosen algorithm family. The successful
patches changed the default `KernelBuilder.build(body)` path directly, replacing
the naive one-slot-per-bundle lowering with a conservative ALU pair packer.

That made the optimization execute in the real measured path.

### Failed Long Runs

The later long runs showed the core failure mode.

#### 100-loop run at `560c4ad`

Run directory:

```text
/Users/m2max/tmp/local-micro-agent-homework-runs/run-20260614-134556-agents-gpt55-request-64k-100loop-560c4ad-ollama-qwen36-35b-a3b-mxfp8-simple-plateau-guard-longrun
```

Result:

- Candidates: 100
- `no_improvement=95`
- `correctness_failure=5`
- `patch_miss=0`
- Metrics observed: `147734` only
- Final direct validation: `CYCLES 147734`

The simple report grouped the repeated idea, but the custom plateau guard was
too fine-grained and only steered once.

#### 50-loop run at `d005fe3`

Run directory:

```text
/Users/m2max/tmp/local-micro-agent-homework-runs/run-20260614-1540-agents-gpt55-request-64k-50loop-d005fe3-ollama-qwen36-35b-a3b-mxfp8-simple-clean-run
```

Result:

- Final state: `failed`
- `loop_count=49`
- Candidate records: 43
- `no_improvement=36`
- `correctness_failure=7`
- `patch_miss=0`
- Metrics observed: `147734` only
- Final direct validation: `CYCLES 147734`

Runtime caveat:

- The last 7 loop attempts hit `Connection refused` from the local Ollama
  endpoint.
- Those attempts consumed loop budget without valid model candidates.
- The run is therefore best interpreted as a 43-candidate valid run plus 7
  infrastructure failures.

## Why Improvements Happened

The improvements were not pure random luck. The strong request/hints pushed the
model into the right strategy family: VLIW packing in `KernelBuilder.build`.

The randomness was inside that family. A candidate helped only when it changed
the default measured path.

Good shape:

```python
def build(self, slots):
    # conservative packer runs for normal self.build(body)
    ...
```

Inert shape:

```python
def build(self, slots, vliw=False):
    if not vliw:
        # old naive lowering
        ...
        return instrs

    # optimized packer is unreachable from self.build(body)
    ...
```

The failed candidates often described the right optimization but hid it behind a
branch that the benchmark never called. They applied cleanly and passed
correctness, but left `CYCLES` unchanged.

## What Worked

- Strong request/hint artifacts changed the model's strategy family.
- XML location hints plus existing line/anchor retargeting eliminated most patch
  landing failures.
- Output-aware repair removed a wrong-format repair path.
- Simple-mode cleanup reduced controller complexity and clarified the preset's
  purpose.
- Deterministic e2e coverage now protects the simple loop's basic ability to go
  from bug to `DONE`.

## What Failed

The local model did not reliably connect optimization intent to the metric-bearing
execution path.

It repeatedly made clean, correctness-preserving edits that were inert for the
measured benchmark. The controller could detect `no_improvement`, but did not
have a semantic gate strong enough to classify:

- "you added an unused optimized branch"
- "the default call path did not change"
- "this is the same metric-neutral schedule family"

Without that feedback, the model kept re-emitting variants of the same broad
`KernelBuilder.build` packer idea.

## Lessons

1. Patch landing is a solvable controller problem.
2. Strategy-family steering is possible with strong prompts.
3. Reliable metric-bearing edits require execution-path awareness, not just
   region/family novelty.
4. A local model can be useful inside a bounded search loop, but this model did
   not autonomously drive the optimization to completion.
5. "Raw-ish simple" and "practical assisted search" should remain separate
   modes. Mixing them made earlier results hard to interpret.

## Future Work

If this line is resumed, the next useful controller feature is not another
simple prompt. It should be a metric-bearing-path check:

- Detect when a candidate adds an optimized branch that no measured call uses.
- Detect whether the default `self.build(body)` path changed materially.
- Feed that finding back as structured negative evidence.
- Optionally reject or cool down branch-gated variants after one no-improvement.

That would move the loop from generic stochastic search toward a more
execution-aware local optimization assistant. It would still not prove local LLM
autonomy, but it would address the specific failure observed in these runs.
