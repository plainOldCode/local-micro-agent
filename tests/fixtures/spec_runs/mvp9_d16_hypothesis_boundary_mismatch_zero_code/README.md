# mvp9_d16_hypothesis_boundary_mismatch_zero_code

Zero-CODE spec quality gate failure fixture from the M2 Max clean spec-mode
smoke for commit `f5daad8`.

Source run:

`/Users/m2max/tmp/local-micro-agent-homework-runs/run-20260613-180907-agents-json-regex-64k-20loop-f5daad8-ollama-qwen36-35b-a3b-mxfp8-mvp9d16-fixture-followup-smoke`

Terminal summary:

- state: `failed`
- stop_reason: `spec_quality_gate_failed`
- code_test_loop_count: `0`
- zero_code_attempt: `True`
- spec_synth_calls_used: `5`

Failure shape:

- The typed hypothesis option was rejected with
  `structural_hypothesis_boundary_kind_mismatch`.
- `accepted_count` stayed `0`.
- The finalizer then emitted runnable tasks with no accepted hypothesis to copy,
  so the quality gate rejected all graph candidates with
  `hypothesis_option_missing`.

Included compact artifacts:

- `terminal_state.json`
- `spec_graph_candidates.jsonl`
- `spec_quality_report.json`
- `spec_hypothesis_options.json`
- `spec_hypothesis_option_rejections.jsonl`

Excluded artifacts include model streams, reasoning text, profile logs, full
trace output, and generated code snapshots.
