# MVP9-D15 Boundary Feedback R2 Fixture

Compact controller artifact fixture from the M2 Max clean spec-mode smoke for
commit `bf69405` (`Preserve hypothesis boundary repairs`).

Source run:

`/Users/m2max/tmp/local-micro-agent-homework-runs/run-20260613-163019-agents-json-regex-64k-20loop-bf69405-ollama-qwen36-35b-a3b-mxfp8-mvp9d15-boundary-feedback-smoke-r2`

The smoke reached three CODE/test loops and failed safely with
`search_frontier_exhausted_after_graph_reseed_exhausted`. Candidate patches were
rejected before apply as active-task drift, leaving the baseline metric unchanged.

This fixture intentionally keeps only stable controller artifacts:

- `terminal_state.json`
- `candidates.jsonl`
- `failure_signatures.jsonl`
- `spec_graph_candidates.jsonl`
- `run_spec.json`
- `todo_plan.json`

Excluded artifacts include model streams, reasoning text, profile logs, full
trace output, and generated code snapshots. Some verbose diagnostic fields are
trimmed so the fixture stays small while preserving the failure shape.
