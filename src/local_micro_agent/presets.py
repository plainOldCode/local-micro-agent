"""Workflow presets: named bundles for the ~80 workflow flags.

A config may set `workflow.preset` to one of the names below instead of
hand-tuning each flag combination. Preset values are defaults: any key set
explicitly in `workflow` wins over the preset value, so a preset can be
used as a base and selectively overridden.

- `minimal`: conservative fix-the-tests loop. All exploration machinery
  (brainstorm, novelty gates, adaptive memory, run spec, structural
  lifecycle) stays off; deterministic test decisions and target-not-found
  repair stay on.
- `search`: long-running metric search. Enables the loop-control machinery
  validated by the 100-loop telemetry runs: conditional reflect, brainstorm
  after repeated rejections, novelty/axis/region gates, adaptive gate
  controller, candidate history and artifacts, and profiling.
- `structural`: `search` plus the scaffold/probe/expand machinery for
  multi-step refactors: run-spec task graph, semantic analysis, structural
  tactic lifecycle, and structural state checkpoints.
"""
from __future__ import annotations

from typing import Any

_MINIMAL: dict[str, Any] = {
    "max_code_test_loops": 3,
    "deterministic_test_decision": True,
    "retry_rejected_candidates": True,
    "repair_target_not_found": True,
    "reflect_before_retry": False,
    "brainstorm_after_rejections": 0,
    "candidate_novelty_gate": False,
    "adaptive_search_memory": False,
    "adaptive_search_reject_cooled_axes": False,
    "adaptive_search_reject_cooled_regions": False,
    "adaptive_gate_controller": False,
    "continue_after_improvement": False,
    "validated_pattern_followup": False,
    "semantic_analysis_after_read": False,
    "run_spec_after_read": False,
    "structural_tactic_lifecycle": False,
    "structural_state_checkpoint": False,
    "record_candidate_artifacts": False,
    "profile_agent": False,
}

_SEARCH: dict[str, Any] = {
    "max_code_test_loops": 25,
    "deterministic_test_decision": True,
    "retry_rejected_candidates": True,
    "repair_target_not_found": True,
    "reflect_before_retry": True,
    "reflect_conditionally": True,
    "brainstorm_after_rejections": 2,
    "candidate_novelty_gate": True,
    "adaptive_search_memory": True,
    "adaptive_search_reject_cooled_axes": True,
    "adaptive_search_reject_cooled_regions": True,
    "adaptive_gate_controller": True,
    "continue_after_improvement": True,
    "validated_pattern_followup": True,
    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
    "record_candidate_artifacts": True,
    "semantic_analysis_after_read": False,
    "run_spec_after_read": False,
    "structural_tactic_lifecycle": False,
    "structural_state_checkpoint": False,
    "profile_agent": True,
}

_STRUCTURAL: dict[str, Any] = {
    **_SEARCH,
    "semantic_analysis_after_read": True,
    "run_spec_after_read": True,
    "structural_tactic_lifecycle": True,
    "structural_state_checkpoint": True,
}

WORKFLOW_PRESETS: dict[str, dict[str, Any]] = {
    "minimal": _MINIMAL,
    "search": _SEARCH,
    "structural": _STRUCTURAL,
}


def apply_workflow_preset(config: dict[str, Any]) -> dict[str, Any]:
    """Return config with `workflow.preset` expanded.

    Explicit workflow keys always win over preset values. Returns the
    original config object unchanged when no preset is requested; raises
    ValueError for an unknown preset name.
    """
    workflow = config.get("workflow")
    if not isinstance(workflow, dict):
        return config
    name = workflow.get("preset")
    if not name:
        return config
    preset = WORKFLOW_PRESETS.get(str(name))
    if preset is None:
        valid = ", ".join(sorted(WORKFLOW_PRESETS))
        raise ValueError(f"Unknown workflow preset {name!r}; valid presets: {valid}")
    merged_workflow = {**preset, **{k: v for k, v in workflow.items() if k != "preset"}}
    merged_workflow["preset"] = str(name)
    return {**config, "workflow": merged_workflow}
