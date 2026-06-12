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
- `spec`: `structural` plus the experimental spec-mode scheduler. A v2
  run_spec task graph becomes the deterministic loop driver, with the
  grounded/spec-quality/design-contract/probe gates enabled as one vetted
  bundle.
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

_SPEC: dict[str, Any] = {
    **_STRUCTURAL,
    "spec_mode": True,
    "run_spec_enabled": True,
    "run_spec_after_read": False,
    "spec_resume": True,
    "spec_max_tasks": 24,
    "spec_task_attempt_budget": 8,
    "spec_task_recovery_rounds": 2,
    "spec_acceptance_dir": ".lma_acceptance",
    "spec_acceptance_command_template": "{quoted_python} -m unittest discover -s {quoted_dir} -p 'test*.py'",
    "spec_acceptance_review": False,
    "spec_default_acceptance_kind": "synthesized",
    "spec_acceptance_synth_retries": 1,
    "spec_regression_scope": "all",
    "spec_invariant_commands": [],
    "spec_progress_path": ".local_micro_agent/spec_progress.jsonl",
    "spec_report_path": ".local_micro_agent/spec_report.md",
    "spec_design_contract_gate": True,
    "spec_grounding_gate": True,
    "spec_quality_gate": True,
    "spec_quality_rewrite_attempts": 2,
    "spec_synth_call_budget": 24,
    "spec_gate_soft_fallback": True,
    "spec_structural_risk_gate": True,
    "spec_two_call_synthesis": True,
    "spec_probe_diff_contract_required": True,
    "probe_diff_contract_gate": True,
    "spec_active_task_drift_streak_limit": 3,
    "spec_active_task_drift_same_fingerprint_limit": 2,
    "spec_active_task_drift_rewrite_attempts": 1,
    "spec_drift_saturation_threshold": 3,
    "spec_portfolio_recovery_rounds": 2,
    "spec_graph_reseed_attempts": 2,
    "continue_after_improvement": False,
    "deterministic_test_decision": True,
}

WORKFLOW_PRESETS: dict[str, dict[str, Any]] = {
    "minimal": _MINIMAL,
    "search": _SEARCH,
    "structural": _STRUCTURAL,
    "spec": _SPEC,
}


def apply_workflow_preset(config: dict[str, Any]) -> dict[str, Any]:
    """Return config with `workflow.preset` expanded.

    Explicit workflow keys always win over preset values. Returns the
    original config object unchanged when no preset is requested; raises
    ValueError for an unknown preset name.

    The expanded workflow records provenance in `preset_defaulted_keys`:
    the preset keys that were NOT explicit in the pre-expansion workflow.
    Re-expanding an already expanded config preserves that record, so
    consumers such as MicroAgent can distinguish a caller-supplied value
    from a preset default even when the config went through load_config().
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
    reserved = {"preset", "preset_defaulted_keys"}
    explicit = {k: v for k, v in workflow.items() if k not in reserved}
    prior_defaulted = workflow.get("preset_defaulted_keys")
    if isinstance(prior_defaulted, list):
        defaulted = sorted(str(key) for key in prior_defaulted if str(key) in preset)
    else:
        defaulted = sorted(key for key in preset if key not in explicit)
    merged_workflow = {**preset, **explicit}
    merged_workflow["preset"] = str(name)
    merged_workflow["preset_defaulted_keys"] = defaulted
    return {**config, "workflow": merged_workflow}
