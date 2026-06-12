from __future__ import annotations

import json
import unittest
from pathlib import Path

from local_micro_agent.orchestrator import load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigFileTests(unittest.TestCase):
    def test_a3b_mxfp8_config_keeps_tuned_lane_split(self) -> None:
        config_path = ROOT / "config/config.qwen36-35b-a3b-coding-mxfp8-ollama.json"
        raw_config = json.loads(config_path.read_text())
        config = load_config(config_path)
        self.assertEqual(raw_config["workflow"]["preset"], "spec")
        for absorbed_key in (
            "spec_design_contract_gate",
            "spec_grounding_gate",
            "spec_quality_gate",
            "spec_structural_risk_gate",
            "spec_two_call_synthesis",
            "spec_probe_diff_contract_required",
            "probe_diff_contract_gate",
        ):
            self.assertNotIn(
                absorbed_key,
                raw_config["workflow"],
                f"{absorbed_key} should come from the spec preset",
            )
        models = config["models"]
        providers = config["providers"]
        workflow = config["workflow"]

        self.assertEqual(models["coder"], "qwen36_a3b_ollama")
        self.assertEqual(models["tester"], "qwen36_a3b_ollama")
        coder = providers["qwen36_a3b_ollama"]
        self.assertFalse(coder["think"])
        self.assertEqual(coder["num_ctx"], 64000)
        self.assertEqual(coder["max_tokens"], 8192)
        self.assertEqual(coder["temperature"], 0.15)
        self.assertEqual(coder["extra_options"]["top_p"], 0.9)
        self.assertEqual(coder["extra_options"]["top_k"], 20)
        self.assertEqual(coder["extra_options"]["min_p"], 0)

        self.assertEqual(models["planner"], "qwen36_a3b_planner_fast")
        planner = providers["qwen36_a3b_planner_fast"]
        self.assertFalse(planner["think"])
        self.assertEqual(planner["num_ctx"], 64000)
        self.assertEqual(planner["max_tokens"], 4096)
        self.assertEqual(planner["temperature"], 0.2)
        self.assertEqual(planner["extra_options"]["top_p"], 0.8)
        self.assertEqual(planner["extra_options"]["top_k"], 20)
        self.assertEqual(planner["extra_options"]["min_p"], 0)

        self.assertEqual(models["plan_final"], "qwen36_a3b_plan_final")
        self.assertEqual(
            workflow["model_role_overrides_by_call_site"]["plan"],
            "plan_final",
        )
        self.assertNotIn("plan", workflow["reasoning_lane_call_sites"])
        plan_final = providers["qwen36_a3b_plan_final"]
        self.assertFalse(plan_final["think"])
        self.assertEqual(plan_final["num_ctx"], 64000)
        self.assertEqual(plan_final["max_tokens"], 12288)
        self.assertEqual(plan_final["temperature"], 0.7)
        self.assertEqual(plan_final["timeout_seconds"], 480)
        self.assertEqual(plan_final["extra_options"]["top_p"], 0.8)
        self.assertEqual(plan_final["extra_options"]["top_k"], 20)
        self.assertEqual(plan_final["extra_options"]["min_p"], 0)
        self.assertEqual(plan_final["extra_options"]["presence_penalty"], 1.2)

        self.assertEqual(models["spec_synth"], "qwen36_a3b_spec_synth")
        self.assertEqual(
            workflow["model_role_overrides_by_call_site"]["spec_synth"],
            "spec_synth",
        )
        spec_synth = providers["qwen36_a3b_spec_synth"]
        self.assertFalse(spec_synth["think"])
        self.assertEqual(spec_synth["num_ctx"], 64000)
        self.assertEqual(spec_synth["max_tokens"], 16384)
        self.assertEqual(spec_synth["temperature"], 0.7)
        self.assertEqual(spec_synth["format"], "json")
        self.assertEqual(spec_synth["timeout_seconds"], 480)
        self.assertEqual(spec_synth["extra_options"]["top_p"], 0.8)
        self.assertEqual(spec_synth["extra_options"]["top_k"], 20)
        self.assertEqual(spec_synth["extra_options"]["min_p"], 0)
        self.assertEqual(spec_synth["extra_options"]["presence_penalty"], 1.2)

        self.assertEqual(workflow["spec_default_acceptance_kind"], "metric")
        self.assertTrue(workflow["spec_force_default_acceptance_kind"])
        self.assertTrue(workflow["spec_force_metric_acceptance"])
        self.assertTrue(workflow["spec_tactic_portfolio"])
        self.assertTrue(workflow["spec_relax_failed_dependencies_with_budget"])
        self.assertTrue(workflow["spec_reopen_failed_portfolio_tasks"])
        self.assertEqual(workflow["spec_portfolio_recovery_rounds"], 2)
        self.assertTrue(workflow["spec_grounding_gate"])
        self.assertTrue(workflow["spec_quality_gate"])
        self.assertEqual(workflow["spec_quality_rewrite_attempts"], 2)
        self.assertEqual(workflow["spec_synth_call_budget"], 24)
        self.assertTrue(workflow["spec_gate_soft_fallback"])
        self.assertTrue(workflow["spec_two_call_synthesis"])
        self.assertEqual(workflow["spec_idea_model_role"], "reasoner")
        self.assertTrue(workflow["spec_thinking_brief_enabled"])
        self.assertEqual(workflow["spec_thinking_brief_model_role"], "reasoner")
        self.assertTrue(workflow["spec_thinking_brief_accept_reasoning_only"])
        self.assertEqual(workflow["spec_thinking_brief_char_limit"], 8000)
        self.assertTrue(workflow["spec_hypothesis_brief_enabled"])
        self.assertEqual(workflow["spec_hypothesis_max_options"], 5)
        self.assertTrue(workflow["spec_hypothesis_require_expected_signal"])
        self.assertTrue(workflow["spec_hypothesis_require_why_not_smaller"])
        self.assertEqual(workflow["spec_finalize_model_role"], "spec_synth")
        self.assertEqual(
            workflow["candidate_history_path"],
            ".local_micro_agent/candidates.jsonl",
        )
        self.assertTrue(workflow["preserve_correct_survivors"])
        self.assertTrue(workflow["spec_local_task_one_change"])
        self.assertEqual(workflow["local_task_max_changes"], 1)


if __name__ == "__main__":
    unittest.main()
