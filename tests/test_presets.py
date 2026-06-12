from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import json

from local_micro_agent.orchestrator import MicroAgent, load_config
from local_micro_agent.presets import WORKFLOW_PRESETS, apply_workflow_preset
from local_micro_agent.state import AgentState


class WorkflowPresetTests(unittest.TestCase):
    def test_no_preset_returns_config_unchanged(self) -> None:
        config = {"workflow": {"max_code_test_loops": 5}}
        self.assertIs(apply_workflow_preset(config), config)

    def test_search_preset_enables_search_machinery(self) -> None:
        config = apply_workflow_preset({"workflow": {"preset": "search"}})
        workflow = config["workflow"]
        self.assertTrue(workflow["adaptive_search_memory"])
        self.assertTrue(workflow["candidate_novelty_gate"])
        self.assertTrue(workflow["continue_after_improvement"])
        self.assertTrue(workflow["deterministic_test_decision"])
        self.assertFalse(workflow["structural_tactic_lifecycle"])

    def test_minimal_preset_disables_exploration_machinery(self) -> None:
        config = apply_workflow_preset({"workflow": {"preset": "minimal"}})
        workflow = config["workflow"]
        self.assertFalse(workflow["adaptive_search_memory"])
        self.assertFalse(workflow["candidate_novelty_gate"])
        self.assertFalse(workflow["continue_after_improvement"])
        self.assertTrue(workflow["deterministic_test_decision"])
        self.assertTrue(workflow["repair_target_not_found"])

    def test_structural_preset_extends_search(self) -> None:
        config = apply_workflow_preset({"workflow": {"preset": "structural"}})
        workflow = config["workflow"]
        self.assertTrue(workflow["adaptive_search_memory"])
        self.assertTrue(workflow["run_spec_after_read"])
        self.assertTrue(workflow["semantic_analysis_after_read"])
        self.assertTrue(workflow["structural_tactic_lifecycle"])
        self.assertTrue(workflow["structural_state_checkpoint"])

    def test_spec_preset_enables_spec_scheduler(self) -> None:
        config = apply_workflow_preset({"workflow": {"preset": "spec"}})
        workflow = config["workflow"]
        self.assertTrue(workflow["spec_mode"])
        self.assertTrue(workflow["run_spec_enabled"])
        self.assertFalse(workflow["run_spec_after_read"])
        self.assertEqual(workflow["spec_task_attempt_budget"], 8)
        self.assertFalse(workflow["continue_after_improvement"])
        self.assertTrue(workflow["deterministic_test_decision"])

    def test_spec_preset_owns_spec_safety_gates(self) -> None:
        config = apply_workflow_preset({"workflow": {"preset": "spec"}})
        workflow = config["workflow"]
        for key in (
            "spec_design_contract_gate",
            "spec_grounding_gate",
            "spec_quality_gate",
            "spec_structural_risk_gate",
            "spec_two_call_synthesis",
            "spec_probe_diff_contract_required",
            "probe_diff_contract_gate",
        ):
            self.assertTrue(workflow[key], key)
        self.assertEqual(workflow["spec_quality_rewrite_attempts"], 2)
        self.assertEqual(workflow["spec_synth_call_budget"], 24)
        self.assertTrue(workflow["spec_gate_soft_fallback"])
        self.assertEqual(workflow["spec_active_task_drift_streak_limit"], 3)
        self.assertEqual(workflow["spec_active_task_drift_same_fingerprint_limit"], 2)
        self.assertEqual(workflow["spec_active_task_drift_rewrite_attempts"], 1)
        self.assertEqual(workflow["spec_drift_saturation_threshold"], 3)
        self.assertEqual(workflow["spec_portfolio_recovery_rounds"], 2)

    def test_explicit_workflow_key_wins_over_preset(self) -> None:
        config = apply_workflow_preset(
            {"workflow": {"preset": "search", "max_code_test_loops": 100}}
        )
        self.assertEqual(config["workflow"]["max_code_test_loops"], 100)
        self.assertTrue(config["workflow"]["adaptive_search_memory"])

    def test_unknown_preset_raises_with_valid_names(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            apply_workflow_preset({"workflow": {"preset": "turbo"}})
        message = str(ctx.exception)
        for name in WORKFLOW_PRESETS:
            self.assertIn(name, message)

    def test_preset_does_not_mutate_original_config(self) -> None:
        original = {"workflow": {"preset": "search"}}
        apply_workflow_preset(original)
        self.assertEqual(original, {"workflow": {"preset": "search"}})

    def test_load_config_expands_preset_before_state_construction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps({"workflow": {"preset": "search"}}))
            config = load_config(config_path)
            self.assertEqual(config["workflow"]["max_code_test_loops"], 25)

    def test_preset_records_defaulted_keys_provenance(self) -> None:
        config = apply_workflow_preset(
            {"workflow": {"preset": "search", "max_code_test_loops": 100}}
        )
        defaulted = config["workflow"]["preset_defaulted_keys"]
        self.assertNotIn("max_code_test_loops", defaulted)
        self.assertIn("adaptive_search_memory", defaulted)

    def test_reexpansion_preserves_defaulted_key_provenance(self) -> None:
        once = apply_workflow_preset({"workflow": {"preset": "search"}})
        twice = apply_workflow_preset(once)
        self.assertEqual(
            once["workflow"]["preset_defaulted_keys"],
            twice["workflow"]["preset_defaulted_keys"],
        )
        self.assertIn("max_code_test_loops", twice["workflow"]["preset_defaulted_keys"])

    def test_preexpanded_config_still_syncs_state_max_loops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps({"workflow": {"preset": "search"}}))
            config = load_config(config_path)
            state = AgentState(repo_root=Path(tmp), user_request="test")
            MicroAgent({**config, "models": {}, "providers": {}, "mcp_servers": {}}, state)
            self.assertEqual(state.max_loops, 25)

    def test_micro_agent_init_syncs_state_max_loops_from_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {"preset": "search"},
            }
            MicroAgent(config, state)
            self.assertEqual(state.max_loops, 25)

    def test_caller_supplied_max_loops_survives_preset_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test", max_loops=7)
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {"preset": "search"},
            }
            MicroAgent(config, state)
            self.assertEqual(state.max_loops, 7)

    def test_caller_supplied_default_loop_count_survives_preset_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test", max_loops=3)
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {"preset": "search"},
            }
            MicroAgent(config, state)
            self.assertEqual(state.max_loops, 3)

    def test_explicit_loop_budget_leaves_state_max_loops_to_caller(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test", max_loops=7)
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {"preset": "search", "max_code_test_loops": 7},
            }
            MicroAgent(config, state)
            self.assertEqual(state.max_loops, 7)

    def test_micro_agent_init_applies_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {"preset": "search", "profile_agent": False},
            }
            agent = MicroAgent(config, state)
            self.assertTrue(agent.config["workflow"]["adaptive_search_memory"])
            self.assertFalse(agent.config["workflow"]["profile_agent"])


if __name__ == "__main__":
    unittest.main()
