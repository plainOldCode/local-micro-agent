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
