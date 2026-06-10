from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ConfigFileTests(unittest.TestCase):
    def test_a3b_mxfp8_config_keeps_tuned_lane_split(self) -> None:
        config = json.loads(
            (ROOT / "config/config.qwen36-35b-a3b-coding-mxfp8-ollama.json").read_text()
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

        self.assertEqual(models["plan_deep"], "qwen36_a3b_plan_deep")
        self.assertEqual(
            workflow["model_role_overrides_by_call_site"]["plan"],
            "plan_deep",
        )
        plan_deep = providers["qwen36_a3b_plan_deep"]
        self.assertTrue(plan_deep["think"])
        self.assertEqual(plan_deep["num_ctx"], 64000)
        self.assertEqual(plan_deep["max_tokens"], 16384)
        self.assertEqual(plan_deep["temperature"], 0.6)
        self.assertEqual(plan_deep["timeout_seconds"], 900)
        self.assertEqual(plan_deep["extra_options"]["top_p"], 0.95)
        self.assertEqual(plan_deep["extra_options"]["top_k"], 20)
        self.assertEqual(plan_deep["extra_options"]["min_p"], 0)

        self.assertEqual(models["spec_synth"], "qwen36_a3b_spec_synth")
        self.assertEqual(
            workflow["model_role_overrides_by_call_site"]["spec_synth"],
            "spec_synth",
        )
        spec_synth = providers["qwen36_a3b_spec_synth"]
        self.assertTrue(spec_synth["think"])
        self.assertEqual(spec_synth["num_ctx"], 64000)
        self.assertEqual(spec_synth["max_tokens"], 16384)
        self.assertEqual(spec_synth["temperature"], 0.6)
        self.assertEqual(spec_synth["extra_options"]["top_p"], 0.95)
        self.assertEqual(spec_synth["extra_options"]["top_k"], 20)
        self.assertEqual(spec_synth["extra_options"]["min_p"], 0)


if __name__ == "__main__":
    unittest.main()
