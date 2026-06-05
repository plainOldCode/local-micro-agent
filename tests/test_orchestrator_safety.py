from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from local_micro_agent.orchestrator import MicroAgent
from local_micro_agent.prompts import code_prompt, reflect_prompt
from local_micro_agent.state import AgentState, AgentStateName


class _BadJsonModel:
    async def chat(self, messages):
        return "{}"


class _BadJsonModelManager:
    def get(self, role):
        return _BadJsonModel()


class _FailingModel:
    async def chat(self, messages):
        raise TimeoutError("model timed out")


class _FailingModelManager:
    def get(self, role):
        return _FailingModel()


class _StaticModel:
    def __init__(self, output: str):
        self.output = output

    async def chat(self, messages):
        return self.output


class _StaticModelManager:
    def __init__(self, output: str):
        self.output = output

    def get(self, role):
        return _StaticModel(self.output)


class _RoleModel:
    def __init__(self, outputs: dict[str, str], seen: dict[str, list[list[dict[str, str]]]], role: str):
        self.outputs = outputs
        self.seen = seen
        self.role = role

    async def chat(self, messages):
        self.seen.setdefault(self.role, []).append(messages)
        return self.outputs[self.role]


class _RoleModelManager:
    def __init__(self, outputs: dict[str, str]):
        self.outputs = outputs
        self.seen: dict[str, list[list[dict[str, str]]]] = {}

    def get(self, role):
        return _RoleModel(self.outputs, self.seen, role)


def run_agent(repo: Path, workflow: dict) -> AgentState:
    config = {
        "models": {},
        "providers": {},
        "mcp_servers": {},
        "workflow": {
            "plan_markdown": "seeded",
            "seed_files": workflow.get("writable_files", ["target.py"]),
            "deterministic_test_decision": True,
            **workflow,
        },
    }
    state = AgentState(
        repo_root=repo,
        user_request="test",
        max_loops=config.get("workflow", {}).get("max_code_test_loops", 3),
    )
    return asyncio.run(MicroAgent(config, state).run())


class OrchestratorSafetyTests(unittest.TestCase):
    def test_failed_candidate_restores_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")

            result = run_agent(
                repo,
                {
                    "writable_files": ["target.py"],
                    "seed_changes": [
                        {
                            "path": "target.py",
                            "target": "value = 'old'\n",
                            "replacement": "value = 'bad'\n",
                        }
                    ],
                    "test_commands": ["python3 -c \"raise SystemExit(1)\""],
                },
            )

            self.assertEqual(result.current, AgentStateName.FAILED)
            self.assertEqual(target.read_text(), "value = 'old'\n")

    def test_failed_candidate_deletes_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            result = run_agent(
                repo,
                {
                    "seed_files": [],
                    "writable_files": ["created.py"],
                    "seed_changes": [{"path": "created.py", "content": "bad = True\n"}],
                    "test_commands": ["python3 -c \"raise SystemExit(1)\""],
                },
            )

            self.assertEqual(result.current, AgentStateName.FAILED)
            self.assertFalse((repo / "created.py").exists())

    def test_metric_rejects_non_improvement_and_restores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")

            result = run_agent(
                repo,
                {
                    "writable_files": ["target.py"],
                    "seed_changes": [
                        {
                            "path": "target.py",
                            "target": "value = 'old'\n",
                            "replacement": "value = 'slower'\n",
                        }
                    ],
                    "test_commands": ["python3 -c \"print('cycles: 200')\""],
                    "metric_regex": r"cycles: (\d+)",
                    "baseline_metric": 100,
                    "accept_if_improved": True,
                },
            )

            self.assertEqual(result.current, AgentStateName.FAILED)
            self.assertEqual(target.read_text(), "value = 'old'\n")

    def test_noop_replacement_is_not_counted_as_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")

            result = run_agent(
                repo,
                {
                    "writable_files": ["target.py"],
                    "seed_changes": [
                        {
                            "path": "target.py",
                            "target": "value = 'old'\n",
                            "replacement": "value = 'old'\n",
                        }
                    ],
                    "test_commands": ["python3 -c \"print('ok')\""],
                },
            )

            self.assertEqual(result.current, AgentStateName.FAILED)
            self.assertEqual(result.scratch["applied_changes"], 0)
            self.assertIn("Replacement is a no-op", "\n".join(result.notes))

    def test_comment_only_replacement_is_not_counted_as_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")

            result = run_agent(
                repo,
                {
                    "writable_files": ["target.py"],
                    "seed_changes": [
                        {
                            "path": "target.py",
                            "target": "value = 'old'\n",
                            "replacement": "# explanatory comment\nvalue = 'old'\n",
                        }
                    ],
                    "test_commands": ["python3 -c \"print('ok')\""],
                },
            )

            self.assertEqual(result.current, AgentStateName.FAILED)
            self.assertEqual(result.scratch["applied_changes"], 0)
            self.assertIn("only changes comments", "\n".join(result.notes))

    def test_plan_reads_readme_as_project_context_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "AGENTS.md").write_text("Only modify target.py.\n")
            (repo / "Readme.md").write_text("Do not change tests. Read target.py.\n")
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            config = {
                "models": {"default": "roles"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["target.py"],
                    "test_commands": ["python3 -c \"print('ok')\""],
                    "deterministic_test_decision": True,
                },
            }
            state = AgentState(repo_root=repo, user_request="test", max_loops=1)
            agent = MicroAgent(config, state)
            models = _RoleModelManager(
                {
                    "planner": "plan",
                    "coder": (
                        '{"changes":[{"path":"target.py","target":"value = '
                        "'old'\\n\",\"replacement\":\"value = 'new'\\n\"}]}"
                    ),
                    "tester": '{"status":"pass"}',
                }
            )
            # The planner is called once for PLAN and once for READ.
            models.outputs["planner"] = "plan"
            agent.models = models

            async def plan_only():
                await agent.mcp.start()
                try:
                    await agent.plan()
                    self.assertIn("AGENTS.md", models.seen["planner"][0][1]["content"])
                    self.assertIn("Only modify target.py", models.seen["planner"][0][1]["content"])
                    self.assertIn("README", models.seen["planner"][0][1]["content"])
                    self.assertIn("Do not change tests", models.seen["planner"][0][1]["content"])
                    self.assertIn("Workflow constraints", models.seen["planner"][0][1]["content"])
                    self.assertIn("target.py", models.seen["planner"][0][1]["content"])
                finally:
                    await agent.mcp.close()

            asyncio.run(plan_only())

    def test_code_prompt_carries_recent_agent_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            state.plan_markdown = "plan"
            state.notes.extend(
                [
                    "Replacement target not found: target.py",
                    "Replacement only changes comments or blank lines: target.py",
                ]
            )

            messages = code_prompt(state)

            self.assertIn("Recent agent feedback", messages[1]["content"])
            self.assertIn("target not found", messages[1]["content"])
            self.assertIn("only changes comments", messages[1]["content"])

    def test_reflect_prompt_carries_retry_context_without_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            state.plan_markdown = "plan"
            state.notes.append("Candidate 1 rejected: no changes applied")

            messages = reflect_prompt(state)

            self.assertIn("Do not write code", messages[0]["content"])
            self.assertIn("Candidate 1 rejected", messages[1]["content"])
            self.assertNotIn("Source files", messages[1]["content"])

    def test_code_prompt_includes_retry_reflection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            state.plan_markdown = "plan"
            state.scratch["reflection"] = "- Previous attempt produced invalid JSON."

            messages = code_prompt(state)

            self.assertIn("Retry reflection", messages[1]["content"])
            self.assertIn("invalid JSON", messages[1]["content"])

    def test_reflect_state_stores_summary_for_next_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(
                repo_root=Path(tmp),
                user_request="test",
                current=AgentStateName.REFLECT,
            )
            config = {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}}
            agent = MicroAgent(config, state)
            agent.models = _StaticModelManager("- Failed because JSON was invalid.")

            asyncio.run(agent.reflect())

            self.assertEqual(state.current, AgentStateName.CODE)
            self.assertIn("JSON was invalid", state.scratch["reflection"])
            self.assertIn("Reflect summary added", "\n".join(state.notes))

    def test_metric_accepts_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")

            result = run_agent(
                repo,
                {
                    "writable_files": ["target.py"],
                    "seed_changes": [
                        {
                            "path": "target.py",
                            "target": "value = 'old'\n",
                            "replacement": "value = 'faster'\n",
                        }
                    ],
                    "test_commands": ["python3 -c \"print('cycles: 80')\""],
                    "metric_regex": r"cycles: (\d+)",
                    "baseline_metric": 100,
                    "accept_if_improved": True,
                },
            )

            self.assertEqual(result.current, AgentStateName.DONE)
            self.assertEqual(target.read_text(), "value = 'faster'\n")

    def test_deterministic_mode_can_retry_rejected_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")

            result = run_agent(
                repo,
                {
                    "max_code_test_loops": 2,
                    "retry_rejected_candidates": True,
                    "writable_files": ["target.py"],
                    "seed_changes": [
                        {
                            "path": "target.py",
                            "target": "value = 'old'\n",
                            "replacement": "value = 'bad'\n",
                        }
                    ],
                    "test_commands": ["python3 -c \"raise SystemExit(1)\""],
                },
            )

            self.assertEqual(result.current, AgentStateName.FAILED)
            self.assertEqual(result.loop_count, 1)
            self.assertEqual(target.read_text(), "value = 'old'\n")

    def test_bad_coder_json_is_rejected_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            config = {
                "models": {"default": "bad"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "plan_markdown": "seeded",
                    "seed_files": [],
                    "writable_files": ["target.py"],
                    "test_commands": ["python3 -c \"print('ok')\""],
                    "deterministic_test_decision": True,
                    "retry_rejected_candidates": True,
                    "max_code_test_loops": 2,
                },
            }
            state = AgentState(repo_root=repo, user_request="test", max_loops=2)
            agent = MicroAgent(config, state)
            agent.models = _BadJsonModelManager()

            result = asyncio.run(agent.run())

            self.assertEqual(result.current, AgentStateName.FAILED)
            self.assertEqual(result.loop_count, 1)
            self.assertIn("Coder output rejected after repair", "\n".join(result.notes))

    def test_coder_transport_error_is_rejected_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            config = {
                "models": {"default": "failing"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "plan_markdown": "seeded",
                    "seed_files": [],
                    "writable_files": ["target.py"],
                    "test_commands": ["python3 -c \"print('ok')\""],
                    "deterministic_test_decision": True,
                    "retry_rejected_candidates": True,
                    "max_code_test_loops": 2,
                },
            }
            state = AgentState(repo_root=repo, user_request="test", max_loops=2)
            agent = MicroAgent(config, state)
            agent.models = _FailingModelManager()

            result = asyncio.run(agent.run())

            self.assertEqual(result.current, AgentStateName.FAILED)
            self.assertEqual(result.loop_count, 1)
            self.assertIn("coder model call failed", "\n".join(result.notes))

    def test_candidate_queue_accepts_best_candidate_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            evaluator = (
                "python3 -c \"from pathlib import Path; "
                "t=Path('target.py').read_text(); "
                "print('cycles: 80' if 'fast' in t else 'cycles: 120')\""
            )
            output = """
{
  "candidates": [
    {
      "id": "slow",
      "changes": [
        {"path": "target.py", "target": "value = 'old'\\n", "replacement": "value = 'slow'\\n"}
      ]
    },
    {
      "id": "fast",
      "changes": [
        {"path": "target.py", "target": "value = 'old'\\n", "replacement": "value = 'fast'\\n"}
      ]
    }
  ]
}
"""
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "plan_markdown": "seeded",
                    "seed_files": ["target.py"],
                    "writable_files": ["target.py"],
                    "test_commands": [evaluator],
                    "deterministic_test_decision": True,
                    "candidate_queue": True,
                    "metric_regex": r"cycles: (\d+)",
                    "baseline_metric": 100,
                    "accept_if_improved": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                },
            }
            state = AgentState(repo_root=repo, user_request="test", max_loops=1)
            agent = MicroAgent(config, state)
            agent.models = _StaticModelManager(output)

            result = asyncio.run(agent.run())

            self.assertEqual(result.current, AgentStateName.DONE)
            self.assertEqual(target.read_text(), "value = 'fast'\n")
            self.assertEqual(result.scratch["last_metric"], 80)
            self.assertIn("Candidate slow", "\n".join(result.notes))
            self.assertIn("Candidate queue accepted metric=80", "\n".join(result.notes))
            history_path = repo / ".local_micro_agent" / "candidates.jsonl"
            history = history_path.read_text()
            self.assertIn('"candidate_id": "slow"', history)
            self.assertIn('"candidate_id": "fast"', history)
            self.assertIn('"status": "accepted"', history)

    def test_context_symbols_limit_code_prompt_to_requested_python_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text(
                "def keep_me():\n"
                "    return 'old'\n\n"
                "def hide_me():\n"
                "    return 'secret'\n"
            )
            output = """
{
  "changes": [
    {"path": "target.py", "target": "return 'old'", "replacement": "return 'new'"}
  ]
}
"""
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "plan_markdown": "seeded",
                    "seed_files": ["target.py"],
                    "writable_files": ["target.py"],
                    "context_symbols": {"target.py": ["keep_me"]},
                    "test_commands": ["python3 -c \"print('ok')\""],
                    "deterministic_test_decision": True,
                },
            }
            state = AgentState(repo_root=repo, user_request="test", max_loops=1)
            agent = MicroAgent(config, state)
            agent.models = _StaticModelManager(output)

            result = asyncio.run(agent.run())

            self.assertEqual(result.current, AgentStateName.DONE)
            self.assertIn("def keep_me", result.file_context[0].content)
            self.assertNotIn("hide_me", result.file_context[0].content)
            self.assertIn("Using symbol context", "\n".join(result.notes))


if __name__ == "__main__":
    unittest.main()
