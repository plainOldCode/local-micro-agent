from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from local_micro_agent.orchestrator import CodeCandidate, MicroAgent
from local_micro_agent.prompts import code_prompt, reflect_prompt
from local_micro_agent.state import AgentState, AgentStateName, CodeChange


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

    def test_code_prompt_can_request_xml_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            state.plan_markdown = "plan"

            messages = code_prompt(state, output_format="xml")

            self.assertIn("Do not output JSON", messages[0]["content"])
            self.assertIn("<search>", messages[0]["content"])
            self.assertIn("<replace>", messages[0]["content"])

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

    def test_candidate_novelty_gate_rejects_repeated_failed_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["target.py"],
                    "test_commands": ["python3 -c \"print('cycles: 120')\""],
                    "candidate_queue": True,
                    "candidate_novelty_gate": True,
                    "metric_regex": r"cycles: (\d+)",
                    "baseline_metric": 100,
                    "accept_if_improved": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            agent = MicroAgent(config, state)
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            candidate = CodeCandidate(
                "repeat",
                [
                    CodeChange(
                        path="target.py",
                        reason="same failed idea",
                        target="value = 'old'\n",
                        replacement="value = 'slow'\n",
                    )
                ],
                "same failed idea",
            )

            async def evaluate_once() -> None:
                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([candidate], {"target.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(evaluate_once())
            first_notes = "\n".join(state.notes)
            self.assertIn("Candidate repeat applied=1 metric=120", first_notes)
            self.assertEqual(target.read_text(), "value = 'old'\n")

            asyncio.run(evaluate_once())
            second_notes = "\n".join(state.notes)
            self.assertIn("forbidden repeated pattern", second_notes)
            history = (repo / ".local_micro_agent" / "candidates.jsonl").read_text()
            self.assertIn('"status": "rejected_repeated_pattern"', history)

    def test_adaptive_search_memory_cools_down_repeated_failed_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["target.py"],
                    "test_commands": ["python3 -c \"print('cycles: 120')\""],
                    "candidate_queue": True,
                    "adaptive_search_memory": True,
                    "adaptive_search_axis_failure_threshold": 3,
                    "adaptive_search_axis_cooldown_loops": 4,
                    "metric_regex": r"cycles: (\d+)",
                    "baseline_metric": 100,
                    "accept_if_improved": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            agent = MicroAgent(config, state)
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            candidate = CodeCandidate(
                "phase",
                [
                    CodeChange(
                        path="target.py",
                        reason="phase interleave tweak",
                        target="value = 'old'\n",
                        replacement="value = 'slow'\n",
                    )
                ],
                "try phase interleave scheduling",
            )

            async def evaluate_three_times() -> None:
                await agent.mcp.start()
                try:
                    for _ in range(3):
                        await agent._evaluate_code_candidates([candidate], {"target.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(evaluate_three_times())

            memory_text = agent._format_adaptive_search_memory()
            self.assertIn("phase_interleave", memory_text)
            self.assertIn('"cooled_down_axes": [\n    "phase_interleave"\n  ]', memory_text)
            history = (repo / ".local_micro_agent" / "candidates.jsonl").read_text()
            self.assertIn('"strategy_axes": ["phase_interleave"]', history)

    def test_code_prompt_includes_adaptive_search_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            config = {
                "models": {"default": "roles"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "plan_markdown": "seeded",
                    "seed_files": ["target.py"],
                    "writable_files": ["target.py"],
                    "test_commands": ["python3 -c \"print('ok')\""],
                    "deterministic_test_decision": True,
                    "adaptive_search_memory": True,
                },
            }
            state = AgentState(
                repo_root=repo,
                user_request="test",
                current=AgentStateName.CODE,
                max_loops=1,
            )
            state.plan_markdown = "seeded"
            state.planned_files = ["target.py"]
            state.file_context = []
            state.scratch["adaptive_search_memory"] = {
                "axes": {
                    "phase_interleave": {
                        "attempts": 3,
                        "failures": 3,
                        "successes": 0,
                        "cooldown_until_loop": 4,
                        "last_status": "rejected",
                        "last_metric": 120,
                        "best_metric": None,
                    }
                },
                "recent": [],
            }
            models = _RoleModelManager(
                {
                    "coder": (
                        '{"changes":[{"path":"target.py","target":"value = '
                        "'old'\\n\",\"replacement\":\"value = 'new'\\n\"}]}"
                    )
                }
            )
            agent = MicroAgent(config, state)
            agent.models = models

            async def code_once() -> None:
                await agent.mcp.start()
                try:
                    await agent.code()
                finally:
                    await agent.mcp.close()

            asyncio.run(code_once())

            coder_messages = models.seen["coder"][0]
            joined = "\n".join(message["content"] for message in coder_messages)
            self.assertIn("Adaptive search memory follows", joined)
            self.assertIn("phase_interleave", joined)
            self.assertIn("cooled_down_axes", joined)

    def test_adaptive_search_can_reject_cooled_axis_before_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["target.py"],
                    "test_commands": ["python3 -c \"print('cycles: 80')\""],
                    "candidate_queue": True,
                    "adaptive_search_memory": True,
                    "adaptive_search_reject_cooled_axes": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            state.scratch["adaptive_search_memory"] = {
                "axes": {
                    "memory_store_layout": {
                        "attempts": 3,
                        "failures": 3,
                        "successes": 0,
                        "cooldown_until_loop": 4,
                        "last_status": "rejected",
                        "last_metric": 120,
                        "best_metric": None,
                    }
                },
                "recent": [],
            }
            candidate = CodeCandidate(
                "cooled",
                [
                    CodeChange(
                        path="target.py",
                        reason="store layout retry",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "retry memory store layout",
            )
            agent = MicroAgent(config, state)

            async def evaluate_once() -> None:
                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([candidate], {"target.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(evaluate_once())

            self.assertEqual(target.read_text(), "value = 'old'\n")
            self.assertIn("cooled strategy axes memory_store_layout", "\n".join(state.notes))
            history = (repo / ".local_micro_agent" / "candidates.jsonl").read_text()
            self.assertIn('"status": "rejected_cooled_axis"', history)

    def test_continue_after_improvement_persists_best_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'fast'\n")
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["target.py"],
                    "continue_after_improvement": True,
                },
            }
            state = AgentState(repo_root=repo, user_request="test", max_loops=2)
            state.loop_count = 0
            state.scratch["metric_improved"] = True
            state.scratch["best_metric"] = 80
            state.scratch["last_metric"] = 80
            state.scratch["pre_code_snapshot"] = {"target.py": "value = 'old'\n"}
            state.proposed_changes = [
                CodeChange(
                    path="target.py",
                    reason="faster",
                    target="value = 'old'\n",
                    replacement="value = 'fast'\n",
                )
            ]
            agent = MicroAgent(config, state)

            async def persist() -> None:
                await agent.mcp.start()
                try:
                    await agent._persist_current_best_state()
                finally:
                    await agent.mcp.close()

            asyncio.run(persist())

            best_state = repo / ".local_micro_agent" / "best_state.json"
            best_patch = repo / ".local_micro_agent" / "best.patch"
            self.assertTrue(agent._should_continue_after_improvement())
            self.assertIn('"metric": 80', best_state.read_text())
            self.assertIn("-value = 'old'", best_patch.read_text())
            self.assertIn("+value = 'fast'", best_patch.read_text())

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
