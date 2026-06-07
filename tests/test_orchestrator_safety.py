from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from local_micro_agent.orchestrator import CodeCandidate, MicroAgent
from local_micro_agent.prompts import code_prompt, reflect_prompt
from local_micro_agent.state import AgentState, AgentStateName, CodeChange, FileSnapshot


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

    def test_reflect_brainstorms_after_rejection_streak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            history_dir = repo / ".local_micro_agent"
            history_dir.mkdir()
            history = history_dir / "candidates.jsonl"
            history.write_text(
                "\n".join(
                    [
                        '{"status":"rejected_axis_drift","strategy_axis":"general_edit","strategy_axes":["phase_interleave"],"reason":"phase retry"}',
                        '{"status":"rejected_cooled_axis","strategy_axis":"precompute_constants","strategy_axes":["precompute_constants"],"reason":"constant retry"}',
                    ]
                )
                + "\n"
            )
            (history_dir / "todo_plan.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "active_todo_id": None,
                        "todos": [
                            {
                                "todo_id": "todo-007-hash_build",
                                "status": "failed",
                                "strategy_axis": "hash_build",
                                "context": "failed hash tactic",
                                "attempts": 1,
                                "last_attempt": {
                                    "status": "rejected",
                                    "metric": 38423,
                                    "reason": "hash got slower",
                                },
                            }
                        ],
                    }
                )
                + "\n"
            )
            config = {
                "models": {"default": "roles"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "brainstorm_after_rejections": 2,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                },
            }
            state = AgentState(repo_root=repo, user_request="test", current=AgentStateName.REFLECT)
            state.plan_markdown = "seeded"
            state.file_context = [FileSnapshot("target.py", target.read_text())]
            models = _RoleModelManager(
                {
                    "brainstorm": (
                        "1. **Strategy Axis:** `hash_build` new_axis_suggestion: new tactic\n"
                        "Try a different hash build tactic.\n"
                        "2. strategy_axis: phase_interleave\nother\n"
                        "3. strategy_axis: branch_control\nthird"
                    )
                }
            )
            agent = MicroAgent(config, state)
            agent.models = models

            asyncio.run(agent.reflect())

            self.assertEqual(state.current, AgentStateName.CODE)
            self.assertIn("new tactic", state.scratch["tactic_library"])
            self.assertEqual(state.scratch["selected_tactic"]["strategy_axis"], "hash_build")
            joined = "\n".join(message["content"] for message in models.seen["brainstorm"][0])
            self.assertIn("Recent reject summary", joined)
            self.assertIn("Known strategy axes", joined)
            self.assertIn("Durable todo ledger summary", joined)
            self.assertIn("hash_build", joined)
            self.assertIn("phase retry", joined)
            self.assertIn("hash got slower", joined)
            tactics = repo / ".local_micro_agent" / "brainstorm_tactics.md"
            self.assertIn("new tactic", tactics.read_text())
            self.assertIn("phase retry", tactics.read_text())
            active_todo = repo / ".local_micro_agent" / "active_todo.json"
            todo_plan = repo / ".local_micro_agent" / "todo_plan.json"
            self.assertIn("hash_build", active_todo.read_text())
            self.assertIn("todo-000-hash_build", todo_plan.read_text())

    def test_all_skipped_streak_requires_new_brainstorm_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            history_dir = repo / ".local_micro_agent"
            history_dir.mkdir()
            (history_dir / "candidates.jsonl").write_text(
                '{"status":"rejected_brainstorm_all_failed_families"}\n'
                '{"status":"rejected_brainstorm_all_failed_families"}\n'
            )
            (history_dir / "brainstorm_selection.jsonl").write_text(
                json.dumps(
                    {
                        "all_skipped": True,
                        "records": [
                            {
                                "skipped": True,
                                "family_aliases": ["hash_reorder", "hash_build"],
                            }
                        ],
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "all_skipped": True,
                        "records": [
                            {
                                "skipped": True,
                                "family_aliases": ["store_address_reuse"],
                            }
                        ],
                    }
                )
                + "\n"
            )
            (history_dir / "failed_tactics.jsonl").write_text(
                json.dumps(
                    {
                        "strategy_axis": "memory_store_layout",
                        "family_key": "store_address_reuse",
                        "context": "failed store address reuse",
                    }
                )
                + "\n"
            )
            config = {
                "models": {"default": "roles"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "brainstorm_after_rejections": 2,
                    "brainstorm_new_family_after_all_skipped": 2,
                    "brainstorm_open_novelty_lanes": [
                        "layout_or_tiling_change: try a small layout probe",
                        "control_or_guard_lowering: try a guard lowering probe",
                    ],
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                },
            }
            state = AgentState(repo_root=repo, user_request="test", current=AgentStateName.REFLECT)
            state.plan_markdown = "seeded"
            state.file_context = [FileSnapshot("target.py", target.read_text())]
            models = _RoleModelManager(
                {
                    "brainstorm": (
                        "1. strategy_axis: memory_store_layout\n"
                        "novelty_lane: layout_or_tiling_change\n"
                        "Try a new family under the same execution axis.\n"
                        "family_key: tree_shape_specialization\n"
                    )
                }
            )
            agent = MicroAgent(config, state)
            agent.models = models

            asyncio.run(agent.reflect())

            joined = "\n".join(message["content"] for message in models.seen["brainstorm"][0])
            self.assertIn("New family required:\ntrue", joined)
            self.assertIn("store_address_reuse", joined)
            self.assertIn("hash_reorder", joined)
            self.assertIn("Open novelty lanes", joined)
            self.assertIn("layout_or_tiling_change", joined)
            self.assertEqual(state.scratch["selected_tactic"]["family_key"], "tree_shape_specialization")
            self.assertEqual(state.scratch["selected_tactic"]["novelty_lane"], "layout_or_tiling_change")

    def test_brainstorm_includes_open_novelty_lanes_before_all_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            history_dir = repo / ".local_micro_agent"
            history_dir.mkdir()
            (history_dir / "candidates.jsonl").write_text(
                '{"status":"rejected","strategy_axis":"general_edit","reason":"same baseline"}\n'
                '{"status":"rejected","strategy_axis":"general_edit","reason":"same baseline"}\n'
            )
            config = {
                "models": {"default": "roles"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "brainstorm_after_rejections": 2,
                    "brainstorm_new_family_after_all_skipped": 0,
                    "brainstorm_open_novelty_lanes": [
                        "coarse_unroll_lane_restructure: try a small unroll probe",
                        "load_latency_scheduling: move one load-use boundary",
                    ],
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                },
            }
            state = AgentState(repo_root=repo, user_request="test", current=AgentStateName.REFLECT)
            state.plan_markdown = "seeded"
            state.file_context = [FileSnapshot("target.py", target.read_text())]
            models = _RoleModelManager(
                {
                    "brainstorm": (
                        "1. strategy_axis: vector_unroll_lane\n"
                        "novelty_lane: coarse_unroll_lane_restructure\n"
                        "Try a small unroll probe.\n"
                        "family_key: unroll_factor_change\n"
                    )
                }
            )
            agent = MicroAgent(config, state)
            agent.models = models

            asyncio.run(agent.reflect())

            joined = "\n".join(message["content"] for message in models.seen["brainstorm"][0])
            self.assertIn("New family required:\nfalse", joined)
            self.assertIn("Open novelty lanes", joined)
            self.assertIn("coarse_unroll_lane_restructure", joined)
            self.assertEqual(
                state.scratch["selected_tactic"]["novelty_lane"],
                "coarse_unroll_lane_restructure",
            )

    def test_selected_brainstorm_tactic_sets_required_axis_for_current_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "adaptive_search_memory": True,
                    "adaptive_search_force_strategy_axis": True,
                    "adaptive_search_axis_pool": ["vector_unroll_lane", "hash_build"],
                },
            }
            state.scratch["selected_tactic"] = {
                "strategy_axis": "hash_build",
                "text": "1. strategy_axis: hash_build",
            }
            state.scratch["selected_tactic_loop"] = 0
            state.scratch["adaptive_search_memory"] = {
                "axes": {
                    "hash_build": {
                        "attempts": 3,
                        "failures": 3,
                        "successes": 0,
                        "cooldown_until_loop": 4,
                    },
                    "phase_interleave": {
                        "attempts": 3,
                        "failures": 3,
                        "successes": 0,
                        "cooldown_until_loop": 4,
                    }
                },
                "recent": [],
            }
            agent = MicroAgent(config, state)

            contract = agent._format_axis_contract()

            self.assertIn('"required_strategy_axis": "hash_build"', contract)
            self.assertIn('"selected_tactic"', contract)

            state.loop_count = 1
            next_contract = agent._format_axis_contract()
            self.assertIn('"required_strategy_axis": "vector_unroll_lane"', next_contract)

    def test_failed_tactic_signature_skips_similar_brainstorm_tactic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "failed_tactics.jsonl").write_text(
                json.dumps(
                    {
                        "context": (
                            "Replace scalar ALU hash stages with VALU vector instructions "
                            "to process lanes per cycle."
                        ),
                        "last_attempt": {
                            "reason": (
                                "Replace scalar ALU hash stages with VALU vector instructions "
                                "to reduce instruction count."
                            )
                        },
                    }
                )
                + "\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {"failed_tactic_similarity_threshold": 0.35},
                },
                AgentState(repo_root=repo, user_request="test"),
            )

            selected = agent._select_brainstorm_tactic(
                "1. strategy_axis: vector_unroll_lane\n"
                "Replace scalar ALU hash stages with VALU vector instructions.\n"
                "2. strategy_axis: branch_control\n"
                "Replace a bounds multiply with a bitwise mask.\n"
            )

            self.assertIsNotNone(selected)
            self.assertEqual(selected["strategy_axis"], "branch_control")
            self.assertIn("Skipped brainstorm tactic", "\n".join(agent.state.notes))

    def test_failed_tactic_family_falls_back_to_strategy_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "failed_tactics.jsonl").write_text(
                json.dumps(
                    {
                        "context": "Old failed tactic with no family key.",
                        "strategy_axis": "branch_control",
                        "last_attempt": {
                            "strategy_axis": "branch_control",
                            "reason": "Use a compare result in a different branch-control shape.",
                        },
                    }
                )
                + "\n"
            )
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=repo, user_request="test"),
            )

            selected = agent._select_brainstorm_tactic(
                "1. strategy_axis: branch_control\n"
                "Use a local comparison shape without declaring a family.\n"
                "2. strategy_axis: hash_build\n"
                "Try a different hash emission shape.\n"
                "family_key: hash_reorder\n"
            )

            self.assertIsNotNone(selected)
            self.assertEqual(selected["strategy_axis"], "hash_build")
            self.assertIn("failed_family=branch_control", "\n".join(agent.state.notes))

    def test_new_tactic_family_can_reuse_failed_strategy_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "failed_tactics.jsonl").write_text(
                json.dumps(
                    {
                        "context": "Old failed tactic with no family key.",
                        "strategy_axis": "branch_control",
                        "last_attempt": {
                            "strategy_axis": "branch_control",
                            "reason": "Use a compare result in a different branch-control shape.",
                        },
                    }
                )
                + "\n"
            )
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=repo, user_request="test"),
            )

            selected = agent._select_brainstorm_tactic(
                "1. strategy_axis: branch_control\n"
                "Use a genuinely different idea family under the same execution axis.\n"
                "family_key: tree_depth_specialization\n"
            )

            self.assertIsNotNone(selected)
            self.assertEqual(selected["strategy_axis"], "branch_control")
            self.assertEqual(selected["family_key"], "tree_depth_specialization")

    def test_adaptive_gate_shadows_under_evidenced_failed_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "failed_tactics.jsonl").write_text(
                json.dumps(
                    {
                        "context": "One failed branch mask tactic.",
                        "strategy_axis": "branch_control",
                        "family_key": "branch_mask",
                        "status": "failed",
                        "attempts": 1,
                    }
                )
                + "\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "adaptive_search_memory": True,
                        "adaptive_gate_controller": True,
                        "adaptive_gate_min_family_attempts_for_hard": 2,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )

            selected = agent._select_brainstorm_tactic(
                "1. strategy_axis: branch_control\n"
                "Use another branch mask variant with a narrower local guard.\n"
                "family_key: branch_mask\n"
                "2. strategy_axis: hash_build\n"
                "Try hash reorder.\n"
                "family_key: hash_reorder\n"
            )

            self.assertIsNotNone(selected)
            self.assertEqual(selected["strategy_axis"], "branch_control")
            self.assertIn("Adaptive gate allowed brainstorm tactic", "\n".join(agent.state.notes))
            decisions = (artifact_dir / "gate_decisions.jsonl").read_text()
            self.assertIn('"mode": "shadow"', decisions)
            self.assertIn("insufficient_failed_family_evidence", decisions)

    def test_adaptive_gate_reopens_families_under_all_skipped_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "brainstorm_selection.jsonl").write_text(
                json.dumps({"all_skipped": True, "records": []}) + "\n"
                + json.dumps({"all_skipped": True, "records": []}) + "\n"
            )
            (artifact_dir / "failed_tactics.jsonl").write_text(
                json.dumps(
                    {
                        "context": "Repeated failed store address reuse.",
                        "strategy_axis": "memory_store_layout",
                        "family_key": "store_address_reuse",
                        "status": "failed",
                        "attempts": 4,
                    }
                )
                + "\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "adaptive_search_memory": True,
                        "adaptive_gate_controller": True,
                        "adaptive_gate_min_family_attempts_for_hard": 1,
                        "adaptive_gate_all_skipped_relax_streak": 2,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )

            selected = agent._select_brainstorm_tactic(
                "1. strategy_axis: memory_store_layout\n"
                "Try a store address reuse variant after search dried up.\n"
                "family_key: store_address_reuse\n"
            )

            self.assertIsNotNone(selected)
            self.assertEqual(selected["strategy_axis"], "memory_store_layout")
            decisions = (artifact_dir / "gate_decisions.jsonl").read_text()
            self.assertIn('"mode": "soft"', decisions)
            self.assertIn("opportunity_pressure_all_skipped", decisions)

    def test_all_failed_brainstorm_tactics_are_persisted_and_skip_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "failed_tactics.jsonl").write_text(
                json.dumps(
                    {
                        "context": "Failed branch tactic",
                        "strategy_axis": "branch_control",
                        "family_key": "branch_mask",
                        "status": "failed",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "context": "Failed memory tactic",
                        "strategy_axis": "memory_store_layout",
                        "family_key": "store_address_reuse",
                        "status": "failed",
                    }
                )
                + "\n"
            )
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.current = AgentStateName.CODE
            state.max_loops = 3
            agent = MicroAgent(config, state)

            selected = agent._select_brainstorm_tactic(
                "1. strategy_axis: branch_control\n"
                "Replace multiply with a mask.\n"
                "family_key: branch_mask\n"
                "2. strategy_axis: memory_store_layout\n"
                "Reuse store addresses.\n"
                "family_key: store_address_reuse\n"
            )
            agent._persist_brainstorm_selection()
            asyncio.run(agent.code())

            self.assertIsNone(selected)
            self.assertEqual(state.loop_count, 1)
            self.assertEqual(state.current, AgentStateName.REFLECT)
            selection = json.loads(
                (artifact_dir / "brainstorm_selection.jsonl").read_text().splitlines()[-1]
            )
            self.assertTrue(selection["all_skipped"])
            history = (artifact_dir / "candidates.jsonl").read_text()
            self.assertIn("rejected_brainstorm_all_failed_families", history)

    def test_selected_brainstorm_tactic_overrides_cooldown_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "adaptive_search_memory": True,
                    "adaptive_search_reject_cooled_axes": True,
                    "adaptive_search_force_strategy_axis": True,
                    "adaptive_search_axis_pool": ["hash_build", "phase_interleave"],
                },
            }
            state.scratch["selected_tactic"] = {
                "strategy_axis": "hash_build",
                "text": "1. strategy_axis: hash_build",
            }
            state.scratch["selected_tactic_loop"] = 0
            state.scratch["required_strategy_axis"] = "hash_build"
            state.scratch["adaptive_search_memory"] = {
                "axes": {
                    "hash_build": {
                        "attempts": 3,
                        "failures": 3,
                        "successes": 0,
                        "cooldown_until_loop": 4,
                    }
                },
                "recent": [],
            }
            agent = MicroAgent(config, state)
            candidate = CodeCandidate(
                "selected",
                [
                    CodeChange(
                        "target.py",
                        "hash build tactic with phase overlap",
                        target="old",
                        replacement="new",
                    )
                ],
                "hash build tactic with phase overlap",
                strategy_axis="hash_build",
            )

            self.assertIsNone(agent._candidate_axis_contract_rejection(candidate))
            self.assertEqual(agent._cooled_candidate_axes(candidate), [])

    def test_code_prompt_includes_tactic_library(self) -> None:
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
            state.scratch["tactic_library"] = "1. Try a tabula rasa data layout tactic."
            state.scratch["active_todo"] = {
                "todo_id": "todo-001",
                "status": "active",
                "strategy_axis": "hash_build",
                "micro_goal": "probe one operation",
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

            joined = "\n".join(message["content"] for message in models.seen["coder"][0])
            self.assertIn("Active durable todo follows", joined)
            self.assertIn("todo-001", joined)
            self.assertIn("Stagnation brainstorm tactics follow", joined)
            self.assertIn("tabula rasa data layout", joined)

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

    def test_strategy_axes_prefer_reason_over_code_body_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                state,
            )
            candidate = CodeCandidate(
                "phase",
                [
                    CodeChange(
                        path="perf_takehome.py",
                        reason="interleave phase index updates",
                        target='self.scratch["ptr"] = old_store\n',
                        replacement='self.scratch["ptr"] = new_store\n',
                    )
                ],
                "interleave phase index updates",
            )

            self.assertEqual(agent._candidate_strategy_axes(candidate), ["phase_interleave"])

    def test_axis_contract_rejects_candidate_with_wrong_declared_axis(self) -> None:
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
                    "adaptive_search_force_strategy_axis": True,
                    "adaptive_search_axis_pool": ["hash_build", "phase_interleave"],
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            state.scratch["required_strategy_axis"] = "hash_build"
            candidate = CodeCandidate(
                "wrong-axis",
                [
                    CodeChange(
                        path="target.py",
                        reason="phase interleave retry",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "phase interleave retry",
                strategy_axis="phase_interleave",
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
            self.assertIn("does not match required hash_build", "\n".join(state.notes))
            history = (repo / ".local_micro_agent" / "candidates.jsonl").read_text()
            self.assertIn('"status": "rejected_wrong_axis"', history)

    def test_axis_contract_rejects_reason_drift_from_required_axis(self) -> None:
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
                    "adaptive_search_force_strategy_axis": True,
                    "adaptive_search_axis_pool": ["hash_build", "phase_interleave"],
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            state.scratch["required_strategy_axis"] = "hash_build"
            candidate = CodeCandidate(
                "drift",
                [
                    CodeChange(
                        path="target.py",
                        reason="phase interleave retry",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "phase interleave retry",
                strategy_axis="hash_build",
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
            self.assertIn("does not substantively target required", "\n".join(state.notes))
            history = (repo / ".local_micro_agent" / "candidates.jsonl").read_text()
            self.assertIn('"status": "rejected_axis_drift"', history)

    def test_family_contract_rejects_selected_tactic_drift_to_failed_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "failed_tactics.jsonl").write_text(
                json.dumps(
                    {
                        "strategy_axis": "memory_store_layout",
                        "family_key": "store_address_reuse",
                        "context": "Phase 4 store address reuse",
                    }
                )
                + "\n"
            )
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["target.py"],
                    "test_commands": ["python3 -c \"print('cycles: 80')\""],
                    "candidate_queue": True,
                    "adaptive_search_memory": True,
                    "adaptive_search_force_strategy_axis": True,
                    "adaptive_search_axis_pool": ["memory_store_layout"],
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                    "failed_tactics_path": ".local_micro_agent/failed_tactics.jsonl",
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            state.scratch["required_strategy_axis"] = "memory_store_layout"
            state.scratch["selected_tactic"] = {
                "strategy_axis": "memory_store_layout",
                "family_key": "double_buffer_scratch_swap",
                "text": "family_key: double_buffer_scratch_swap",
            }
            state.scratch["selected_tactic_loop"] = 0
            candidate = CodeCandidate(
                "family-drift",
                [
                    CodeChange(
                        path="target.py",
                        reason="Phase 4 store address reuse",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "Phase 4 store address reuse",
                strategy_axis="memory_store_layout",
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
            self.assertIn("instead of selected family_key double_buffer_scratch_swap", "\n".join(state.notes))
            history = (repo / ".local_micro_agent" / "candidates.jsonl").read_text()
            self.assertIn('"status": "rejected_family_drift"', history)

    def test_axis_contract_allows_required_axis_with_secondary_keywords(self) -> None:
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
                    "adaptive_search_force_strategy_axis": True,
                    "adaptive_search_axis_pool": ["hash_build", "instruction_scheduling"],
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                    "accept_if_improved": True,
                    "baseline_metric": 100,
                    "metric_regex": "cycles: (\\d+)",
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            state.scratch["required_strategy_axis"] = "hash_build"
            candidate = CodeCandidate(
                "hash",
                [
                    CodeChange(
                        path="target.py",
                        reason="hash build bundle scheduling",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "hash build bundle scheduling",
                strategy_axis="hash_build",
            )
            agent = MicroAgent(config, state)

            async def evaluate_once() -> None:
                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([candidate], {"target.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(evaluate_once())

            self.assertEqual(target.read_text(), "value = 'new'\n")
            history = (repo / ".local_micro_agent" / "candidates.jsonl").read_text()
            self.assertIn('"status": "improved"', history)

    def test_axis_contract_prompt_sets_required_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "adaptive_search_memory": True,
                    "adaptive_search_force_strategy_axis": True,
                    "adaptive_search_axis_pool": ["vector_unroll_lane", "hash_build"],
                },
            }
            agent = MicroAgent(config, state)

            contract = agent._format_axis_contract()

            self.assertIn('"required_strategy_axis": "vector_unroll_lane"', contract)
            self.assertIn('"required_axis_guidance"', contract)
            self.assertIn("per-lane or unroll-lane structure", contract)
            self.assertEqual(state.scratch["required_strategy_axis"], "vector_unroll_lane")

    def test_code_prompt_keeps_source_before_dynamic_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 1\n")
            state = AgentState(repo_root=repo, user_request="test")
            state.plan_markdown = "Plan text"
            state.file_context = [FileSnapshot("target.py", "value = 1\n")]
            state.notes.append("dynamic feedback")

            user_content = code_prompt(state)[1]["content"]

            self.assertLess(
                user_content.index("Source files:"),
                user_content.index("Latest test summary:"),
            )
            self.assertLess(
                user_content.index("Latest test summary:"),
                user_content.index("Recent agent feedback:"),
            )

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

    def test_validated_todo_status_is_not_downgraded_by_later_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            todo = {
                "todo_id": "todo-001-vector_unroll_lane",
                "status": "validated",
                "attempts": 1,
            }
            plan = {
                "version": 1,
                "active_todo_id": todo["todo_id"],
                "todos": [todo],
            }
            (artifact_dir / "todo_plan.json").write_text(json.dumps(plan) + "\n")
            (artifact_dir / "active_todo.json").write_text(json.dumps(todo) + "\n")
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=repo, user_request="test"),
            )

            agent._update_todo_status_from_attempt(
                {
                    "todo_id": "todo-001-vector_unroll_lane",
                    "status": "rejected_axis_drift",
                }
            )

            updated = json.loads((artifact_dir / "todo_plan.json").read_text())
            active = json.loads((artifact_dir / "active_todo.json").read_text())
            self.assertEqual(updated["todos"][0]["status"], "validated")
            self.assertEqual(active["status"], "validated")
            self.assertEqual(updated["todos"][0]["attempts"], 2)
            self.assertIsNone(updated["active_todo_id"])
            self.assertFalse((artifact_dir / "failed_tactics.jsonl").exists())

    def test_persist_todo_plan_appends_instead_of_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            first = {
                "todo_id": "todo-001-vector_unroll_lane",
                "status": "validated",
                "strategy_axis": "vector_unroll_lane",
            }
            active = {
                "todo_id": "todo-002-hash_build",
                "status": "active",
                "strategy_axis": "hash_build",
            }
            plan = {
                "version": 1,
                "active_todo_id": active["todo_id"],
                "todos": [first, active],
            }
            (artifact_dir / "todo_plan.json").write_text(json.dumps(plan) + "\n")
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=repo, user_request="test"),
            )

            agent._persist_todo_plan(
                {
                    "todo_id": "todo-003-memory_store_layout",
                    "status": "active",
                    "strategy_axis": "memory_store_layout",
                }
            )

            updated = json.loads((artifact_dir / "todo_plan.json").read_text())
            statuses = {todo["todo_id"]: todo["status"] for todo in updated["todos"]}
            self.assertEqual(updated["active_todo_id"], "todo-003-memory_store_layout")
            self.assertEqual(statuses["todo-001-vector_unroll_lane"], "validated")
            self.assertEqual(statuses["todo-002-hash_build"], "superseded")
            self.assertEqual(statuses["todo-003-memory_store_layout"], "active")

    def test_terminal_active_todo_is_not_injected_into_code_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "active_todo.json").write_text(
                json.dumps(
                    {
                        "todo_id": "todo-001-hash_build",
                        "status": "failed",
                        "strategy_axis": "hash_build",
                    }
                )
                + "\n"
            )
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=repo, user_request="test"),
            )

            self.assertEqual(agent._format_active_todo(), "")

    def test_failed_todo_writes_failed_tactic_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            todo = {
                "todo_id": "todo-001-hash_build",
                "status": "active",
                "strategy_axis": "hash_build",
                "context": "hash tactic",
            }
            plan = {
                "version": 1,
                "active_todo_id": todo["todo_id"],
                "todos": [todo],
            }
            (artifact_dir / "todo_plan.json").write_text(json.dumps(plan) + "\n")
            (artifact_dir / "active_todo.json").write_text(json.dumps(todo) + "\n")
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=repo, user_request="test"),
            )

            agent._update_todo_status_from_attempt(
                {
                    "todo_id": "todo-001-hash_build",
                    "status": "rejected",
                    "metric": 38423,
                    "reason": "got slower",
                }
            )

            failed = (artifact_dir / "failed_tactics.jsonl").read_text()
            updated = json.loads((artifact_dir / "todo_plan.json").read_text())
            self.assertIn("todo-001-hash_build", failed)
            self.assertIn("got slower", failed)
            self.assertIsNone(updated["active_todo_id"])

    def test_non_improving_todo_attempt_stays_active_until_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            todo = {
                "todo_id": "todo-001-instruction_scheduling",
                "status": "active",
                "strategy_axis": "instruction_scheduling",
                "context": "vliw packing tactic",
            }
            plan = {
                "version": 1,
                "active_todo_id": todo["todo_id"],
                "todos": [todo],
            }
            (artifact_dir / "todo_plan.json").write_text(json.dumps(plan) + "\n")
            (artifact_dir / "active_todo.json").write_text(json.dumps(todo) + "\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {"todo_attempt_budget": 3},
                },
                AgentState(repo_root=repo, user_request="test"),
            )

            agent._update_todo_status_from_attempt(
                {
                    "todo_id": "todo-001-instruction_scheduling",
                    "status": "rejected",
                    "metric": 147734,
                    "failed": False,
                    "reason": "valid probe but no metric improvement",
                }
            )

            updated = json.loads((artifact_dir / "todo_plan.json").read_text())
            active = json.loads((artifact_dir / "active_todo.json").read_text())
            self.assertEqual(updated["active_todo_id"], "todo-001-instruction_scheduling")
            self.assertEqual(updated["todos"][0]["status"], "attempted")
            self.assertEqual(updated["todos"][0]["attempts"], 1)
            self.assertEqual(active["status"], "attempted")
            self.assertFalse((artifact_dir / "failed_tactics.jsonl").exists())

    def test_failed_todo_attempt_stays_active_until_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            todo = {
                "todo_id": "todo-001-hash_build",
                "status": "active",
                "strategy_axis": "hash_build",
                "context": "hash reorder tactic",
            }
            plan = {
                "version": 1,
                "active_todo_id": todo["todo_id"],
                "todos": [todo],
            }
            (artifact_dir / "todo_plan.json").write_text(json.dumps(plan) + "\n")
            (artifact_dir / "active_todo.json").write_text(json.dumps(todo) + "\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {"todo_attempt_budget": 3},
                },
                AgentState(repo_root=repo, user_request="test"),
            )

            agent._update_todo_status_from_attempt(
                {
                    "todo_id": "todo-001-hash_build",
                    "status": "rejected_no_changes",
                    "metric": None,
                    "failed": True,
                    "reason": "search block missed the current source",
                }
            )

            updated = json.loads((artifact_dir / "todo_plan.json").read_text())
            active = json.loads((artifact_dir / "active_todo.json").read_text())
            self.assertEqual(updated["active_todo_id"], "todo-001-hash_build")
            self.assertEqual(updated["todos"][0]["status"], "attempted")
            self.assertEqual(updated["todos"][0]["attempts"], 1)
            self.assertEqual(active["status"], "attempted")
            self.assertFalse((artifact_dir / "failed_tactics.jsonl").exists())

    def test_active_todo_blocks_brainstorm_until_budget_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "candidates.jsonl").write_text(
                '{"status":"rejected","metric":147734,"failed":false}\n'
                '{"status":"rejected","metric":147734,"failed":false}\n'
            )
            todo = {
                "todo_id": "todo-001-instruction_scheduling",
                "status": "attempted",
                "strategy_axis": "instruction_scheduling",
                "context": "vliw packing tactic",
                "attempts": 1,
            }
            (artifact_dir / "active_todo.json").write_text(json.dumps(todo) + "\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "brainstorm_after_rejections": 2,
                        "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                        "todo_attempt_budget": 3,
                    },
                },
                AgentState(repo_root=repo, user_request="test", loop_count=2),
            )

            self.assertFalse(agent._should_brainstorm())

            todo["attempts"] = 3
            (artifact_dir / "active_todo.json").write_text(json.dumps(todo) + "\n")
            agent.state.scratch.pop("active_todo", None)
            self.assertTrue(agent._should_brainstorm())

    def test_active_todo_contract_rejects_axis_drift_before_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            todo = {
                "todo_id": "todo-001-phase_interleave",
                "status": "attempted",
                "strategy_axis": "phase_interleave",
                "family_key": "phase_pipeline",
                "context": "phase pipeline tactic",
                "attempts": 1,
            }
            plan = {
                "version": 1,
                "active_todo_id": todo["todo_id"],
                "todos": [todo],
            }
            (artifact_dir / "todo_plan.json").write_text(json.dumps(plan) + "\n")
            (artifact_dir / "active_todo.json").write_text(json.dumps(todo) + "\n")
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["target.py"],
                    "test_commands": ["python3 -c \"print('cycles: 80')\""],
                    "candidate_queue": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                    "todo_attempt_budget": 3,
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            agent = MicroAgent(config, state)
            candidate = CodeCandidate(
                "drift",
                [
                    CodeChange(
                        path="target.py",
                        reason="basic hazard-aware VLIW bundle packer",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "basic hazard-aware VLIW bundle packer",
                strategy_axis="instruction_scheduling",
            )

            async def evaluate_once() -> None:
                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([candidate], {"target.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(evaluate_once())

            self.assertEqual(target.read_text(), "value = 'old'\n")
            history = (artifact_dir / "candidates.jsonl").read_text()
            attempts = (artifact_dir / "todo_attempts.jsonl").read_text()
            updated = json.loads((artifact_dir / "todo_plan.json").read_text())
            self.assertIn('"status": "rejected_todo_axis_drift"', history)
            self.assertIn("todo-001-phase_interleave", attempts)
            self.assertEqual(updated["active_todo_id"], "todo-001-phase_interleave")
            self.assertEqual(updated["todos"][0]["status"], "attempted")
            self.assertEqual(updated["todos"][0]["attempts"], 2)
            self.assertFalse((artifact_dir / "failed_tactics.jsonl").exists())

    def test_active_todo_contract_rejects_family_drift_before_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            todo = {
                "todo_id": "todo-001-phase_interleave",
                "status": "attempted",
                "strategy_axis": "phase_interleave",
                "family_key": "phase_pipeline",
                "context": "phase pipeline tactic",
                "attempts": 1,
            }
            plan = {
                "version": 1,
                "active_todo_id": todo["todo_id"],
                "todos": [todo],
            }
            (artifact_dir / "todo_plan.json").write_text(json.dumps(plan) + "\n")
            (artifact_dir / "active_todo.json").write_text(json.dumps(todo) + "\n")
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["target.py"],
                    "test_commands": ["python3 -c \"print('cycles: 80')\""],
                    "candidate_queue": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                    "todo_attempt_budget": 3,
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            agent = MicroAgent(config, state)
            candidate = CodeCandidate(
                "family-drift",
                [
                    CodeChange(
                        path="target.py",
                        reason="phase stage hash reorder tmp1 tmp2",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "phase stage hash reorder tmp1 tmp2",
                strategy_axis="phase_interleave",
            )

            async def evaluate_once() -> None:
                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([candidate], {"target.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(evaluate_once())

            self.assertEqual(target.read_text(), "value = 'old'\n")
            history = (artifact_dir / "candidates.jsonl").read_text()
            updated = json.loads((artifact_dir / "todo_plan.json").read_text())
            self.assertIn('"status": "rejected_todo_family_drift"', history)
            self.assertEqual(updated["active_todo_id"], "todo-001-phase_interleave")
            self.assertEqual(updated["todos"][0]["status"], "attempted")
            self.assertEqual(updated["todos"][0]["attempts"], 2)
            self.assertFalse((artifact_dir / "failed_tactics.jsonl").exists())

    def test_non_improving_todo_fails_when_budget_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            todo = {
                "todo_id": "todo-001-instruction_scheduling",
                "status": "attempted",
                "strategy_axis": "instruction_scheduling",
                "context": "vliw packing tactic",
                "attempts": 2,
            }
            plan = {
                "version": 1,
                "active_todo_id": todo["todo_id"],
                "todos": [todo],
            }
            (artifact_dir / "todo_plan.json").write_text(json.dumps(plan) + "\n")
            (artifact_dir / "active_todo.json").write_text(json.dumps(todo) + "\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {"todo_attempt_budget": 3},
                },
                AgentState(repo_root=repo, user_request="test"),
            )

            agent._update_todo_status_from_attempt(
                {
                    "todo_id": "todo-001-instruction_scheduling",
                    "status": "rejected",
                    "metric": 147734,
                    "failed": False,
                    "reason": "third valid probe still did not improve",
                }
            )

            updated = json.loads((artifact_dir / "todo_plan.json").read_text())
            failed = (artifact_dir / "failed_tactics.jsonl").read_text()
            self.assertEqual(updated["todos"][0]["status"], "failed")
            self.assertEqual(updated["todos"][0]["attempts"], 3)
            self.assertIsNone(updated["active_todo_id"])
            self.assertIn("third valid probe still did not improve", failed)


if __name__ == "__main__":
    unittest.main()
