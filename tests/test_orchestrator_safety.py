from __future__ import annotations

import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from local_micro_agent.orchestrator import CodeCandidate, CodeDecision, MicroAgent, ReadDecision
from local_micro_agent.models import ModelResponse, _ollama_usage, _openai_usage
from local_micro_agent.prompts import (
    brainstorm_prompt,
    code_prompt,
    reflect_prompt,
    semantic_analysis_prompt,
    spec_prompt,
)
from local_micro_agent.state import (
    AgentState,
    AgentStateName,
    CodeChange,
    ExternalContext,
    FileSnapshot,
    TestResult,
)


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


class _UsageModel:
    async def chat(self, messages):
        return ModelResponse(
            '{"changes":[]}',
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "provider_prompt_eval_count": 10,
                "provider_eval_count": 5,
                "provider_prompt_eval_duration_ns": 2_000_000_000,
                "provider_eval_duration_ns": 1_000_000_000,
                "provider_total_duration_ns": 3_000_000_000,
            },
        )


class _UsageModelManager:
    def get(self, role):
        return _UsageModel()


class _StreamingModel:
    supports_streaming = True

    def __init__(self, chunks: list[str]):
        self.chunks = chunks

    async def chat(self, messages, stream_callback=None):
        for chunk in self.chunks:
            if stream_callback is not None:
                stream_callback(chunk)
        return "".join(self.chunks)


class _StreamingModelManager:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks

    def get(self, role):
        return _StreamingModel(self.chunks)


class _SequenceModel:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs

    async def chat(self, messages):
        if len(self.outputs) > 1:
            return self.outputs.pop(0)
        return self.outputs[0]


class _SequenceModelManager:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs

    def get(self, role):
        return _SequenceModel(self.outputs)


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


TAKEHOME_AXIS_POOL = [
    "hash_build",
    "phase_interleave",
    "vector_unroll_lane",
    "memory_store_layout",
    "precompute_constants",
    "branch_control",
    "instruction_scheduling",
    "general_edit",
]

TAKEHOME_AXIS_GUIDANCE = {
    "vector_unroll_lane": {
        "focus": "Change per-lane or unroll-lane structure.",
        "try": ["change lane-local temporary reuse"],
        "avoid_drift": ["global phase rewrite"],
    }
}


def takehome_workflow(**overrides: object) -> dict:
    workflow = {
        "adaptive_search_axis_pool": TAKEHOME_AXIS_POOL,
        "adaptive_search_axis_guidance": TAKEHOME_AXIS_GUIDANCE,
    }
    workflow.update(overrides)
    return workflow


class OrchestratorSafetyTests(unittest.TestCase):
    def test_ollama_usage_maps_provider_token_stats(self) -> None:
        usage = _ollama_usage(
            {
                "prompt_eval_count": 12,
                "eval_count": 7,
                "prompt_eval_duration": 3_000_000_000,
                "eval_duration": 1_400_000_000,
                "total_duration": 5_000_000_000,
            }
        )

        self.assertEqual(usage["prompt_tokens"], 12)
        self.assertEqual(usage["completion_tokens"], 7)
        self.assertEqual(usage["total_tokens"], 19)
        self.assertEqual(usage["provider_prompt_eval_count"], 12)
        self.assertEqual(usage["provider_eval_count"], 7)
        self.assertEqual(usage["provider_prompt_eval_duration_ns"], 3_000_000_000)
        self.assertEqual(usage["provider_eval_duration_ns"], 1_400_000_000)
        self.assertEqual(usage["provider_total_duration_ns"], 5_000_000_000)

    def test_openai_usage_maps_token_stats(self) -> None:
        usage = _openai_usage(
            {
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 13,
                    "total_tokens": 24,
                }
            }
        )

        self.assertEqual(usage["prompt_tokens"], 11)
        self.assertEqual(usage["completion_tokens"], 13)
        self.assertEqual(usage["total_tokens"], 24)

    def test_log_prefix_includes_timestamp(self) -> None:
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            MicroAgent._log("CODE loop=1")

        self.assertRegex(
            output.getvalue().strip(),
            r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4}\] "
            r"\[local-micro-agent\] CODE loop=1$",
        )

    def test_default_adaptive_axes_are_domain_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=Path(tmp), user_request="test"),
            )

            axes = agent._strategy_axis_pool()

            self.assertIn("performance", axes)
            self.assertIn("api_contract", axes)
            for takehome_axis in (
                "hash_build",
                "phase_interleave",
                "vector_unroll_lane",
                "memory_store_layout",
                "instruction_scheduling",
            ):
                self.assertNotIn(takehome_axis, axes)
            self.assertEqual(agent._family_key_strategy_axes("store_address_reuse"), [])
            self.assertEqual(
                agent._tactic_family_key("Precompute store addresses in phase 4."),
                "",
            )

    def test_patch_change_applies_when_touched_files_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=repo, user_request="test"),
            )
            change = CodeChange(
                path="target.py",
                reason="allowed patch",
                patch=(
                    "diff --git a/target.py b/target.py\n"
                    "--- a/target.py\n"
                    "+++ b/target.py\n"
                    "@@ -1 +1 @@\n"
                    "-value = 'old'\n"
                    "+value = 'new'\n"
                ),
            )

            async def apply_once() -> int:
                await agent.mcp.start()
                try:
                    return await agent._apply_changes([change], {"target.py"})
                finally:
                    await agent.mcp.close()

            applied = asyncio.run(apply_once())

            self.assertEqual(applied, 1)
            self.assertEqual(target.read_text(), "value = 'new'\n")

    def test_patch_change_cannot_touch_files_outside_allowed_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            forbidden = repo / "forbidden.py"
            target.write_text("value = 'old'\n")
            forbidden.write_text("secret = 'old'\n")
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=repo, user_request="test"),
            )
            change = CodeChange(
                path="target.py",
                reason="spoofed allowed path",
                patch=(
                    "diff --git a/forbidden.py b/forbidden.py\n"
                    "--- a/forbidden.py\n"
                    "+++ b/forbidden.py\n"
                    "@@ -1 +1 @@\n"
                    "-secret = 'old'\n"
                    "+secret = 'new'\n"
                ),
            )

            async def apply_once() -> int:
                await agent.mcp.start()
                try:
                    return await agent._apply_changes([change], {"target.py"})
                finally:
                    await agent.mcp.close()

            applied = asyncio.run(apply_once())

            self.assertEqual(applied, 0)
            self.assertEqual(target.read_text(), "value = 'old'\n")
            self.assertEqual(forbidden.read_text(), "secret = 'old'\n")
            self.assertIn(
                "Patch rejected: touches out-of-plan files: forbidden.py",
                "\n".join(agent.state.notes),
            )

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

    def test_profile_agent_records_phase_and_test_command_spans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")

            result = run_agent(
                repo,
                {
                    "profile_agent": True,
                    "writable_files": ["target.py"],
                    "seed_changes": [
                        {
                            "path": "target.py",
                            "target": "value = 'old'\n",
                            "replacement": "value = 'new'\n",
                        }
                    ],
                    "test_commands": ["python3 -c \"print('ok')\""],
                },
            )

            self.assertEqual(result.current, AgentStateName.DONE)
            profile_path = repo / ".local_micro_agent" / "profile_events.jsonl"
            rows = [
                json.loads(line)
                for line in profile_path.read_text().splitlines()
                if line.strip()
            ]
            self.assertTrue(any(row["event_type"] == "phase" for row in rows))
            self.assertTrue(any(row.get("phase") == "CODE" for row in rows))
            command_events = [
                row for row in rows if row["event_type"] == "test_command"
            ]
            self.assertEqual(len(command_events), 1)
            self.assertEqual(command_events[0]["exit_code"], 0)
            self.assertGreaterEqual(command_events[0]["elapsed_ms"], 0)

    def test_profile_agent_records_model_call_spans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            state = AgentState(repo_root=repo, user_request="test")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {"profile_agent": True},
            }
            agent = MicroAgent(config, state)
            agent.models = _StaticModelManager('{"changes":[]}')

            decision = asyncio.run(
                agent._json_call(
                    "coder",
                    [{"role": "user", "content": "make no changes"}],
                    schema=CodeDecision,
                )
            )

            self.assertIsNotNone(decision)
            profile_path = repo / ".local_micro_agent" / "profile_events.jsonl"
            rows = [
                json.loads(line)
                for line in profile_path.read_text().splitlines()
                if line.strip()
            ]
            model_events = [
                row for row in rows if row["event_type"] == "model_call"
            ]
            self.assertEqual(len(model_events), 1)
            self.assertEqual(model_events[0]["role"], "coder")
            self.assertEqual(model_events[0]["call_site"], "json_call")
            self.assertGreater(model_events[0]["prompt_chars"], 0)
            self.assertGreater(model_events[0]["output_chars"], 0)

    def test_profile_agent_records_model_token_rates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            state = AgentState(repo_root=repo, user_request="test")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {"profile_agent": True},
            }
            agent = MicroAgent(config, state)
            agent.models = _UsageModelManager()

            decision = asyncio.run(
                agent._json_call(
                    "coder",
                    [{"role": "user", "content": "make no changes"}],
                    schema=CodeDecision,
                )
            )

            self.assertIsNotNone(decision)
            profile_path = repo / ".local_micro_agent" / "profile_events.jsonl"
            rows = [
                json.loads(line)
                for line in profile_path.read_text().splitlines()
                if line.strip()
            ]
            model_event = next(row for row in rows if row["event_type"] == "model_call")
            self.assertEqual(model_event["prompt_tokens"], 10)
            self.assertEqual(model_event["completion_tokens"], 5)
            self.assertEqual(model_event["total_tokens"], 15)
            self.assertEqual(model_event["provider_prompt_eval_count"], 10)
            self.assertEqual(model_event["provider_eval_count"], 5)
            self.assertEqual(model_event["provider_prompt_eval_duration_ms"], 2000.0)
            self.assertEqual(model_event["provider_eval_duration_ms"], 1000.0)
            self.assertEqual(model_event["provider_total_duration_ms"], 3000.0)
            self.assertEqual(model_event["prompt_tokens_per_second"], 5.0)
            self.assertEqual(model_event["completion_tokens_per_second"], 5.0)
            self.assertEqual(model_event["total_tokens_per_second"], 5.0)
            self.assertGreater(model_event["wall_tokens_per_second"], 0)

    def test_reasoning_lane_routes_freeform_calls_without_touching_json_or_coder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            config = {
                "models": {
                    "default": "exact",
                    "planner": "exact",
                    "coder": "exact",
                    "brainstorm": "explore",
                    "reasoner": "reason",
                },
                "providers": {},
                "mcp_servers": {},
                "workflow": {"reasoning_lane_enabled": True},
            }
            agent = MicroAgent(config, AgentState(repo_root=repo, user_request="test"))
            models = _RoleModelManager(
                {
                    "reasoner": "freeform reasoning",
                    "planner": '{"files":["target.py"]}',
                    "coder": '{"changes":[]}',
                    "brainstorm": "1. strategy_axis: general_edit\ntry one thing",
                }
            )
            agent.models = models
            messages = [{"role": "user", "content": "go"}]

            async def run_calls() -> None:
                await agent._model_chat("planner", messages, call_site="plan")
                await agent._json_call("planner", messages, ReadDecision)
                await agent._model_chat("planner", messages, call_site="semantic_analysis")
                await agent._model_chat("reflector", messages, call_site="reflect")
                await agent._model_chat("brainstorm", messages, call_site="brainstorm")
                await agent._model_chat("coder", messages, call_site="json_call")

            asyncio.run(run_calls())

            self.assertEqual(len(models.seen["reasoner"]), 3)
            self.assertEqual(len(models.seen["planner"]), 1)
            self.assertEqual(len(models.seen["brainstorm"]), 1)
            self.assertEqual(len(models.seen["coder"]), 1)
            self.assertNotIn("reflector", models.seen)

    def test_profile_agent_records_streaming_model_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            state = AgentState(repo_root=repo, user_request="test")
            config = {
                "models": {"default": "stream"},
                "providers": {"stream": {"kind": "ollama_native", "model": "fake"}},
                "mcp_servers": {},
                "workflow": {
                    "profile_agent": True,
                    "profile_model_stream_log_interval_chars": 0,
                },
            }
            agent = MicroAgent(config, state)
            agent.models = _StreamingModelManager(['{"changes":', "[]}"])

            decision = asyncio.run(
                agent._json_call(
                    "coder",
                    [{"role": "user", "content": "make no changes"}],
                    schema=CodeDecision,
                )
            )

            self.assertIsNotNone(decision)
            stream_files = sorted((repo / ".local_micro_agent" / "model_streams").glob("*.txt"))
            self.assertEqual(len(stream_files), 1)
            self.assertEqual(stream_files[0].read_text(), '{"changes":[]}')
            profile_path = repo / ".local_micro_agent" / "profile_events.jsonl"
            rows = [
                json.loads(line)
                for line in profile_path.read_text().splitlines()
                if line.strip()
            ]
            model_event = next(row for row in rows if row["event_type"] == "model_call")
            self.assertTrue(model_event["streaming"])
            self.assertEqual(model_event["stream_chunks"], 2)
            self.assertEqual(model_event["stream_chars"], len('{"changes":[]}'))
            self.assertEqual(model_event["stream_path"], ".local_micro_agent/model_streams/" + stream_files[0].name)

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
            hint_dir = repo / "hints"
            hint_dir.mkdir()
            (hint_dir / "notes.md").write_text("# External hints\nPrefer small edits.\n")
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
                    "todo_soft_until_first_improvement": False,
                    "external_context_paths": ["hints/notes.md"],
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
                    self.assertIn(
                        "External advisory context follows",
                        models.seen["planner"][0][1]["content"],
                    )
                    self.assertIn("Prefer small edits", models.seen["planner"][0][1]["content"])
                finally:
                    await agent.mcp.close()

            asyncio.run(plan_only())

    def test_read_loads_external_context_paths_separately_from_source_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("value = 1\n")
            hint_dir = repo / "hints"
            hint_dir.mkdir()
            (hint_dir / "perf.md").write_text(
                "# Optimization notes\nPrefer latency hiding near the hot loop.\n"
            )
            state = AgentState(repo_root=repo, user_request="test")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "seed_files": ["target.py"],
                    "external_context_paths": ["hints/perf.md"],
                    "external_context_char_limit": 2000,
                    "external_context_item_char_limit": 1000,
                },
            }
            agent = MicroAgent(config, state)

            async def read_once() -> None:
                await agent.mcp.start()
                try:
                    await agent.read()
                finally:
                    await agent.mcp.close()

            asyncio.run(read_once())

            self.assertEqual([snap.path for snap in state.file_context], ["target.py"])
            self.assertEqual(len(state.external_context), 1)
            self.assertEqual(state.external_context[0].source, "hints/perf.md")
            self.assertEqual(state.external_context[0].kind, "hint")
            self.assertEqual(state.external_context[0].trust, "advisory")
            self.assertEqual(state.external_context[0].title, "Optimization notes")
            self.assertIn("latency hiding", state.external_context[0].content)
            self.assertIn("Loaded external context", "\n".join(state.notes))

    def test_read_cannot_promote_external_context_to_source_or_writable_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            hint_dir = repo / "hints"
            hint_dir.mkdir()
            (hint_dir / "perf.md").write_text("# Advisory hint\nDo not edit me.\n")
            state = AgentState(repo_root=repo, user_request="test")
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "external_context_paths": ["hints/perf.md"],
                    "external_context_char_limit": 2000,
                    "external_context_item_char_limit": 1000,
                },
            }
            agent = MicroAgent(config, state)
            agent.models = _StaticModelManager('{"files":["hints/perf.md"]}')

            async def read_once() -> None:
                await agent.mcp.start()
                try:
                    await agent.read()
                finally:
                    await agent.mcp.close()

            asyncio.run(read_once())

            self.assertEqual(state.planned_files, [])
            self.assertEqual(state.file_context, [])
            self.assertEqual(agent._writable_files(), set())
            self.assertEqual(len(state.external_context), 1)
            self.assertIn("Skipped advisory external context", "\n".join(state.notes))

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

            self.assertIn("Recent agent feedback", messages[2]["content"])
            self.assertIn("target not found", messages[2]["content"])
            self.assertIn("only changes comments", messages[2]["content"])

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

            self.assertIn("Retry reflection", messages[2]["content"])
            self.assertIn("invalid JSON", messages[2]["content"])

    def test_code_prompt_includes_semantic_analysis_with_stable_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            state.plan_markdown = "plan"
            state.file_context = [FileSnapshot("target.py", "value = 1\n")]
            state.scratch["semantic_analysis"] = (
                "- Execution model: writes become visible after each step."
            )

            messages = code_prompt(state)

            self.assertIn("Semantic analysis", messages[1]["content"])
            self.assertIn("writes become visible", messages[1]["content"])
            self.assertNotIn("Semantic analysis", messages[2]["content"])

    def test_prompts_include_external_advisory_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            state.plan_markdown = "plan"
            state.file_context = [FileSnapshot("target.py", "value = 1\n")]
            state.external_context = [
                ExternalContext(
                    kind="hint",
                    source="hints/perf.md",
                    title="Optimization notes",
                    content="Prefer latency hiding near the hot loop.",
                    sha256="abc123",
                )
            ]

            code_messages = code_prompt(state)
            semantic_messages = semantic_analysis_prompt(state)
            brainstorm_messages = brainstorm_prompt(
                state,
                reject_summary="[]",
                cooled_axes=[],
                known_axes=["performance"],
            )
            reflect_messages = reflect_prompt(state)

            for content in (
                code_messages[1]["content"],
                semantic_messages[1]["content"],
                brainstorm_messages[1]["content"],
                reflect_messages[1]["content"],
            ):
                self.assertIn("External advisory context follows", content)
                self.assertIn("hints/perf.md", content)
                self.assertIn("latency hiding", content)
                self.assertIn("Local source files, tests", content)

    def test_brainstorm_prompt_includes_semantic_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            state.plan_markdown = "plan"
            state.file_context = [FileSnapshot("target.py", "value = 1\n")]
            state.scratch["semantic_analysis"] = "- API contract: keep output stable."

            messages = brainstorm_prompt(
                state,
                reject_summary="[]",
                cooled_axes=[],
                known_axes=["performance"],
            )

            self.assertIn("Semantic analysis", messages[1]["content"])
            self.assertIn("keep output stable", messages[1]["content"])

    def test_code_prompt_can_request_xml_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            state.plan_markdown = "plan"

            messages = code_prompt(state, output_format="xml")

            self.assertIn("Do not output JSON", messages[0]["content"])
            self.assertIn("<search>", messages[0]["content"])
            self.assertIn("<replace>", messages[0]["content"])

    def test_read_can_persist_semantic_analysis_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("value = 1\n")
            state = AgentState(repo_root=repo, user_request="test")
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "seed_files": ["target.py"],
                    "semantic_analysis_after_read": True,
                    "semantic_analysis_path": ".local_micro_agent/semantic.md",
                },
            }
            agent = MicroAgent(config, state)
            agent.models = _StaticModelManager(
                "- Execution model: reads happen before writes become visible."
            )

            async def read_once() -> None:
                await agent.mcp.start()
                try:
                    await agent.read()
                finally:
                    await agent.mcp.close()

            asyncio.run(read_once())

            artifact = repo / ".local_micro_agent" / "semantic.md"
            self.assertTrue(artifact.exists())
            self.assertIn("reads happen before writes", artifact.read_text())
            self.assertIn("reads happen before writes", state.scratch["semantic_analysis"])
            self.assertEqual(state.current, AgentStateName.CODE)

    def test_semantic_analysis_filters_background_before_code_prompt(self) -> None:
        raw = """# Code-usable facts
- Preserve the public API contract.

# Background / non-constraints
- Best known benchmark note: Claude reached 1,363 cycles.

# Hazards and ordering constraints
- Reads happen before delayed writes become visible.
- There is no intra-step data dependency hazard.
"""

        curated = MicroAgent._curate_semantic_analysis(raw, 4000)

        self.assertIn("Preserve the public API contract", curated)
        self.assertIn("Reads happen before delayed writes", curated)
        self.assertNotIn("1,363", curated)
        self.assertNotIn("Claude", curated)
        self.assertNotIn("no intra-step data dependency hazard", curated.lower())
        self.assertIn("Controller validation", curated)
        self.assertIn("execution hazard", curated)

    def test_semantic_analysis_keeps_code_symbols_that_look_like_model_names(self) -> None:
        raw = """# Code-usable facts
- Preserve the claude_response field.
- Do not rename opuses in the public payload.

Background / non-constraints
- Best known benchmark note: Claude reached 1,363 cycles.
"""

        curated = MicroAgent._curate_semantic_analysis(raw, 4000)

        self.assertIn("claude_response", curated)
        self.assertIn("opuses", curated)
        self.assertNotIn("1,363", curated)
        self.assertNotIn("Background / non-constraints", curated)

    def test_read_persists_raw_and_curated_semantic_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("value = 1\n")
            state = AgentState(repo_root=repo, user_request="test")
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "seed_files": ["target.py"],
                    "semantic_analysis_after_read": True,
                    "semantic_analysis_path": ".local_micro_agent/semantic.md",
                    "semantic_analysis_curated_path": ".local_micro_agent/semantic.curated.md",
                },
            }
            agent = MicroAgent(config, state)
            agent.models = _StaticModelManager(
                "# Code-usable facts\n"
                "- Preserve behavior.\n\n"
                "# Background / non-constraints\n"
                "- Best known benchmark note: Claude reached 1,363 cycles.\n"
            )

            async def read_once() -> None:
                await agent.mcp.start()
                try:
                    await agent.read()
                finally:
                    await agent.mcp.close()

            asyncio.run(read_once())

            raw = repo / ".local_micro_agent" / "semantic.md"
            curated = repo / ".local_micro_agent" / "semantic.curated.md"
            self.assertIn("1,363", raw.read_text())
            self.assertNotIn("1,363", curated.read_text())
            self.assertNotIn("1,363", state.scratch["semantic_analysis"])

    def test_read_can_persist_run_spec_and_feed_code_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("value = 1\n")
            state = AgentState(repo_root=repo, user_request="speed up target")
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "seed_files": ["target.py"],
                    "run_spec_after_read": True,
                    "run_spec_path": ".local_micro_agent/run_spec.json",
                },
            }
            agent = MicroAgent(config, state)
            agent.models = _StaticModelManager(
                json.dumps(
                    {
                        "spec_id": "target-speed",
                        "objective": "Reduce repeated target work.",
                        "invariants": ["Preserve target output."],
                        "known_facts": ["target.py defines value."],
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "reduce repeated work",
                                "strategy_axis": "performance",
                                "family_key": "hot_path",
                                "expected_signal": "cycles decreases",
                                "status": "open",
                            }
                        ],
                        "decision_rules": ["patch_miss means refresh source"],
                    }
                )
            )

            async def read_once() -> None:
                await agent.mcp.start()
                try:
                    await agent.read()
                finally:
                    await agent.mcp.close()

            asyncio.run(read_once())

            spec_path = repo / ".local_micro_agent" / "run_spec.json"
            self.assertTrue(spec_path.exists())
            self.assertEqual(state.scratch["run_spec"]["spec_id"], "target-speed")
            messages = code_prompt(state)
            joined = "\n".join(message["content"] for message in messages)
            self.assertIn("Run-local spec", joined)
            self.assertIn("task-001", joined)

    def test_run_spec_artifact_is_not_loaded_without_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("value = 1\n")
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "run_spec.json").write_text(
                json.dumps(
                    {
                        "spec_id": "old-run",
                        "task_graph": [
                            {
                                "task_id": "old-task",
                                "strategy_axis": "performance",
                                "status": "open",
                            }
                        ],
                    }
                )
                + "\n"
            )
            state = AgentState(repo_root=repo, user_request="new unrelated task")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "seed_files": ["target.py"],
                    "run_spec_path": ".local_micro_agent/run_spec.json",
                },
            }
            agent = MicroAgent(config, state)

            async def read_once() -> None:
                await agent.mcp.start()
                try:
                    await agent.read()
                finally:
                    await agent.mcp.close()

            asyncio.run(read_once())

            self.assertNotIn("run_spec", state.scratch)
            joined = "\n".join(message["content"] for message in code_prompt(state))
            self.assertNotIn("old-task", joined)

    def test_spec_prompt_requests_task_graph_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            state.plan_markdown = "plan"
            state.file_context = [FileSnapshot("target.py", "value = 1\n")]

            messages = spec_prompt(state)

            self.assertIn("SPEC node", messages[0]["content"])
            self.assertIn("task_graph", messages[0]["content"])
            self.assertIn("target.py", messages[1]["content"])

    def test_structural_tactic_creates_structural_probe_todo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=Path(tmp), user_request="test"),
            )

            agent._create_active_todo_from_selected_tactic(
                {
                    "strategy_axis": "performance",
                    "family_key": "scheduler_rewrite",
                    "text": (
                        "family_key: scheduler_rewrite\n"
                        "Rewrite the scheduler with a guarded pipeline probe."
                    ),
                }
            )

            todo = agent.state.scratch["active_todo"]
            self.assertEqual(todo["tactic_stage"], "structural_probe")
            self.assertIn("scaffold/probe", todo["micro_goal"])

    def test_active_todo_links_to_run_spec_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=Path(tmp), user_request="test", loop_count=7),
            )
            agent.state.scratch["run_spec"] = {
                "spec_id": "perf",
                "task_graph": [
                    {
                        "task_id": "task-002",
                        "strategy_axis": "performance",
                        "family_key": "hot_path",
                        "status": "open",
                    }
                ],
            }

            agent._create_active_todo_from_selected_tactic(
                {
                    "strategy_axis": "performance",
                    "family_key": "hot_path",
                    "text": (
                        "strategy_axis: performance\n"
                        "family_key: hot_path\n"
                        "Hook: target.py"
                    ),
                }
            )

            todo = agent.state.scratch["active_todo"]
            self.assertEqual(todo["spec_task_id"], "task-002")

    def test_structural_correctness_failure_does_not_exhaust_todo_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["target.py"],
                    "test_commands": ["python3 -c \"raise AssertionError('invariant mismatch')\""],
                    "candidate_queue": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                    "todo_plan_path": ".local_micro_agent/todo_plan.json",
                    "active_todo_path": ".local_micro_agent/active_todo.json",
                    "todo_attempts_path": ".local_micro_agent/todo_attempts.jsonl",
                    "todo_attempt_budget": 1,
                    "todo_soft_until_first_improvement": False,
                    "structural_tactic_lifecycle": True,
                    "structural_tactic_soft_failures": 2,
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            agent = MicroAgent(config, state)
            todo = {
                "todo_id": "todo-000-performance",
                "status": "active",
                "strategy_axis": "performance",
                "family_key": "scheduler_rewrite",
                "tactic_stage": "structural_probe",
                "context": "Rewrite scheduler with a guarded probe.",
                "attempts": 0,
            }
            state.scratch["active_todo"] = todo
            agent._persist_todo_plan(todo)
            candidate = CodeCandidate(
                "structural",
                [
                    CodeChange(
                        path="target.py",
                        reason="todo-000-performance guarded scheduler probe",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "todo-000-performance guarded scheduler probe",
                strategy_axis="performance",
            )

            async def evaluate_once() -> None:
                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([candidate], {"target.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(evaluate_once())

            history = (repo / ".local_micro_agent" / "candidates.jsonl").read_text()
            self.assertIn('"failure_class": "invariant_broken"', history)
            self.assertIn('"budget_counted": false', history)
            active_todo = json.loads((repo / ".local_micro_agent" / "active_todo.json").read_text())
            self.assertEqual(active_todo["status"], "active")
            self.assertEqual(active_todo.get("attempts", 0), 0)
            self.assertEqual(active_todo.get("non_budget_attempts"), 1)
            self.assertNotIn("patch_failures", active_todo)

    def test_structural_probe_checkpoint_persists_without_metric_gain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "parser.py"
            target.write_text("def parse(value):\n    return value.strip()\n")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["parser.py"],
                    "test_commands": ["python3 -c \"print('cycles: 100')\""],
                    "metric_regex": r"cycles: (\d+)",
                    "baseline_metric": 100,
                    "candidate_queue": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                    "record_candidate_artifacts": True,
                    "todo_plan_path": ".local_micro_agent/todo_plan.json",
                    "active_todo_path": ".local_micro_agent/active_todo.json",
                    "todo_attempts_path": ".local_micro_agent/todo_attempts.jsonl",
                    "todo_soft_until_first_improvement": True,
                    "structural_tactic_lifecycle": True,
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"parser.py": target.read_text()}
            agent = MicroAgent(config, state)
            todo = {
                "todo_id": "todo-000-parser_refactor",
                "status": "active",
                "strategy_axis": "parser_refactor",
                "family_key": "parser_adapter",
                "tactic_stage": "structural_probe",
                "context": "Refactor parser through a behavior-preserving adapter.",
                "attempts": 0,
            }
            state.scratch["active_todo"] = todo
            agent._persist_todo_plan(todo)
            candidate = CodeCandidate(
                "adapter",
                [
                    CodeChange(
                        path="parser.py",
                        reason="todo-000-parser_refactor adapter scaffold",
                        target="def parse(value):\n    return value.strip()\n",
                        replacement=(
                            "def _parse_adapter(value):\n"
                            "    return value.strip()\n\n"
                            "def parse(value):\n"
                            "    return _parse_adapter(value)\n"
                        ),
                    )
                ],
                "todo-000-parser_refactor adapter scaffold",
                strategy_axis="parser_refactor",
            )

            async def evaluate_once() -> None:
                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([candidate], {"parser.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(evaluate_once())

            self.assertEqual(target.read_text(), "def parse(value):\n    return value.strip()\n")
            history_record = json.loads(
                (repo / ".local_micro_agent" / "candidates.jsonl").read_text()
            )
            self.assertEqual(history_record["todo_id"], "todo-000-parser_refactor")
            self.assertEqual(history_record["failure_class"], "probe_no_signal")
            self.assertEqual(
                history_record["stage_result"], "probe_validated_no_metric_gain"
            )
            state_record = json.loads(
                (repo / ".local_micro_agent" / "structural_state.json").read_text()
            )
            checkpoint = state_record["checkpoints"][0]
            self.assertEqual(checkpoint["strategy_axis"], "parser_refactor")
            self.assertEqual(checkpoint["todo_id"], "todo-000-parser_refactor")
            self.assertEqual(checkpoint["stage_result"], "probe_validated_no_metric_gain")
            checkpoint_patch = repo / checkpoint["patch_path"]
            self.assertIn("_parse_adapter", checkpoint_patch.read_text())
            self.assertIn("_parse_adapter", agent._format_structural_state_context())
            active_todo = json.loads(
                (repo / ".local_micro_agent" / "active_todo.json").read_text()
            )
            self.assertEqual(active_todo["status"], "validated")

    def test_reflect_state_stores_summary_for_next_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(
                repo_root=Path(tmp),
                user_request="test",
                current=AgentStateName.REFLECT,
            )
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": takehome_workflow(),
            }
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

    def test_candidate_history_records_no_change_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["target.py"],
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                    "record_candidate_artifacts": True,
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            agent = MicroAgent(config, state)
            candidate = CodeCandidate(
                "miss",
                [
                    CodeChange(
                        path="target.py",
                        reason="try target block",
                        target="value = 'missing'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "try target block",
            )

            async def evaluate_once() -> None:
                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([candidate], {"target.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(evaluate_once())

            self.assertEqual(target.read_text(), "value = 'old'\n")
            history_path = repo / ".local_micro_agent" / "candidates.jsonl"
            record = json.loads(history_path.read_text().splitlines()[0])
            self.assertEqual(record["status"], "rejected_no_changes")
            self.assertIn("Replacement target not found", record["failure_detail"])
            self.assertIn("Replacement target not found", record["no_change_reason"])
            self.assertEqual(record["failure_class"], "patch_miss")
            self.assertIn("Retarget", " ".join(record["next_actions"]))
            self.assertIn("refreshing file context", record["recovery_hint"])
            artifact_path = repo / record["artifact_path"]
            self.assertTrue(artifact_path.exists())
            artifact = json.loads(artifact_path.read_text())
            self.assertEqual(artifact["candidate_id"], "miss")
            self.assertEqual(artifact["failure_class"], "patch_miss")
            self.assertIn("Replacement target not found", agent._format_candidate_history())

    def test_candidate_history_records_structured_failure_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                    "record_candidate_artifacts": True,
                },
            }
            agent = MicroAgent(config, AgentState(repo_root=repo, user_request="test"))
            candidate = CodeCandidate(
                "broken",
                [CodeChange("target.py", "semantic edit", content="bad = True\n")],
                "try a semantic edit",
                "correctness",
            )
            result = TestResult(
                command="python3 -m pytest",
                exit_code=1,
                stdout="FAILED test_target.py::test_contract",
                stderr="AssertionError: contract changed",
            )
            extra = agent._candidate_history_extra(
                candidate,
                status="rejected",
                metric=None,
                applied=1,
                failed=True,
                patch_text="diff --git a/target.py b/target.py\n",
                results=[result],
                failure_detail=agent._candidate_failure_detail([], [result], failed=True),
            )

            agent._append_candidate_history(
                candidate,
                status="rejected",
                metric=None,
                applied=1,
                failed=True,
                extra=extra,
            )

            record = json.loads((artifact_dir / "candidates.jsonl").read_text())
            self.assertEqual(record["failure_class"], "correctness_failure")
            self.assertIn("failing command", " ".join(record["next_actions"]))
            self.assertIn("failing assertion", record["recovery_hint"])
            self.assertIn("correctness_failure", agent._format_recent_reject_summary())
            artifact = json.loads((repo / record["artifact_path"]).read_text())
            self.assertEqual(artifact["failure_class"], "correctness_failure")
            self.assertTrue((repo / record["test_output_path"]).exists())

    def test_todo_attempt_copies_structured_candidate_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            todo = {
                "todo_id": "todo-001-performance",
                "status": "active",
                "strategy_axis": "performance",
                "context": "try one performance tactic",
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
                    "workflow": {
                        "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                        "todo_attempt_budget": 3,
                        "todo_soft_until_first_improvement": False,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = todo
            candidate = CodeCandidate(
                "slow",
                [CodeChange("target.py", "performance edit", content="value = 1\n")],
                "same performance tactic",
                "performance",
            )
            extra = agent._candidate_history_extra(
                candidate,
                status="rejected",
                metric=120,
                applied=1,
                failed=False,
                patch_text="",
                results=[TestResult("python3 bench.py", 0, "cycles: 120\n", "")],
            )

            agent._append_candidate_history(
                candidate,
                status="rejected",
                metric=120,
                applied=1,
                failed=False,
                extra=extra,
            )

            attempts = [
                json.loads(line)
                for line in (artifact_dir / "todo_attempts.jsonl").read_text().splitlines()
            ]
            self.assertEqual(attempts[0]["failure_class"], "no_improvement")
            self.assertIn("vary the tactic", attempts[0]["recovery_hint"])
            todo_summary = agent._format_todo_ledger_summary()
            self.assertIn("last_failure_class", todo_summary)
            self.assertIn("no_improvement", todo_summary)

    def test_diagnostic_commands_are_recorded_as_candidate_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("value = 1\n")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                    "record_candidate_artifacts": True,
                    "test_commands": ["python3 -c \"print('tests ok')\""],
                    "diagnostic_commands": [
                        {
                            "name": "shape",
                            "command": "python3 -c \"print('OBS instructions=7 bundles=7')\"",
                        }
                    ],
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            agent = MicroAgent(config, state)
            state.scratch["pre_code_snapshot"] = {"target.py": "value = 1\n"}
            candidate = CodeCandidate(
                "diag",
                [CodeChange("target.py", "change value", content="value = 2\n")],
                "make an observable edit",
                "performance",
            )

            async def run_once() -> None:
                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([candidate], {"target.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(run_once())

            record = json.loads((repo / ".local_micro_agent/candidates.jsonl").read_text())
            self.assertIn("diagnostic_summary", record)
            self.assertIn("instructions=7", record["diagnostic_summary"])
            self.assertIn("diagnostics", record)
            self.assertIn("Candidate diag diagnostics", "\n".join(state.notes))
            artifact = json.loads((repo / record["artifact_path"]).read_text())
            self.assertIn("diagnostics", artifact)
            self.assertIn("OBS instructions=7", artifact["diagnostics"][0]["output"])

    def test_soft_todo_still_formats_observation_chain_for_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "todo_soft_until_first_improvement": True,
                    "pre_improvement_todo_blocks_brainstorm": True,
                    "observation_backed_todo_continuation": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            agent = MicroAgent(config, state)
            state.scratch["active_todo"] = {
                "todo_id": "todo-001-performance",
                "status": "active",
                "strategy_axis": "performance",
                "tactic_stage": "local_edit",
                "context": "reduce generated work",
                "last_attempt": {
                    "loop": 0,
                    "todo_id": "todo-001-performance",
                    "candidate_id": "1",
                    "status": "rejected",
                    "metric": 100,
                    "strategy_axis": "performance",
                    "failure_class": "no_improvement",
                    "summary": "metric stayed flat",
                    "diagnostic_summary": "shape exit=0: instructions unchanged",
                    "recovery_hint": "move the edit site",
                },
            }

            self.assertEqual(agent._format_active_todo(), "")
            chain = agent._format_todo_observation_chain()

            self.assertIn("todo-001-performance", chain)
            self.assertIn("instructions unchanged", chain)
            self.assertIn("move the edit", chain)

    def test_candidate_observation_updates_run_spec_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            spec = {
                "spec_id": "perf",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "strategy_axis": "performance",
                        "family_key": "hot_path",
                        "status": "open",
                    }
                ],
            }
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "todo_soft_until_first_improvement": False,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            todo = {
                "todo_id": "todo-001-performance",
                "status": "active",
                "strategy_axis": "performance",
                "spec_task_id": "task-001",
            }
            agent.state.scratch["run_spec"] = spec
            agent.state.scratch["active_todo"] = todo
            candidate = CodeCandidate(
                "miss",
                [CodeChange("target.py", "performance edit", target="old", replacement="new")],
                "same performance tactic",
                "performance",
            )
            extra = agent._candidate_history_extra(
                candidate,
                status="rejected_no_changes",
                metric=None,
                applied=0,
                failed=True,
                patch_text="",
                results=[],
                failure_detail="Replacement target not found: target.py",
                no_change_reason="Replacement target not found: target.py",
            )

            agent._append_candidate_history(
                candidate,
                status="rejected_no_changes",
                metric=None,
                applied=0,
                failed=True,
                extra=extra,
            )

            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            task = persisted["task_graph"][0]
            self.assertEqual(task["status"], "needs_repair")
            self.assertEqual(task["decision_hint"], "repair_with_fresh_source_context_before_retry")
            self.assertEqual(task["last_observation"]["failure_class"], "patch_miss")

    def test_target_not_found_repair_can_fix_search_block_within_candidate(self) -> None:
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
                    "test_commands": [
                        (
                            "python3 -c \"from pathlib import Path; "
                            "t=Path('target.py').read_text(); "
                            "print('cycles: 80' if 'fast' in t else 'cycles: 120')\""
                        )
                    ],
                    "candidate_queue": True,
                    "metric_regex": r"cycles: (\d+)",
                    "baseline_metric": 100,
                    "accept_if_improved": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                    "record_candidate_artifacts": True,
                    "repair_target_not_found": True,
                    "code_output_format": "xml",
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            agent = MicroAgent(config, state)
            agent.models = _StaticModelManager(
                """
<candidates>
<candidate id="fixed">
<strategy_axis>general_edit</strategy_axis>
<reason>Repair stale search text using the current source.</reason>
<search>
value = 'old'
</search>
<replace>
value = 'fast'
</replace>
</candidate>
</candidates>
"""
            )
            candidate = CodeCandidate(
                "miss",
                [
                    CodeChange(
                        path="target.py",
                        reason="make it fast",
                        target="value = 'missing'\n",
                        replacement="value = 'fast'\n",
                    )
                ],
                "make it fast",
            )

            async def evaluate_once() -> None:
                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([candidate], {"target.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(evaluate_once())

            self.assertEqual(target.read_text(), "value = 'fast'\n")
            self.assertIn("target-not-found repair generated", "\n".join(state.notes))
            rows = [
                json.loads(line)
                for line in (repo / ".local_micro_agent" / "candidates.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(rows[0]["candidate_id"], "miss-repair1")
            self.assertEqual(rows[0]["status"], "improved")
            self.assertEqual(rows[0]["repair_parent_id"], "miss")
            self.assertEqual(rows[0]["metric"], 80)
            self.assertEqual(rows[1]["status"], "accepted")

    def test_target_not_found_repair_failure_records_repair_parent(self) -> None:
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
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                    "record_candidate_artifacts": True,
                    "repair_target_not_found": True,
                    "code_output_format": "xml",
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            agent = MicroAgent(config, state)
            agent.models = _StaticModelManager(
                """
<candidates>
<candidate id="still-missing">
<strategy_axis>general_edit</strategy_axis>
<reason>Still stale.</reason>
<change>
<path>target.py</path>
<search>
value = 'also missing'
</search>
<replace>
value = 'fast'
</replace>
</change>
</candidate>
</candidates>
"""
            )
            candidate = CodeCandidate(
                "miss",
                [
                    CodeChange(
                        path="target.py",
                        reason="make it fast",
                        target="value = 'missing'\n",
                        replacement="value = 'fast'\n",
                    )
                ],
                "make it fast",
            )

            async def evaluate_once() -> None:
                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([candidate], {"target.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(evaluate_once())

            self.assertEqual(target.read_text(), "value = 'old'\n")
            record = json.loads(
                (repo / ".local_micro_agent" / "candidates.jsonl")
                .read_text()
                .splitlines()[0]
            )
            self.assertEqual(record["candidate_id"], "miss-repair1")
            self.assertEqual(record["status"], "rejected_no_changes")
            self.assertEqual(record["repair_parent_id"], "miss")
            self.assertIn("Replacement target not found", record["failure_detail"])

    def test_target_not_found_repair_uses_loose_parser_after_json_repair(self) -> None:
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
                    "metric_regex": r"cycles: (\d+)",
                    "baseline_metric": 100,
                    "accept_if_improved": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                    "repair_target_not_found": True,
                    "code_output_format": "xml",
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            agent = MicroAgent(config, state)
            agent.models = _SequenceModelManager(
                [
                    "<candidates><candidate><search>value = 'old'</search></candidate></candidates>",
                    json.dumps(
                        {
                            "candidates": [
                                {
                                    "search": "value = 'old'",
                                    "replacement": "value = 'fast'",
                                    "strategy_axis": "general_edit",
                                    "reason": "loose repaired json",
                                }
                            ]
                        }
                    ),
                ]
            )
            candidate = CodeCandidate(
                "miss",
                [
                    CodeChange(
                        path="target.py",
                        reason="make it fast",
                        target="value = 'missing'\n",
                        replacement="value = 'fast'\n",
                    )
                ],
                "make it fast",
            )

            async def evaluate_once() -> None:
                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([candidate], {"target.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(evaluate_once())

            self.assertEqual(target.read_text(), "value = 'fast'\n")
            record = json.loads(
                (repo / ".local_micro_agent" / "candidates.jsonl")
                .read_text()
                .splitlines()[0]
            )
            self.assertEqual(record["candidate_id"], "miss-repair1")
            self.assertEqual(record["repair_parent_id"], "miss")
            self.assertEqual(record["status"], "improved")

    def test_candidate_artifacts_record_patch_and_test_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["target.py"],
                    "test_commands": [
                        (
                            "python3 -c \"import sys; print('cycles: 120'); "
                            "print('boom', file=sys.stderr); sys.exit(1)\""
                        )
                    ],
                    "candidate_queue": True,
                    "metric_regex": r"cycles: (\d+)",
                    "baseline_metric": 100,
                    "accept_if_improved": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                    "record_candidate_artifacts": True,
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            agent = MicroAgent(config, state)
            candidate = CodeCandidate(
                "slow",
                [
                    CodeChange(
                        path="target.py",
                        reason="make it slow",
                        target="value = 'old'\n",
                        replacement="value = 'slow'\n",
                    )
                ],
                "make it slow",
            )

            async def evaluate_once() -> None:
                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([candidate], {"target.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(evaluate_once())

            self.assertEqual(target.read_text(), "value = 'old'\n")
            record = json.loads(
                (repo / ".local_micro_agent" / "candidates.jsonl")
                .read_text()
                .splitlines()[0]
            )
            self.assertEqual(record["status"], "rejected")
            self.assertEqual(record["metric"], 120)
            self.assertIn("patch_path", record)
            self.assertIn("test_output_path", record)
            patch_text = (repo / record["patch_path"]).read_text()
            test_text = (repo / record["test_output_path"]).read_text()
            self.assertIn("-value = 'old'", patch_text)
            self.assertIn("+value = 'slow'", patch_text)
            self.assertIn("boom", test_text)
            self.assertIn("exit_code=1", test_text)

    def test_adaptive_search_memory_cools_down_repeated_failed_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": takehome_workflow(
                    writable_files=["target.py"],
                    test_commands=["python3 -c \"print('cycles: 120')\""],
                    candidate_queue=True,
                    adaptive_search_memory=True,
                    adaptive_search_axis_failure_threshold=3,
                    adaptive_search_axis_cooldown_loops=4,
                    metric_regex=r"cycles: (\d+)",
                    baseline_metric=100,
                    accept_if_improved=True,
                    candidate_history_path=".local_micro_agent/candidates.jsonl",
                ),
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
                "try phase interleave overlap",
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
                "workflow": takehome_workflow(
                    brainstorm_after_rejections=2,
                    candidate_history_path=".local_micro_agent/candidates.jsonl",
                ),
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
                "workflow": takehome_workflow(
                    brainstorm_after_rejections=2,
                    brainstorm_new_family_after_all_skipped=2,
                    brainstorm_open_novelty_lanes=[
                        "layout_or_tiling_change: try a small layout probe",
                        "control_or_guard_lowering: try a guard lowering probe",
                    ],
                    candidate_history_path=".local_micro_agent/candidates.jsonl",
                ),
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
                "workflow": takehome_workflow(
                    brainstorm_after_rejections=2,
                    brainstorm_new_family_after_all_skipped=0,
                    brainstorm_open_novelty_lanes=[
                        "coarse_unroll_lane_restructure: try a small unroll probe",
                        "load_latency_scheduling: move one load-use boundary",
                    ],
                    candidate_history_path=".local_micro_agent/candidates.jsonl",
                ),
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

    def test_strict_axis_pool_rejects_observed_non_pool_brainstorm_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "adaptive_search_axis_pool": ["allowed_axis"],
                    "adaptive_search_strict_axis_pool": True,
                },
            }
            state.scratch["adaptive_search_memory"] = {
                "axes": {
                    "outside_axis": {
                        "attempts": 1,
                        "failures": 1,
                        "successes": 0,
                    }
                },
                "recent": [],
            }
            agent = MicroAgent(config, state)

            selected = agent._select_brainstorm_tactic(
                "1. strategy_axis: outside_axis\n"
                "family_key: outside_family\n"
                "Hook: retry the observed outside axis.\n"
                "2. strategy_axis: allowed_axis\n"
                "family_key: allowed_family\n"
                "Hook: try the configured strict-pool axis.\n"
            )

            self.assertIsNotNone(selected)
            self.assertEqual(selected["strategy_axis"], "allowed_axis")
            records = agent.state.scratch["brainstorm_selection"]
            self.assertEqual(records[0]["reason"], "unknown_axis")
            self.assertTrue(records[0]["skipped"])

    def test_strict_axis_contract_ignores_selected_non_pool_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "adaptive_search_memory": True,
                    "adaptive_search_force_strategy_axis": True,
                    "adaptive_search_axis_pool": ["allowed_axis"],
                    "adaptive_search_strict_axis_pool": True,
                },
            }
            state.scratch["selected_tactic"] = {
                "strategy_axis": "outside_axis",
                "family_key": "outside_family",
                "text": "1. strategy_axis: outside_axis",
            }
            state.scratch["selected_tactic_loop"] = 0
            state.scratch["adaptive_search_memory"] = {
                "axes": {
                    "outside_axis": {
                        "attempts": 1,
                        "failures": 1,
                        "successes": 0,
                    }
                },
                "recent": [],
            }
            agent = MicroAgent(config, state)

            contract = agent._format_axis_contract()
            payload = json.loads(contract)

            self.assertIn('"required_strategy_axis": "allowed_axis"', contract)
            self.assertIn('"known_strategy_axes": [\n    "allowed_axis"', contract)
            self.assertNotIn('"outside_axis"', contract)
            self.assertNotIn('"outside_family"', contract)
            self.assertIsNone(payload["required_family_key"])
            self.assertEqual(payload["selected_tactic"], {})

    def test_strict_axis_pool_ignores_non_pool_selected_family_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "failed_tactics.jsonl").write_text(
                json.dumps(
                    {
                        "strategy_axis": "allowed_axis",
                        "family_key": "failed_family",
                        "status": "failed",
                        "attempts": 2,
                    }
                )
                + "\n"
            )
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "adaptive_search_axis_pool": ["allowed_axis"],
                    "adaptive_search_strict_axis_pool": True,
                    "failed_tactics_path": ".local_micro_agent/failed_tactics.jsonl",
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["selected_tactic"] = {
                "strategy_axis": "outside_axis",
                "family_key": "outside_family",
                "text": "1. strategy_axis: outside_axis",
            }
            state.scratch["selected_tactic_loop"] = 0
            agent = MicroAgent(config, state)
            candidate = CodeCandidate(
                "candidate",
                [
                    CodeChange(
                        path="target.py",
                        reason="family_key: failed_family\ntry failed family",
                        target="old",
                        replacement="new",
                    )
                ],
                "family_key: failed_family\ntry failed family",
                strategy_axis="allowed_axis",
            )

            self.assertIsNone(agent._candidate_family_contract_rejection(candidate))

    def test_strict_axis_pool_rejects_unknown_candidate_without_force_contract(self) -> None:
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
                    "adaptive_search_axis_pool": ["allowed_axis"],
                    "adaptive_search_strict_axis_pool": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            agent = MicroAgent(config, state)
            candidate = CodeCandidate(
                "outside",
                [
                    CodeChange(
                        path="target.py",
                        reason="outside axis edit",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "outside axis edit",
                strategy_axis="outside_axis",
            )

            async def evaluate_once() -> None:
                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([candidate], {"target.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(evaluate_once())

            self.assertEqual(target.read_text(), "value = 'old'\n")
            history = (repo / ".local_micro_agent" / "candidates.jsonl").read_text()
            self.assertIn('"status": "rejected_unknown_axis"', history)

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
                    "workflow": takehome_workflow(failed_tactic_similarity_threshold=0.35),
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
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": takehome_workflow()},
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
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": takehome_workflow()},
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

    def test_brainstorm_selection_scores_recent_validated_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "validated_patterns.jsonl").write_text(
                json.dumps(
                    {
                        "strategy_axis": "memory_store_layout",
                        "family_key": "store_address_reuse",
                        "status": "validated",
                        "last_attempt": {
                            "strategy_axes": ["memory_store_layout"],
                            "metric": 100,
                        },
                    }
                )
                + "\n"
            )
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": takehome_workflow()},
                AgentState(repo_root=repo, user_request="test"),
            )

            selected = agent._select_brainstorm_tactic(
                "1. strategy_axis: branch_control\n"
                "family_key: branch_mask\n"
                "Hook: try a small branch-control variant.\n"
                "2. strategy_axis: memory_store_layout\n"
                "family_key: store_address_reuse\n"
                "Hook: extend the current validated local store layout pattern.\n"
            )

            self.assertIsNotNone(selected)
            self.assertEqual(selected["strategy_axis"], "memory_store_layout")
            records = agent.state.scratch["brainstorm_selection"]
            selected_record = next(record for record in records if record.get("selected"))
            self.assertIn(
                "extends_recent_validated_pattern",
                selected_record["score_reasons"],
            )

    def test_brainstorm_selection_treats_family_key_as_freeform_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": takehome_workflow()},
                AgentState(repo_root=repo, user_request="test"),
            )

            selected = agent._select_brainstorm_tactic(
                "1. strategy_axis: instruction_scheduling\n"
                "family_key: unroll_factor_change\n"
                "Hook: unroll a loop.\n"
                "2. strategy_axis: memory_store_layout\n"
                "family_key: store_address_reuse\n"
                "Hook: reuse one stored address.\n"
            )

            self.assertIsNotNone(selected)
            self.assertEqual(selected["strategy_axis"], "instruction_scheduling")
            records = agent.state.scratch["brainstorm_selection"]
            self.assertEqual(records[0]["family_key"], "unroll_factor_change")
            self.assertFalse(records[0]["skipped"])

    def test_brainstorm_selection_accepts_dynamic_axis_from_request_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": takehome_workflow()},
                AgentState(repo_root=repo, user_request="test"),
            )

            selected = agent._select_brainstorm_tactic(
                "1. strategy_axis: store_address_reuse\n"
                "family_key: store_address_reuse\n"
                "Hook: reuse one stored address for repeated stores.\n"
            )

            self.assertIsNotNone(selected)
            self.assertEqual(selected["strategy_axis"], "store_address_reuse")
            records = agent.state.scratch["brainstorm_selection"]
            self.assertEqual(records[0]["declared_axis"], "store_address_reuse")
            self.assertEqual(records[0]["axis"], "store_address_reuse")

    def test_brainstorm_selection_parses_axis_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": takehome_workflow()},
                AgentState(repo_root=repo, user_request="test"),
            )

            selected = agent._select_brainstorm_tactic(
                "1. **Tactic**: Use `hash_reorder` under the `hash_build` axis "
                "to flatten independent hash stages.\n"
                "family_key: hash_reorder\n"
            )

            self.assertIsNotNone(selected)
            self.assertEqual(selected["strategy_axis"], "hash_build")
            records = agent.state.scratch["brainstorm_selection"]
            self.assertEqual(records[0]["axis_source"], "axis_phrase")

    def test_family_axis_matching_only_matches_explicit_known_axis_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": takehome_workflow()},
                AgentState(repo_root=repo, user_request="test"),
            )

            self.assertEqual(
                agent._family_key_strategy_axes("memory_store_layout"),
                ["memory_store_layout"],
            )
            self.assertEqual(agent._family_key_strategy_axes("list_scheduler_rewrite"), [])

    def test_axis_matching_accepts_safe_word_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": takehome_workflow()},
                AgentState(repo_root=repo, user_request="test"),
            )

            axes = agent._strategy_axes_for_text(
                "Process two lanes with loop unrolling to expose parallelism.",
                agent._strategy_axis_keywords(),
            )

            self.assertIn("vector_unroll_lane", axes)
            self.assertNotIn("memory_store_layout", axes)

    def test_family_key_matching_does_not_substring_match_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": takehome_workflow()},
                AgentState(repo_root=repo, user_request="test"),
            )
            text = (
                "Precompute store addresses for indices and values outside the inner loop "
                "to reduce ALU pressure."
            )

            self.assertNotEqual(agent._tactic_family_key(text), "valu_vectorization")
            self.assertNotIn("valu_vectorization", agent._tactic_family_aliases(text))

    def test_family_key_requires_explicit_model_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": takehome_workflow()},
                AgentState(repo_root=repo, user_request="test"),
            )
            text = (
                "Precompute store addresses for indices and values outside the hash loop "
                "to eliminate redundant ALU operations."
            )

            self.assertEqual(agent._tactic_family_key(text), "")
            self.assertEqual(agent._tactic_family_aliases(text), set())
            explicit = f"{text}\nfamily_key: store_address_reuse\n"
            self.assertEqual(agent._tactic_family_key(explicit), "store_address_reuse")
            self.assertIn("store_address_reuse", agent._tactic_family_aliases(explicit))
            self.assertNotIn("hash_constant_fold", agent._tactic_family_aliases(text))

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
                    "workflow": takehome_workflow(
                        adaptive_search_memory=True,
                        adaptive_gate_controller=True,
                        adaptive_gate_min_family_attempts_for_hard=2,
                    ),
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
                    "workflow": takehome_workflow(
                        adaptive_search_memory=True,
                        adaptive_gate_controller=True,
                        adaptive_gate_min_family_attempts_for_hard=1,
                        adaptive_gate_all_skipped_relax_streak=2,
                    ),
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
                    "todo_soft_until_first_improvement": False,
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

            coder_messages = models.seen["coder"][0]
            joined = "\n".join(message["content"] for message in coder_messages)
            self.assertIn("Active durable todo follows", joined)
            self.assertIn("todo-001", joined)
            self.assertIn("Stagnation brainstorm tactics follow", joined)
            self.assertIn("tabula rasa data layout", joined)
            self.assertEqual([message["role"] for message in coder_messages], ["system", "user", "user"])
            self.assertNotIn("Active durable todo follows", coder_messages[1]["content"])
            self.assertIn("Active durable todo follows", coder_messages[-1]["content"])
            self.assertIn("Stagnation brainstorm tactics follow", coder_messages[-1]["content"])

    def test_adaptive_search_can_reject_cooled_axis_before_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": takehome_workflow(
                    writable_files=["target.py"],
                    test_commands=["python3 -c \"print('cycles: 80')\""],
                    candidate_queue=True,
                    adaptive_search_memory=True,
                    adaptive_search_reject_cooled_axes=True,
                    candidate_history_path=".local_micro_agent/candidates.jsonl",
                ),
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
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": takehome_workflow()},
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

    def test_axis_contract_allows_reason_drift_with_matching_declared_axis(self) -> None:
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
            self.assertNotIn("does not substantively target required", "\n".join(state.notes))
            history = (repo / ".local_micro_agent" / "candidates.jsonl").read_text()
            self.assertNotIn('"status": "rejected_axis_drift"', history)
            self.assertIn('"status": "rejected"', history)

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
                "workflow": takehome_workflow(
                    writable_files=["target.py"],
                    test_commands=["python3 -c \"print('cycles: 80')\""],
                    candidate_queue=True,
                    adaptive_search_memory=True,
                    adaptive_search_force_strategy_axis=True,
                    adaptive_search_axis_pool=["memory_store_layout"],
                    candidate_history_path=".local_micro_agent/candidates.jsonl",
                    failed_tactics_path=".local_micro_agent/failed_tactics.jsonl",
                ),
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
                        reason="family_key: store_address_reuse\nPhase 4 store address reuse",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "family_key: store_address_reuse\nPhase 4 store address reuse",
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
                "workflow": takehome_workflow(
                    adaptive_search_memory=True,
                    adaptive_search_force_strategy_axis=True,
                    adaptive_search_axis_pool=["vector_unroll_lane", "hash_build"],
                ),
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

            messages = code_prompt(state)
            stable_content = messages[1]["content"]
            dynamic_content = messages[2]["content"]

            self.assertIn("Source files:", stable_content)
            self.assertNotIn("Recent agent feedback:", stable_content)
            self.assertIn("Dynamic context for this CODE attempt", dynamic_content)
            self.assertIn("Latest test summary:", dynamic_content)
            self.assertIn("Recent agent feedback:", dynamic_content)

    def test_code_prompt_can_use_legacy_single_user_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            state.plan_markdown = "Plan text"
            state.file_context = [FileSnapshot("target.py", "value = 1\n")]
            state.notes.append("dynamic feedback")

            messages = code_prompt(state, cache_friendly_layout=False)

            self.assertEqual(len(messages), 2)
            self.assertIn("Source files:", messages[1]["content"])
            self.assertIn("Latest test summary:", messages[1]["content"])
            self.assertIn("dynamic feedback", messages[1]["content"])

    def test_code_attempt_includes_refreshed_writable_source_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'current'\n")
            config = {
                "models": {"default": "roles"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "plan_markdown": "seeded",
                    "seed_files": ["target.py"],
                    "writable_files": ["target.py"],
                    "prompt_cache_friendly_layout": True,
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
            state.file_context = [FileSnapshot("target.py", "value = 'stale'\n")]
            models = _RoleModelManager(
                {
                    "coder": (
                        '{"changes":[{"path":"target.py",'
                        '"target":"value = \'current\'\\n",'
                        '"replacement":"value = \'new\'\\n",'
                        '"reason":"edit current source"}]}'
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
            dynamic_content = coder_messages[-1]["content"]
            self.assertIn("Current writable source context follows", dynamic_content)
            self.assertIn("value = 'current'", dynamic_content)
            self.assertNotIn("value = 'stale'", dynamic_content)
            self.assertEqual(target.read_text(), "value = 'new'\n")

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
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": takehome_workflow()},
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
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": takehome_workflow()},
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
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": takehome_workflow()},
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
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": takehome_workflow()},
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

    def test_patch_application_failure_does_not_consume_todo_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            todo = {
                "todo_id": "todo-001-instruction_scheduling",
                "status": "active",
                "strategy_axis": "instruction_scheduling",
                "context": "narrow edit tactic",
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
                    "workflow": {
                        "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                        "todo_attempt_budget": 3,
                        "todo_soft_until_first_improvement": False,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = todo

            agent._append_candidate_history(
                CodeCandidate(
                    "1",
                    [CodeChange("target.py", "stale search", target="old", replacement="new")],
                    "same idea with stale search",
                    "instruction_scheduling",
                ),
                status="rejected_no_changes",
                metric=None,
                applied=0,
                failed=True,
                extra={
                    "no_change_reason": "Replacement target not found: target.py",
                    "failure_detail": "Replacement target not found: target.py",
                },
            )

            updated = json.loads((artifact_dir / "todo_plan.json").read_text())
            active = json.loads((artifact_dir / "active_todo.json").read_text())
            attempts = [
                json.loads(line)
                for line in (artifact_dir / "todo_attempts.jsonl").read_text().splitlines()
            ]
            self.assertEqual(updated["active_todo_id"], "todo-001-instruction_scheduling")
            self.assertEqual(updated["todos"][0].get("attempts", 0), 0)
            self.assertEqual(updated["todos"][0].get("patch_failures"), 1)
            self.assertEqual(active.get("patch_failures"), 1)
            self.assertFalse(attempts[0]["budget_counted"])

    def test_validated_pattern_followup_creates_active_todo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "candidates.jsonl").write_text(
                json.dumps(
                    {
                        "loop": 4,
                        "candidate_id": "1",
                        "status": "improved",
                        "metric": 90,
                        "strategy_axis": "memory_store_layout",
                        "strategy_axes": ["memory_store_layout"],
                        "family_aliases": ["store_address_reuse"],
                        "reason": "validated local redundancy removal",
                        "changes": [{"path": "target.py", "mode": "replacement"}],
                    }
                )
                + "\n"
            )
            state = AgentState(repo_root=repo, user_request="test")
            state.planned_files = ["target.py"]
            state.loop_count = 4
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": takehome_workflow(
                        candidate_history_path=".local_micro_agent/candidates.jsonl",
                        validated_pattern_followup=True,
                    ),
                },
                state,
            )

            agent._create_validated_pattern_followup_todo()

            plan = json.loads((artifact_dir / "todo_plan.json").read_text())
            active = json.loads((artifact_dir / "active_todo.json").read_text())
            self.assertEqual(plan["active_todo_id"], "todo-004-memory_store_layout-followup")
            self.assertEqual(active["strategy_axis"], "memory_store_layout")
            self.assertEqual(active["source"], "validated_pattern_followup")
            self.assertIn("validated local redundancy removal", active["context"])

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
                        "todo_soft_until_first_improvement": False,
                    },
                },
                AgentState(repo_root=repo, user_request="test", loop_count=2),
            )

            self.assertFalse(agent._should_brainstorm())

            todo["attempts"] = 3
            (artifact_dir / "active_todo.json").write_text(json.dumps(todo) + "\n")
            agent.state.scratch.pop("active_todo", None)
            self.assertTrue(agent._should_brainstorm())

    def test_soft_active_todo_still_blocks_brainstorm_before_first_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "candidates.jsonl").write_text(
                '{"status":"rejected","metric":147734,"failed":false}\n'
                '{"status":"rejected_todo_axis_drift","metric":147734,"failed":false}\n'
            )
            todo = {
                "todo_id": "todo-001-memory_store_layout",
                "status": "attempted",
                "strategy_axis": "memory_store_layout",
                "family_key": "store_address_reuse",
                "context": "store address reuse tactic",
                "attempts": 1,
            }
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "brainstorm_after_rejections": 2,
                        "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                        "todo_attempt_budget": 3,
                        "todo_soft_until_first_improvement": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test", loop_count=2),
            )
            agent.state.scratch["active_todo"] = todo
            drift_candidate = CodeCandidate(
                "soft-drift",
                [
                    CodeChange(
                        path="target.py",
                        reason="hazard-aware VLIW bundle scheduler",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "Implement a hazard-aware VLIW bundle scheduler.",
                strategy_axis="instruction_scheduling",
            )

            self.assertFalse(agent._has_active_todo_budget())
            self.assertTrue(agent._has_active_todo_brainstorm_budget())
            self.assertEqual(agent._active_todo_id(), "todo-001-memory_store_layout")
            self.assertFalse(agent._should_brainstorm())
            self.assertIsNone(agent._active_todo_contract_rejection(drift_candidate))

            agent.config["workflow"]["pre_improvement_todo_blocks_brainstorm"] = False
            self.assertFalse(agent._has_active_todo_brainstorm_budget())
            self.assertEqual(agent._active_todo_id(), "")
            self.assertTrue(agent._should_brainstorm())

            agent.config["workflow"]["pre_improvement_todo_blocks_brainstorm"] = True
            todo["attempts"] = 3
            self.assertFalse(agent._has_active_todo_brainstorm_budget())
            self.assertTrue(agent._should_brainstorm())

    def test_active_todo_is_soft_before_first_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            todo = {
                "todo_id": "todo-001-memory_store_layout",
                "status": "attempted",
                "strategy_axis": "memory_store_layout",
                "family_key": "store_address_reuse",
                "context": "store address reuse tactic",
                "attempts": 1,
            }
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                        "todo_attempt_budget": 3,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = todo
            candidate = CodeCandidate(
                "vliw-recovery",
                [
                    CodeChange(
                        path="target.py",
                        reason="hazard-aware VLIW bundle scheduler",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "Implement a hazard-aware VLIW bundle scheduler.",
                strategy_axis="instruction_scheduling",
            )

            self.assertFalse(agent._has_active_todo_budget())
            self.assertEqual(agent._format_active_todo(), "")
            self.assertEqual(agent._active_todo_id(), "todo-001-memory_store_layout")
            self.assertIsNone(agent._active_todo_contract_rejection(candidate))

    def test_active_todo_is_hard_after_first_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "candidates.jsonl").write_text(
                '{"status":"improved","metric":119062,"failed":false}\n'
            )
            todo = {
                "todo_id": "todo-001-memory_store_layout",
                "status": "attempted",
                "strategy_axis": "memory_store_layout",
                "family_key": "store_address_reuse",
                "context": "store address reuse tactic",
                "attempts": 1,
            }
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                        "todo_attempt_budget": 3,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = todo
            candidate = CodeCandidate(
                "late-drift",
                [
                    CodeChange(
                        path="target.py",
                        reason="hazard-aware VLIW bundle scheduler",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "Implement a hazard-aware VLIW bundle scheduler.",
                strategy_axis="instruction_scheduling",
            )

            rejection = agent._active_todo_contract_rejection(candidate)
            self.assertIsNotNone(rejection)
            self.assertEqual(rejection[0], "rejected_todo_axis_drift")

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
                "workflow": takehome_workflow(
                    writable_files=["target.py"],
                    test_commands=["python3 -c \"print('cycles: 80')\""],
                    candidate_queue=True,
                    candidate_history_path=".local_micro_agent/candidates.jsonl",
                    todo_attempt_budget=3,
                    todo_soft_until_first_improvement=False,
                ),
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

    def test_active_todo_contract_allows_axis_word_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            todo = {
                "todo_id": "todo-001-vector_unroll_lane",
                "status": "attempted",
                "strategy_axis": "vector_unroll_lane",
                "family_key": "unroll_factor_change",
                "context": "lane-level loop unrolling tactic",
                "attempts": 1,
            }
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {"todo_soft_until_first_improvement": False},
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = todo
            candidate = CodeCandidate(
                "variant",
                [
                    CodeChange(
                        path="target.py",
                        reason="process two lanes with loop unrolling",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "Implement loop unrolling by factor of 2 to process two lanes.",
                strategy_axis="vector_unroll_lane",
            )

            self.assertIsNone(agent._active_todo_contract_rejection(candidate))

    def test_active_todo_contract_trusts_matching_declared_dynamic_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            todo = {
                "todo_id": "todo-001-vliw_packing",
                "status": "attempted",
                "strategy_axis": "vliw_packing",
                "family_key": "",
                "context": "bundle scheduler tactic",
                "attempts": 1,
            }
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {"todo_soft_until_first_improvement": False},
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = todo
            candidate = CodeCandidate(
                "declared-axis",
                [
                    CodeChange(
                        path="target.py",
                        reason="basic hazard-aware VLIW bundle packer",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "Implement a hazard-aware VLIW bundle scheduler.",
                strategy_axis="vliw_packing",
            )

            self.assertIn("vliw_packing", agent._candidate_strategy_axes(candidate))
            self.assertNotIn(
                "vliw_packing", agent._candidate_reason_strategy_axes(candidate)
            )
            self.assertIsNone(agent._active_todo_contract_rejection(candidate))

    def test_axis_contract_trusts_matching_declared_dynamic_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "adaptive_search_force_strategy_axis": True,
                        "adaptive_search_axis_pool": ["multi_engine_slot_accumulation"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["required_strategy_axis"] = (
                "multi_engine_slot_accumulation"
            )
            candidate = CodeCandidate(
                "declared-axis",
                [
                    CodeChange(
                        path="target.py",
                        reason="basic hazard-aware VLIW bundle packer",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "Implement a hazard-aware VLIW bundle scheduler.",
                strategy_axis="multi_engine_slot_accumulation",
            )

            self.assertIn(
                "multi_engine_slot_accumulation",
                agent._candidate_strategy_axes(candidate),
            )
            self.assertNotIn(
                "multi_engine_slot_accumulation",
                agent._candidate_reason_strategy_axes(candidate),
            )
            self.assertIsNone(agent._candidate_axis_contract_rejection(candidate))

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
                "workflow": takehome_workflow(
                    writable_files=["target.py"],
                    test_commands=["python3 -c \"print('cycles: 80')\""],
                    candidate_queue=True,
                    candidate_history_path=".local_micro_agent/candidates.jsonl",
                    todo_attempt_budget=3,
                    todo_soft_until_first_improvement=False,
                ),
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            agent = MicroAgent(config, state)
            candidate = CodeCandidate(
                "family-drift",
                [
                    CodeChange(
                        path="target.py",
                        reason="family_key: hash_reorder\nphase stage hash reorder tmp1 tmp2",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "family_key: hash_reorder\nphase stage hash reorder tmp1 tmp2",
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

    def test_active_todo_duplicate_variant_rejected_before_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            last_attempt = {
                "loop": 4,
                "todo_id": "todo-001-phase_interleave",
                "candidate_id": "1",
                "status": "rejected",
                "metric": 147734,
                "failed": False,
                "strategy_axis": "phase_interleave",
                "reason": "phase pipeline interleave adjacent batch items",
            }
            todo = {
                "todo_id": "todo-001-phase_interleave",
                "status": "attempted",
                "strategy_axis": "phase_interleave",
                "family_key": "phase_pipeline",
                "context": "phase pipeline tactic",
                "attempts": 1,
                "last_attempt": last_attempt,
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
                "workflow": takehome_workflow(
                    writable_files=["target.py"],
                    test_commands=["python3 -c \"print('cycles: 80')\""],
                    candidate_queue=True,
                    candidate_history_path=".local_micro_agent/candidates.jsonl",
                    todo_attempt_budget=3,
                    todo_soft_until_first_improvement=False,
                ),
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            agent = MicroAgent(config, state)
            candidate = CodeCandidate(
                "duplicate",
                [
                    CodeChange(
                        path="target.py",
                        reason="phase pipeline interleave adjacent batch items",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "phase pipeline interleave adjacent batch items",
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
            attempts = (artifact_dir / "todo_attempts.jsonl").read_text()
            updated = json.loads((artifact_dir / "todo_plan.json").read_text())
            self.assertIn('"status": "rejected_todo_duplicate_variant"', history)
            self.assertIn("repeats rejected variant", attempts)
            self.assertEqual(updated["active_todo_id"], "todo-001-phase_interleave")
            self.assertEqual(updated["todos"][0]["status"], "attempted")
            self.assertEqual(updated["todos"][0]["attempts"], 2)

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
