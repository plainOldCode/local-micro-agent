from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

import local_micro_agent.models as model_module
from local_micro_agent.orchestrator import CodeCandidate, CodeDecision, MicroAgent, ReadDecision
from local_micro_agent.models import (
    ModelResponse,
    OllamaNativeModel,
    OpenAICompatibleModel,
    _ollama_usage,
    _openai_usage,
)
from local_micro_agent.prompts import (
    brainstorm_prompt,
    code_prompt,
    read_prompt,
    reflect_prompt,
    semantic_analysis_prompt,
    spec_hypothesis_repair_prompt,
    spec_idea_prompt,
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
from local_micro_agent.validators import JsonValidationError


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


class _SpecFallbackModelManager:
    def __init__(self, fallback_output: str):
        self.fallback_output = fallback_output

    def get(self, role):
        if role == "reasoner":
            return _FailingModel()
        return _StaticModel(self.fallback_output)


class _SpecParseFallbackModelManager:
    def __init__(self, fallback_output: str):
        self.fallback_output = fallback_output

    def get(self, role):
        if role == "reasoner":
            return _StaticModel("not json")
        return _StaticModel(self.fallback_output)


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


class _ReasoningOnlyModel:
    async def chat(self, messages):
        return ModelResponse(
            "",
            usage={
                "reasoning_only_response": True,
                "reasoning_content_chars": 1024,
                "completion_tokens": 4096,
            },
        )


class _ReasoningOnlyModelManager:
    def get(self, role):
        return _ReasoningOnlyModel()


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


class _ReasoningStreamingModel:
    supports_streaming = True

    async def chat(self, messages, stream_callback=None):
        if stream_callback is not None:
            stream_callback({"kind": "reasoning", "content": "think-a"})
            stream_callback({"kind": "reasoning", "content": "think-b"})
            stream_callback('{"changes":')
            stream_callback("[]}")
        return ModelResponse(
            '{"changes":[]}',
            usage={"reasoning_content_chars": len("think-athink-b")},
        )


class _ReasoningStreamingModelManager:
    def get(self, role):
        return _ReasoningStreamingModel()


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


class _RoleSequenceModel:
    def __init__(
        self,
        outputs: dict[str, list[str]],
        seen: dict[str, list[list[dict[str, str]]]],
        role: str,
    ):
        self.outputs = outputs
        self.seen = seen
        self.role = role

    async def chat(self, messages):
        self.seen.setdefault(self.role, []).append(messages)
        outputs = self.outputs[self.role]
        if len(outputs) > 1:
            return outputs.pop(0)
        return outputs[0]


class _RoleSequenceModelManager:
    def __init__(self, outputs: dict[str, list[str]]):
        self.outputs = outputs
        self.seen: dict[str, list[list[dict[str, str]]]] = {}

    def get(self, role):
        return _RoleSequenceModel(self.outputs, self.seen, role)


class _SpecIdeaReasoningOnlyManager:
    def __init__(self, finalizer_output: str):
        self.finalizer_output = finalizer_output
        self.seen: dict[str, int] = {}

    def get(self, role):
        self.seen[role] = self.seen.get(role, 0) + 1
        if role == "reasoner":
            return _ReasoningOnlyModel()
        return _StaticModel(self.finalizer_output)


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

    def test_openai_compatible_model_sends_think_and_extra_body(self) -> None:
        captured = {}
        original = model_module._post_json

        def fake_post_json(url, payload, headers, timeout):
            captured.update(
                {"url": url, "payload": payload, "headers": headers, "timeout": timeout}
            )
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }

        model_module._post_json = fake_post_json
        try:
            response = asyncio.run(
                OpenAICompatibleModel(
                    base_url="http://localhost:1234/v1",
                    model="local",
                    api_key_env="MISSING_API_KEY",
                    temperature=0.2,
                    max_tokens=42,
                    timeout_seconds=7,
                    think=False,
                    extra_body={"enableThinking": False, "custom": {"field": 1}},
                ).chat([{"role": "user", "content": "hello"}])
            )
        finally:
            model_module._post_json = original

        self.assertEqual(response.content, "ok")
        self.assertEqual(captured["url"], "http://localhost:1234/v1/chat/completions")
        self.assertEqual(captured["timeout"], 7)
        self.assertEqual(captured["headers"]["Authorization"], "Bearer local")
        self.assertEqual(captured["payload"]["model"], "local")
        self.assertEqual(captured["payload"]["temperature"], 0.2)
        self.assertEqual(captured["payload"]["max_tokens"], 42)
        self.assertFalse(captured["payload"]["think"])
        self.assertFalse(captured["payload"]["enable_thinking"])
        self.assertFalse(captured["payload"]["enableThinking"])
        self.assertEqual(captured["payload"]["custom"], {"field": 1})

    def test_ollama_native_model_sends_top_level_format(self) -> None:
        captured = {}
        original = model_module._post_json

        def fake_post_json(url, payload, headers, timeout):
            captured.update(
                {"url": url, "payload": payload, "headers": headers, "timeout": timeout}
            )
            return {
                "message": {"content": "{}"},
                "prompt_eval_count": 1,
                "eval_count": 1,
            }

        model_module._post_json = fake_post_json
        try:
            response = asyncio.run(
                OllamaNativeModel(
                    base_url="http://localhost:11434",
                    model="local",
                    temperature=0.7,
                    max_tokens=128,
                    num_ctx=4096,
                    think=False,
                    output_format="json",
                    timeout_seconds=9,
                    extra_options={"top_p": 0.8},
                ).chat([{"role": "user", "content": "json"}])
            )
        finally:
            model_module._post_json = original

        self.assertEqual(response.content, "{}")
        self.assertEqual(captured["url"], "http://localhost:11434/api/chat")
        self.assertEqual(captured["timeout"], 9)
        self.assertEqual(captured["payload"]["format"], "json")
        self.assertEqual(captured["payload"]["options"]["num_predict"], 128)
        self.assertEqual(captured["payload"]["options"]["num_ctx"], 4096)
        self.assertEqual(captured["payload"]["options"]["top_p"], 0.8)
        self.assertNotIn("format", captured["payload"]["options"])

    def test_openai_compatible_model_can_prefill_disabled_thinking(self) -> None:
        captured = {}
        original = model_module._post_json
        messages = [{"role": "user", "content": "hello"}]

        def fake_post_json(url, payload, headers, timeout):
            captured.update({"payload": payload})
            return {"choices": [{"message": {"content": "ok"}}]}

        model_module._post_json = fake_post_json
        try:
            response = asyncio.run(
                OpenAICompatibleModel(
                    base_url="http://localhost:1234/v1",
                    model="local",
                    think=False,
                    disable_thinking_with_assistant_prefill=True,
                ).chat(messages)
            )
        finally:
            model_module._post_json = original

        self.assertEqual(response.content, "ok")
        self.assertEqual(messages, [{"role": "user", "content": "hello"}])
        self.assertEqual(
            captured["payload"]["messages"],
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "<think>\n\n</think>\n\n"},
            ],
        )

    def test_openai_compatible_model_uses_streaming_helper(self) -> None:
        captured = {}
        original = model_module._post_openai_stream

        def fake_stream(url, payload, headers, timeout, stream_callback):
            captured.update({"url": url, "payload": payload, "timeout": timeout})
            stream_callback("he")
            stream_callback("llo")
            return ModelResponse(
                "hello",
                usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            )

        model_module._post_openai_stream = fake_stream
        try:
            chunks: list[str] = []
            response = asyncio.run(
                OpenAICompatibleModel(
                    base_url="http://localhost:1234/v1",
                    model="local",
                    think=True,
                ).chat(
                    [{"role": "user", "content": "hello"}],
                    stream_callback=chunks.append,
                )
            )
        finally:
            model_module._post_openai_stream = original

        self.assertEqual(response.content, "hello")
        self.assertEqual(chunks, ["he", "llo"])
        self.assertEqual(captured["url"], "http://localhost:1234/v1/chat/completions")
        self.assertTrue(captured["payload"]["think"])

    def test_ollama_stream_captures_thinking_separately(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __iter__(self):
                rows = [
                    {"message": {"thinking": "think-a"}},
                    {"message": {"thinking": "think-b", "content": "ok"}},
                    {"done": True, "prompt_eval_count": 2, "eval_count": 3},
                ]
                return iter((json.dumps(row).encode("utf-8") for row in rows))

        original = model_module.urllib.request.urlopen
        model_module.urllib.request.urlopen = lambda request, timeout: FakeResponse()
        try:
            chunks: list[object] = []
            response = model_module._post_ollama_stream(
                "http://localhost:11434/api/chat",
                {"model": "fake", "messages": [], "stream": True},
                {},
                30,
                chunks.append,
            )
        finally:
            model_module.urllib.request.urlopen = original

        self.assertEqual(response.content, "ok")
        self.assertEqual(
            chunks,
            [
                {"kind": "reasoning", "content": "think-a"},
                {"kind": "reasoning", "content": "think-b"},
                {"kind": "content", "content": "ok"},
            ],
        )
        self.assertEqual(response.usage["reasoning_content_chars"], len("think-athink-b"))
        self.assertEqual(response.reasoning, "think-athink-b")
        self.assertFalse(response.usage["reasoning_only_response"])
        self.assertEqual(response.usage["completion_tokens"], 3)

    def test_openai_stream_marks_reasoning_only_response(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __iter__(self):
                rows = [
                    'data: {"choices":[{"delta":{"reasoning_content":"think-a"}}]}',
                    'data: {"choices":[{"delta":{"reasoning_content":"think-b"}}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}',
                    "data: [DONE]",
                ]
                return iter((f"{row}\n".encode("utf-8") for row in rows))

        original = model_module.urllib.request.urlopen
        model_module.urllib.request.urlopen = lambda request, timeout: FakeResponse()
        try:
            chunks: list[object] = []
            response = model_module._post_openai_stream(
                "http://localhost:1234/v1/chat/completions",
                {"model": "fake", "messages": [], "stream": True},
                {},
                30,
                chunks.append,
            )
        finally:
            model_module.urllib.request.urlopen = original

        self.assertEqual(response.content, "")
        self.assertEqual(
            chunks,
            [
                {"kind": "reasoning", "content": "think-a"},
                {"kind": "reasoning", "content": "think-b"},
            ],
        )
        self.assertEqual(response.usage["reasoning_content_chars"], len("think-athink-b"))
        self.assertEqual(response.reasoning, "think-athink-b")
        self.assertTrue(response.usage["reasoning_only_response"])
        self.assertEqual(response.usage["completion_tokens"], 3)

    def test_openai_stream_marks_whitespace_final_as_reasoning_only(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __iter__(self):
                rows = [
                    'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}',
                    'data: {"choices":[{"delta":{"content":"  \\n\\t"}}],"usage":{"completion_tokens":3}}',
                    "data: [DONE]",
                ]
                return iter((f"{row}\n".encode("utf-8") for row in rows))

        original = model_module.urllib.request.urlopen
        model_module.urllib.request.urlopen = lambda request, timeout: FakeResponse()
        try:
            response = model_module._post_openai_stream(
                "http://localhost:1234/v1/chat/completions",
                {"model": "fake", "messages": [], "stream": True},
                {},
                30,
                None,
            )
        finally:
            model_module.urllib.request.urlopen = original

        self.assertEqual(response.content, "  \n\t")
        self.assertTrue(response.usage["reasoning_only_response"])

    def test_openai_non_stream_marks_whitespace_final_as_reasoning_only(self) -> None:
        original = model_module._post_json

        def fake_post_json(url, payload, headers, timeout):
            return {
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "think",
                            "content": " \n",
                        }
                    }
                ],
                "usage": {"completion_tokens": 3},
            }

        model_module._post_json = fake_post_json
        try:
            response = asyncio.run(
                OpenAICompatibleModel(
                    base_url="http://localhost:1234/v1",
                    model="fake",
                ).chat([])
            )
        finally:
            model_module._post_json = original

        self.assertEqual(response.content, " \n")
        self.assertEqual(response.reasoning, "think")
        self.assertTrue(response.usage["reasoning_only_response"])

    def test_model_token_budget_fields_warn_near_input_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {"prompt_token_budget_warn_ratio": 0.9},
            }
            agent = MicroAgent(config, state)

            fields = agent._model_token_budget_fields(
                {"num_ctx": 100, "max_tokens": 20},
                {"prompt_tokens": 75},
                prompt_chars=0,
                role="coder",
                call_site="json_call",
            )

            self.assertEqual(fields["input_token_budget"], 80)
            self.assertTrue(fields["input_token_budget_warning"])
            self.assertIn("Prompt token budget pressure", "\n".join(state.notes))
            self.assertEqual(len(state.scratch["prompt_token_budget_warnings"]), 1)

            agent._model_token_budget_fields(
                {"num_ctx": 100, "max_tokens": 20},
                {"prompt_tokens": 76},
                prompt_chars=0,
                role="coder",
                call_site="json_call",
            )

            self.assertEqual(
                "\n".join(state.notes).count("Prompt token budget pressure"),
                1,
            )
            self.assertEqual(len(state.scratch["prompt_token_budget_warnings"]), 2)

    def test_dynamic_suffix_blocks_shrink_to_input_token_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            config = {
                "models": {"coder": "local"},
                "providers": {
                    "local": {
                        "kind": "ollama_native",
                        "num_ctx": 100,
                        "max_tokens": 20,
                    }
                },
                "mcp_servers": {},
                "workflow": {
                    "prompt_chars_per_token_estimate": 1,
                    "prompt_token_budget_target_ratio": 0.5,
                },
            }
            agent = MicroAgent(config, state)
            blocks = ["a" * 1000, "b" * 1000]

            shrunk = agent._shrink_dynamic_suffix_blocks(
                [{"role": "system", "content": "stable"}],
                blocks,
                role="coder",
            )

            self.assertLess(sum(len(block) for block in shrunk), sum(len(block) for block in blocks))
            self.assertIn("Shrank dynamic CODE context", "\n".join(state.notes))
            self.assertEqual(len(state.scratch["prompt_token_budget_warnings"]), 1)

    def test_openai_stream_payload_preserves_include_usage_default(self) -> None:
        payload = {
            "model": "local",
            "messages": [],
            "stream_options": {"foo": "bar"},
        }

        stream_payload = model_module._openai_stream_payload(payload)

        self.assertTrue(stream_payload["stream"])
        self.assertEqual(
            stream_payload["stream_options"],
            {"foo": "bar", "include_usage": True},
        )
        self.assertEqual(payload["stream_options"], {"foo": "bar"})

    def test_openai_stream_payload_allows_include_usage_override(self) -> None:
        payload = {
            "model": "local",
            "messages": [],
            "stream_options": {"include_usage": False},
        }

        stream_payload = model_module._openai_stream_payload(payload)

        self.assertEqual(stream_payload["stream_options"], {"include_usage": False})

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
                    result = await agent._apply_changes([change], {"target.py"})
                    return result.applied
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
                    result = await agent._apply_changes([change], {"target.py"})
                    return result.applied
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

    def test_json_call_rejects_reasoning_only_response(self) -> None:
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
            agent.models = _ReasoningOnlyModelManager()

            with self.assertRaises(JsonValidationError):
                asyncio.run(
                    agent._json_call(
                        "planner",
                        [{"role": "user", "content": "choose files"}],
                        schema=ReadDecision,
                    )
                )

            self.assertIn("Rejected reasoning-only", "\n".join(state.notes))
            profile_path = repo / ".local_micro_agent" / "profile_events.jsonl"
            rows = [
                json.loads(line)
                for line in profile_path.read_text().splitlines()
                if line.strip()
            ]
            model_event = next(row for row in rows if row["event_type"] == "model_call")
            self.assertFalse(model_event["success"])
            self.assertTrue(model_event["reasoning_only_response"])
            self.assertTrue(model_event["rejected_reasoning_only"])
            self.assertIn("reasoning-only", model_event["error"])

    def test_reasoning_only_allowed_call_site_bypasses_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            state = AgentState(repo_root=repo, user_request="test")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {"reasoning_only_allowed_call_sites": ["plan"]},
            }
            agent = MicroAgent(config, state)
            agent.models = _ReasoningOnlyModelManager()

            output = asyncio.run(
                agent._model_chat(
                    "planner",
                    [{"role": "user", "content": "plan"}],
                    call_site="plan",
                )
            )

            self.assertEqual(output, "")
            self.assertNotIn("Rejected reasoning-only", "\n".join(state.notes))

    def test_plan_falls_back_after_reasoning_only_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            state = AgentState(repo_root=repo, user_request="test")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["target.py"],
                    "test_commands": ["python3 -m pytest -q"],
                },
            }
            agent = MicroAgent(config, state)
            agent.models = _ReasoningOnlyModelManager()

            asyncio.run(agent.plan())

            self.assertEqual(state.current, AgentStateName.READ)
            self.assertIn("# Fallback Plan", state.plan_markdown)
            self.assertIn("PLAN model call failed", "\n".join(state.notes))

    def test_read_falls_back_to_configured_files_after_json_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("value = 1\n")
            state = AgentState(repo_root=repo, user_request="test")
            state.plan_markdown = "Plan"
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {"writable_files": ["target.py"]},
            }
            agent = MicroAgent(config, state)
            agent.models = _BadJsonModelManager()

            async def read_once() -> None:
                await agent.mcp.start()
                try:
                    await agent.read()
                finally:
                    await agent.mcp.close()

            asyncio.run(read_once())

            self.assertEqual(state.planned_files, ["target.py"])
            self.assertEqual([snap.path for snap in state.file_context], ["target.py"])
            self.assertIn("READ decision failed", "\n".join(state.notes))

    def test_read_notes_empty_fallback_after_json_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            state = AgentState(repo_root=repo, user_request="test")
            state.plan_markdown = "Plan"
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {},
            }
            agent = MicroAgent(config, state)
            agent.models = _BadJsonModelManager()

            async def read_once() -> None:
                await agent.mcp.start()
                try:
                    await agent.read()
                finally:
                    await agent.mcp.close()

            asyncio.run(read_once())

            self.assertEqual(state.planned_files, [])
            self.assertIn("READ fallback file list is empty", "\n".join(state.notes))

    def test_test_decision_falls_back_to_pytest_result_after_json_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            state = AgentState(
                repo_root=repo,
                user_request="test",
                current=AgentStateName.TEST,
                max_loops=1,
            )
            state.scratch["applied_changes"] = 1
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "test_commands": ["python3 -c \"print('ok')\""],
                },
            }
            agent = MicroAgent(config, state)
            agent.models = _BadJsonModelManager()

            async def test_once() -> None:
                await agent.mcp.start()
                try:
                    await agent.test()
                finally:
                    await agent.mcp.close()

            asyncio.run(test_once())

            self.assertEqual(state.current, AgentStateName.DONE)
            self.assertIn("TEST decision failed", "\n".join(state.notes))

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

    def test_call_site_model_override_keeps_reflect_on_fast_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            config = {
                "models": {
                    "default": "exact",
                    "reflector": "reflect-fast",
                    "reasoner": "deep-reason",
                },
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "reasoning_lane_enabled": True,
                    "reasoning_lane_call_sites": ["plan", "semantic_analysis", "reflect"],
                    "model_role_overrides_by_call_site": {"reflect": "reflector"},
                },
            }
            agent = MicroAgent(config, AgentState(repo_root=repo, user_request="test"))
            models = _RoleModelManager(
                {
                    "reflector": "fast reflect",
                    "reasoner": "deep reasoning",
                }
            )
            agent.models = models

            asyncio.run(
                agent._model_chat(
                    "reflector", [{"role": "user", "content": "reflect"}], call_site="reflect"
                )
            )

            self.assertEqual(len(models.seen["reflector"]), 1)
            self.assertNotIn("reasoner", models.seen)

    def test_call_site_model_override_can_pin_plan_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            config = {
                "models": {
                    "default": "exact",
                    "planner": "planner-default",
                    "plan_deep": "plan-deep",
                    "reasoner": "deep-reason",
                },
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "reasoning_lane_enabled": True,
                    "reasoning_lane_call_sites": ["plan"],
                    "reasoning_lane_model_role": "reasoner",
                    "model_role_overrides_by_call_site": {"plan": "plan_deep"},
                },
            }
            agent = MicroAgent(config, AgentState(repo_root=repo, user_request="test"))
            models = _RoleModelManager(
                {
                    "plan_deep": "plan output",
                    "reasoner": "deep reasoning",
                    "planner": "default planner",
                }
            )
            agent.models = models

            asyncio.run(
                agent._model_chat(
                    "planner", [{"role": "user", "content": "plan"}], call_site="plan"
                )
            )

            self.assertEqual(len(models.seen["plan_deep"]), 1)
            self.assertNotIn("reasoner", models.seen)
            self.assertNotIn("planner", models.seen)

    def test_deep_reasoning_escalates_reflect_after_repeated_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            config = {
                "models": {
                    "default": "exact",
                    "reflector": "reflect-fast",
                    "reasoner": "deep-reason",
                },
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "reasoning_lane_enabled": True,
                    "reasoning_lane_call_sites": ["plan", "semantic_analysis"],
                    "model_role_overrides_by_call_site": {"reflect": "reflector"},
                    "deep_reasoning_enabled": True,
                    "deep_reasoning_model_role": "reasoner",
                    "deep_reasoning_call_sites": ["reflect"],
                    "deep_reasoning_after_same_failure_class": 3,
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["retry_failure_class_counts"] = {"correctness_failure": 3}
            agent = MicroAgent(config, state)
            models = _RoleModelManager(
                {
                    "reflector": "fast reflect",
                    "reasoner": "deep reasoning",
                }
            )
            agent.models = models

            asyncio.run(
                agent._model_chat(
                    "reflector", [{"role": "user", "content": "reflect"}], call_site="reflect"
                )
            )

            self.assertEqual(len(models.seen["reasoner"]), 1)
            self.assertNotIn("reflector", models.seen)
            self.assertIn("Escalating model call", "\n".join(state.notes))

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

    def test_profile_agent_records_reasoning_stream_separately(self) -> None:
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
            agent.models = _ReasoningStreamingModelManager()

            decision = asyncio.run(
                agent._json_call(
                    "coder",
                    [{"role": "user", "content": "make no changes"}],
                    schema=CodeDecision,
                )
            )

            self.assertIsNotNone(decision)
            stream_files = sorted((repo / ".local_micro_agent" / "model_streams").glob("*.txt"))
            reasoning_files = sorted(
                (repo / ".local_micro_agent" / "model_streams").glob("*.reasoning.txt")
            )
            content_files = [
                path for path in stream_files if not path.name.endswith(".reasoning.txt")
            ]
            self.assertEqual(len(content_files), 1)
            self.assertEqual(len(reasoning_files), 1)
            self.assertEqual(content_files[0].read_text(), '{"changes":[]}')
            self.assertEqual(reasoning_files[0].read_text(), "think-athink-b")
            profile_path = repo / ".local_micro_agent" / "profile_events.jsonl"
            rows = [
                json.loads(line)
                for line in profile_path.read_text().splitlines()
                if line.strip()
            ]
            model_event = next(row for row in rows if row["event_type"] == "model_call")
            self.assertEqual(model_event["stream_chunks"], 2)
            self.assertEqual(model_event["reasoning_stream_chunks"], 2)
            self.assertEqual(model_event["reasoning_stream_chars"], len("think-athink-b"))
            self.assertEqual(model_event["reasoning_content_chars"], len("think-athink-b"))

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

    def test_read_cannot_promote_external_context_via_path_alias(self) -> None:
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
            agent.models = _StaticModelManager('{"files":["hints/../hints/perf.md"]}')

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

    def test_spec_hypothesis_repair_prompt_targets_validation_issues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="optimize")
            state.plan_markdown = "plan"
            messages = spec_hypothesis_repair_prompt(
                state,
                brief="BEGIN_HYPOTHESIS_OPTION hyp\nchange_boundary.kind: local_edit\nEND_HYPOTHESIS_OPTION",
                options_payload={
                    "accepted_count": 0,
                    "rejected": [
                        {
                            "hypothesis_id": "hyp",
                            "issues": [
                                "structural_hypothesis_boundary_kind_mismatch",
                                "unresolved_or_non_writable_boundary:target.py::foo (note)",
                            ],
                        }
                    ],
                },
                focus="allowed_target_regions: target.py::foo",
            )

            content = messages[1]["content"]
            self.assertIn("Fix every validation issue", content)
            self.assertIn("structural_hypothesis_boundary_kind_mismatch", content)
            self.assertIn("structural_probe|structural_expand", content)
            self.assertIn("exact deterministic grounding regions", content)
            self.assertIn("do not add parenthetical notes", content)

    def test_change_targets_unique_class_method_from_unqualified_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            source = (
                "class KernelBuilder:\n"
                "    def build_hash(self):\n"
                "        return 1\n"
            )
            (repo / "target.py").write_text(source)
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=repo, user_request="test"),
            )
            change = CodeChange(
                path="target.py",
                target="    def build_hash(self):\n        return 1",
                replacement="    def build_hash(self):\n        return 2",
                reason="edit build_hash",
            )

            self.assertTrue(agent._change_targets_any_symbol(change, ["build_hash"]))

    def test_change_targets_unqualified_class_method_requires_unique_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            source = (
                "class A:\n"
                "    def build_hash(self):\n"
                "        return 1\n"
                "class B:\n"
                "    def build_hash(self):\n"
                "        return 2\n"
            )
            (repo / "target.py").write_text(source)
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=repo, user_request="test"),
            )
            change = CodeChange(
                path="target.py",
                target="    def build_hash(self):\n        return 1",
                replacement="    def build_hash(self):\n        return 3",
                reason="edit build_hash",
            )

            self.assertFalse(agent._change_targets_any_symbol(change, ["build_hash"]))

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

    def test_spec_mode_schedules_v2_tasks_to_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "run_spec.json").write_text(
                json.dumps(
                    {
                        "version": 2,
                        "spec_id": "two-step",
                        "objective": "Create two deliverables in order.",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "write a",
                                "deliverables": ["a.txt"],
                                "read_hints": ["a.txt"],
                                "acceptance": {
                                    "kind": "command",
                                    "commands": [
                                        "python3 -c \"from pathlib import Path; assert Path('a.txt').read_text() == 'done-a'\""
                                    ],
                                },
                                "budget": {"attempts_max": 2},
                            },
                            {
                                "task_id": "task-002",
                                "title": "write b",
                                "depends_on": ["task-001"],
                                "deliverables": ["b.txt"],
                                "read_hints": ["b.txt"],
                                "acceptance": {
                                    "kind": "command",
                                    "commands": [
                                        "python3 -c \"from pathlib import Path; assert Path('b.txt').read_text() == 'done-b'\""
                                    ],
                                },
                                "budget": {"attempts_max": 2},
                            },
                        ],
                    }
                )
                + "\n"
            )

            result = run_agent(
                repo,
                {
                    "spec_mode": True,
                    "run_spec_enabled": True,
                    "run_spec_path": ".local_micro_agent/run_spec.json",
                    "max_code_test_loops": 5,
                    "writable_files": ["*.txt"],
                    "seed_files": [],
                    "seed_changes": [
                        {"path": "a.txt", "content": "done-a"},
                        {"path": "b.txt", "content": "done-b"},
                    ],
                },
            )

            self.assertEqual(result.current, AgentStateName.DONE)
            self.assertEqual((repo / "a.txt").read_text(), "done-a")
            self.assertEqual((repo / "b.txt").read_text(), "done-b")
            spec = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual(spec["progress"], {"total": 2, "closed": 2, "deferred": 0, "failed": 0})
            self.assertEqual([task["status"] for task in spec["task_graph"]], ["closed", "closed"])
            self.assertEqual(spec["task_graph"][0]["budget"]["attempts_used"], 1)
            self.assertEqual(spec["task_graph"][1]["budget"]["attempts_used"], 1)

    def test_spec_mode_closes_context_only_task_without_code_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "Readme.md").write_text("context\n")
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "run_spec.json").write_text(
                json.dumps(
                    {
                        "version": 2,
                        "spec_id": "context-then-code",
                        "objective": "Read context then write deliverable.",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "read context",
                                "read_hints": ["Readme.md"],
                                "deliverables": [],
                                "acceptance": {"kind": "metric", "commands": []},
                                "budget": {"attempts_max": 1},
                            },
                            {
                                "task_id": "task-002",
                                "title": "write target",
                                "depends_on": ["task-001"],
                                "deliverables": ["target.txt"],
                                "read_hints": ["target.txt"],
                                "acceptance": {
                                    "kind": "command",
                                    "commands": [
                                        "python3 -c \"from pathlib import Path; assert Path('target.txt').read_text() == 'done'\""
                                    ],
                                },
                                "budget": {"attempts_max": 1},
                            },
                        ],
                    }
                )
                + "\n"
            )

            result = run_agent(
                repo,
                {
                    "spec_mode": True,
                    "run_spec_enabled": True,
                    "run_spec_path": ".local_micro_agent/run_spec.json",
                    "max_code_test_loops": 3,
                    "writable_files": ["*.txt"],
                    "seed_files": [],
                    "seed_changes": [{"path": "target.txt", "content": "done"}],
                },
            )

            self.assertEqual(result.current, AgentStateName.DONE)
            self.assertEqual(result.loop_count, 1)
            spec = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual([task["status"] for task in spec["task_graph"]], ["closed", "closed"])
            self.assertEqual(spec["task_graph"][0]["budget"]["attempts_used"], 0)
            self.assertEqual(spec["task_graph"][0]["decision_hint"], "context_only")
            self.assertIn("Closed context-only spec task: task-001", "\n".join(result.notes))
            progress_events = (artifact_dir / "spec_progress.jsonl").read_text()
            self.assertIn('"reason": "context_only"', progress_events)

    def test_spec_metric_task_requires_improvement_before_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "run_spec.json").write_text(
                json.dumps(
                    {
                        "version": 2,
                        "spec_id": "metric-task",
                        "objective": "Improve the measured cycle metric.",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "optimize target",
                                "deliverables": ["target.py"],
                                "acceptance": {"kind": "metric", "commands": []},
                                "budget": {"attempts_max": 3},
                            }
                        ],
                    }
                )
                + "\n"
            )

            result = run_agent(
                repo,
                {
                    "spec_mode": True,
                    "run_spec_enabled": True,
                    "run_spec_path": ".local_micro_agent/run_spec.json",
                    "max_code_test_loops": 1,
                    "writable_files": ["target.py"],
                    "seed_files": [],
                    "seed_changes": [
                        {
                            "path": "target.py",
                            "target": "value = 'old'\n",
                            "replacement": "value = 'new'\n",
                        }
                    ],
                    "test_commands": ["python3 -c \"print('cycles: 100')\""],
                    "metric_regex": r"cycles: (\d+)",
                    "baseline_metric": 100,
                },
            )

            self.assertEqual(result.current, AgentStateName.FAILED)
            self.assertEqual(target.read_text(), "value = 'old'\n")
            spec = json.loads((artifact_dir / "run_spec.json").read_text())
            task = spec["task_graph"][0]
            self.assertEqual(task["status"], "in_progress")
            self.assertEqual(spec["last_stop_reason"], "max_code_test_loops")
            self.assertEqual(task["last_observation"]["metric"], 100)
            self.assertEqual(task["last_observation"]["baseline"], 100)
            self.assertFalse(task["last_observation"]["improved"])
            self.assertEqual(task["last_observation"]["failure_class"], "no_improvement")
            self.assertIn("metric_no_improvement", task["decision_hint"])
            report = (artifact_dir / "spec_report.md").read_text()
            self.assertIn("stop_reason: `max_code_test_loops`", report)
            notes = "\n".join(result.notes)
            self.assertIn("Spec metric task requires improvement before close", notes)

    def test_candidate_history_recovers_metric_plateau_task_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "active_todo.json").write_text(
                json.dumps(
                    {
                        "todo_id": "task-002",
                        "spec_task_id": "task-002",
                        "status": "active",
                        "tactic_stage": "local_edit",
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
                        "spec_mode": True,
                        "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                        "todo_attempts_path": ".local_micro_agent/todo_attempts.jsonl",
                        "active_todo_path": ".local_micro_agent/active_todo.json",
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            candidate = CodeCandidate(
                "neutral",
                [
                    CodeChange(
                        "target.py",
                        "metric-neutral helper call",
                        target="value = old()\n",
                        replacement="value = new()\n",
                    )
                ],
                "correct but no metric gain",
                strategy_axis="general_edit",
            )
            agent.state.scratch["metric_acceptance"] = {
                "requires_improvement": True,
                "metric": 147734,
                "baseline": 147734,
                "improved": False,
                "failed": True,
                "failure_class": "no_improvement",
            }

            agent._append_candidate_history(
                candidate,
                status="rejected",
                metric=147734,
                applied=1,
                failed=True,
                extra={
                    "failure_class": "no_improvement",
                    "summary": "Metric candidate=147734 baseline=147734 improved=False.",
                    "tactic_stage": "local_edit",
                },
            )

            record = json.loads((artifact_dir / "candidates.jsonl").read_text())
            attempt = json.loads((artifact_dir / "todo_attempts.jsonl").read_text())
            self.assertEqual(record["todo_id"], "task-002")
            self.assertEqual(record["spec_task_id"], "task-002")
            self.assertEqual(record["spec_task_identity_source"], "active_todo_file")
            self.assertEqual(attempt["todo_id"], "task-002")
            self.assertEqual(attempt["spec_task_id"], "task-002")
            metric_acceptance = agent.state.scratch["metric_acceptance"]
            self.assertTrue(metric_acceptance["candidate_transition_bound"])
            self.assertEqual(metric_acceptance["candidate_id"], "neutral")
            self.assertEqual(metric_acceptance["loop"], 0)
            self.assertEqual(metric_acceptance["candidate_failure_class"], "no_improvement")
            self.assertEqual(metric_acceptance["spec_task_id"], "task-002")

    def test_metric_neutral_plateau_repeats_defer_task(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                artifact_dir = repo / ".local_micro_agent"
                artifact_dir.mkdir()
                task = {
                    "task_id": "task-001",
                    "title": "optimize target",
                    "status": "in_progress",
                    "target_regions": ["target.py::build"],
                    "tactic_stage": "local_edit",
                    "deliverables": ["target.py"],
                    "acceptance": {"kind": "metric"},
                    "budget": {"attempts_max": 4},
                }
                spec = {
                    "version": 2,
                    "spec_id": "metric-task",
                    "objective": "Improve cycles.",
                    "active_task_id": "task-001",
                    "task_graph": [task],
                }
                agent = MicroAgent(
                    {
                        "models": {},
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_progress_path": ".local_micro_agent/spec_progress.jsonl",
                            "failure_signatures_path": ".local_micro_agent/failure_signatures.jsonl",
                            "spec_metric_neutral_plateau_same_fingerprint_limit": 2,
                            "spec_task_attempt_budget": 4,
                        },
                    },
                    AgentState(repo_root=repo, user_request="test", max_loops=10),
                )
                agent.state.scratch["run_spec"] = spec
                agent.state.scratch["current_spec_task_id"] = "task-001"
                agent.state.scratch["current_spec_task"] = task
                agent.state.test_results = [
                    TestResult("python3 test.py", 0, stdout="cycles: 147734\n")
                ]

                for _ in range(2):
                    loop = agent.state.loop_count
                    candidate_id = f"neutral-{loop}"
                    agent.state.scratch["metric_acceptance"] = {
                        "requires_improvement": True,
                        "metric": 147734,
                        "baseline": 147734,
                        "improved": False,
                        "failed": True,
                        "failure_class": "no_improvement",
                        "summary": "Metric candidate=147734 baseline=147734 improved=False.",
                        "candidate_transition_bound": True,
                        "candidate_id": candidate_id,
                        "loop": loop,
                        "todo_id": "task-001",
                        "spec_task_id": "task-001",
                        "candidate_status": "rejected",
                        "candidate_failure_class": "no_improvement",
                    }
                    agent.state.scratch["last_candidate_observation"] = {
                        "candidate_id": candidate_id,
                        "loop": loop,
                        "status": "rejected",
                        "fingerprint": "same-neutral-shape",
                        "failure_class": "no_improvement",
                        "summary": "correctness passed without metric gain",
                        "tactic_stage": "local_edit",
                        "spec_task_id": "task-001",
                        "todo_id": "task-001",
                    }
                    await agent._handle_spec_task_test_result(failed=True)

                persisted = json.loads((artifact_dir / "run_spec.json").read_text())
                persisted_task = persisted["task_graph"][0]
                rows = [
                    json.loads(line)
                    for line in (artifact_dir / "failure_signatures.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(
                    persisted_task["status"],
                    "deferred_no_improvement_plateau",
                )
                self.assertEqual(
                    persisted_task["plateau"]["plateau_task_local_count"],
                    2,
                )
                self.assertEqual(
                    [row["failure_class"] for row in rows],
                    ["metric_neutral_plateau", "metric_neutral_plateau"],
                )
                self.assertEqual(rows[-1]["status"], "deferred_no_improvement_plateau")
                self.assertEqual(rows[-1]["plateau_task_local_count"], 2)
                progress = [
                    json.loads(line)
                    for line in (artifact_dir / "spec_progress.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(progress[-1]["reason"], "metric_neutral_plateau")
                self.assertEqual(agent.state.current, AgentStateName.SCHEDULE)

        asyncio.run(run_case())

    def test_metric_neutral_plateau_without_task_identity_is_graph_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "failure_signatures_path": ".local_micro_agent/failure_signatures.jsonl",
                        "spec_metric_neutral_plateau_same_fingerprint_limit": 1,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )

            record = agent._record_metric_neutral_plateau_signature(
                spec={"version": 2, "spec_id": "metric-task", "task_graph": []},
                task={},
                candidate_record={
                    "candidate_id": "neutral",
                    "loop": 0,
                    "status": "rejected",
                    "fingerprint": "same-neutral-shape",
                    "failure_class": "no_improvement",
                    "summary": "correct but no metric gain",
                },
                metric_observation={
                    "requires_improvement": True,
                    "metric": 147734,
                    "baseline": 147734,
                    "improved": False,
                    "failure_class": "no_improvement",
                    "summary": "Metric candidate=147734 baseline=147734 improved=False.",
                    "candidate_transition_bound": True,
                    "candidate_id": "neutral",
                    "loop": 0,
                    "candidate_status": "rejected",
                    "candidate_failure_class": "no_improvement",
                },
            )

            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record["failure_class"], "metric_neutral_plateau")
            self.assertEqual(record["task_id"], "")
            self.assertNotIn("spec_task_id", record)
            self.assertEqual(record["plateau_task_local_count"], 0)
            self.assertEqual(record["status"], "rejected_no_improvement")

    def test_metric_neutral_plateau_ignores_candidate_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "failure_signatures_path": ".local_micro_agent/failure_signatures.jsonl",
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )

            record = agent._record_metric_neutral_plateau_signature(
                spec={"version": 2, "spec_id": "metric-task", "task_graph": []},
                task={
                    "task_id": "task-001",
                    "target_regions": ["target.py::build"],
                    "tactic_stage": "local_edit",
                },
                candidate_record={
                    "candidate_id": "current",
                    "loop": 2,
                    "status": "rejected",
                    "fingerprint": "same-neutral-shape",
                    "failure_class": "no_improvement",
                },
                metric_observation={
                    "requires_improvement": True,
                    "metric": 147734,
                    "baseline": 147734,
                    "improved": False,
                    "failure_class": "no_improvement",
                    "candidate_transition_bound": True,
                    "candidate_id": "previous",
                    "loop": 2,
                    "candidate_status": "rejected",
                    "candidate_failure_class": "no_improvement",
                },
            )

            self.assertIsNone(record)
            self.assertFalse((artifact_dir / "failure_signatures.jsonl").exists())
            ignored = agent.state.scratch["stale_metric_observation_ignored"]
            self.assertEqual(ignored[-1]["reason"], "candidate_id_mismatch")

    def test_metric_neutral_plateau_ignores_stale_metric_on_active_task_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "failure_signatures_path": ".local_micro_agent/failure_signatures.jsonl",
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )

            record = agent._record_metric_neutral_plateau_signature(
                spec={"version": 2, "spec_id": "metric-task", "task_graph": []},
                task={
                    "task_id": "task-002",
                    "target_regions": ["target.py::build"],
                    "tactic_stage": "structural_probe",
                },
                candidate_record={
                    "candidate_id": "loop-000-single",
                    "loop": 0,
                    "status": "rejected_active_task_shape_drift",
                    "fingerprint": "drift-shape",
                    "failure_class": "active_task_drift",
                    "spec_task_id": "task-002",
                    "todo_id": "task-002",
                },
                metric_observation={
                    "requires_improvement": True,
                    "metric": 147734,
                    "baseline": 147734,
                    "improved": False,
                    "failure_class": "no_improvement",
                    "candidate_transition_bound": True,
                    "candidate_id": "loop-000-single",
                    "loop": 0,
                    "todo_id": "task-002",
                    "spec_task_id": "task-002",
                    "candidate_status": "rejected_active_task_shape_drift",
                    "candidate_failure_class": "active_task_drift",
                },
            )

            self.assertIsNone(record)
            self.assertFalse((artifact_dir / "failure_signatures.jsonl").exists())
            ignored = agent.state.scratch["stale_metric_observation_ignored"]
            self.assertEqual(
                ignored[-1]["reason"],
                "candidate_failure_class:active_task_drift",
            )
            summary = agent._terminal_metric_neutral_plateau_summary([])
            self.assertEqual(summary, {})

    def test_portfolio_reopen_blocks_plateau_cooled_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_reopen_failed_portfolio_tasks": True,
                        "spec_portfolio_recovery_rounds": 2,
                        "failure_signatures_path": ".local_micro_agent/failure_signatures.jsonl",
                        "spec_metric_neutral_plateau_region_tactic_limit": 2,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            task = {
                "task_id": "task-002",
                "status": "failed",
                "target_regions": ["target.py::build"],
                "tactic_stage": "local_edit",
                "deliverables": ["target.py"],
                "budget": {"attempts_used": 4, "attempts_max": 4},
            }
            for index in range(2):
                agent._append_failure_signature(
                    phase="metric_gate",
                    spec={"version": 2, "spec_id": "metric-task", "task_graph": [task]},
                    task=task,
                    status="rejected_no_improvement",
                    failure_class="metric_neutral_plateau",
                    issue_code=f"metric_neutral_plateau_shape_{index}",
                    issue_scope="candidate_delta",
                    extra={"plateau_signature_key": f"shape-{index}"},
                )

            reopened = agent._reopen_failed_spec_portfolio_tasks([task])

            self.assertEqual(reopened, [])
            self.assertEqual(task["status"], "deferred_no_improvement_plateau")
            blocked = agent.state.scratch["spec_plateau_reopen_blocked_tasks"]
            self.assertEqual(blocked[0]["task_id"], "task-002")
            self.assertEqual(blocked[0]["plateau_region_tactic_count"], 2)

    def test_plateau_cooldown_affects_graph_score_and_reseed_focus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "failure_signatures_path": ".local_micro_agent/failure_signatures.jsonl",
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            cooled_task = {
                "task_id": "task-002",
                "status": "failed",
                "target_regions": ["target.py::build"],
                "tactic_stage": "local_edit",
                "deliverables": ["target.py"],
            }
            agent._append_failure_signature(
                phase="metric_gate",
                spec={"version": 2, "spec_id": "metric-task", "task_graph": [cooled_task]},
                task=cooled_task,
                status="deferred_no_improvement_plateau",
                failure_class="metric_neutral_plateau",
                issue_code="metric_neutral_plateau_same_shape",
                issue_scope="candidate_delta",
                extra={"plateau_signature_key": "same-shape"},
            )
            candidate_graph = {
                "version": 2,
                "spec_id": "candidate",
                "task_graph": [
                    {
                        "task_id": "task-new",
                        "status": "open",
                        "target_regions": ["target.py::build"],
                        "tactic_stage": "local_edit",
                    }
                ],
            }

            score = agent._spec_graph_candidate_score(candidate_graph)
            focus = agent._spec_graph_reseed_focus(
                candidate_graph,
                candidate_graph["task_graph"],
                reseed_attempt=1,
                reseed_attempts_max=2,
                cooldown_keys=agent._current_failure_cooldown_keys(),
            )

            self.assertEqual(score["plateau_cooldown_hits"], 1)
            self.assertIn("Metric-neutral plateau cooldown keys", focus)
            self.assertIn("Do not simply rename task ids", focus)

    def test_spec_report_includes_metric_neutral_plateau_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            task = {
                "task_id": "task-002",
                "status": "deferred_no_improvement_plateau",
                "target_regions": ["target.py::build"],
                "tactic_stage": "local_edit",
                "deliverables": ["target.py"],
            }
            spec = {
                "version": 2,
                "spec_id": "metric-task",
                "objective": "Improve cycles.",
                "task_graph": [task],
                "last_stop_reason": "search_frontier_exhausted",
            }
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_report_path": ".local_micro_agent/spec_report.md",
                        "spec_terminal_state_path": ".local_micro_agent/terminal_state.json",
                        "failure_signatures_path": ".local_micro_agent/failure_signatures.jsonl",
                    },
                },
                AgentState(
                    repo_root=repo,
                    user_request="test",
                    current=AgentStateName.FAILED,
                    max_loops=10,
                ),
            )
            agent.state.scratch["run_spec"] = spec
            agent._append_failure_signature(
                phase="metric_gate",
                spec=spec,
                task=task,
                status="deferred_no_improvement_plateau",
                failure_class="metric_neutral_plateau",
                issue_code="metric_neutral_plateau_same_shape",
                issue_scope="candidate_delta",
                extra={"plateau_signature_key": "same-shape"},
            )

            agent._persist_spec_report()

            report = (artifact_dir / "spec_report.md").read_text()
            terminal = json.loads((artifact_dir / "terminal_state.json").read_text())
            self.assertIn("## Metric-Neutral Plateau", report)
            self.assertIn("plateau_count: 1", report)
            self.assertEqual(terminal["metric_neutral_plateau_count"], 1)
            self.assertEqual(terminal["metric_neutral_plateau_task_ids"], ["task-002"])
            self.assertEqual(
                terminal["metric_neutral_plateau_deferred_task_ids"],
                ["task-002"],
            )

    def test_spec_mode_reopens_failed_prerequisite_when_budget_remains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            spec = {
                "version": 2,
                "spec_id": "recover-chain",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "title": "context",
                        "status": "closed",
                    },
                    {
                        "task_id": "task-002",
                        "title": "required implementation",
                        "status": "failed",
                        "depends_on": ["task-001"],
                        "deliverables": ["target.txt"],
                        "decision_hint": "budget_exhausted",
                        "last_observation": {"summary": "metric stayed flat"},
                        "budget": {"attempts_max": 3, "attempts_used": 3},
                    },
                    {
                        "task_id": "task-003",
                        "title": "dependent implementation",
                        "status": "open",
                        "depends_on": ["task-002"],
                        "deliverables": ["target.txt"],
                    },
                ],
            }
            (artifact_dir / "run_spec.json").write_text(json.dumps(spec) + "\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_enabled": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_task_recovery_rounds": 1,
                        "max_code_test_loops": 10,
                        "writable_files": ["target.txt"],
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            agent.state.loop_count = 4

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.TASK_READ)
            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            task = persisted["task_graph"][1]
            self.assertEqual(task["status"], "in_progress")
            self.assertEqual(task["recovery_rounds"], 1)
            self.assertEqual(task["budget"]["attempts_used"], 0)
            self.assertEqual(task["attempts"], 0)
            self.assertEqual(task["attempts_total"], 3)
            self.assertIn("recovery_after_failure", task["decision_hint"])
            self.assertEqual(persisted["active_task_id"], "task-002")
            progress_events = (artifact_dir / "spec_progress.jsonl").read_text()
            self.assertIn('"event": "reopened"', progress_events)
            self.assertIn('"reopened_tasks": ["task-002"]', progress_events)

    def test_spec_mode_reports_blocked_when_recovery_rounds_are_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            spec = {
                "version": 2,
                "spec_id": "blocked-chain",
                "task_graph": [
                    {"task_id": "task-001", "title": "context", "status": "closed"},
                    {
                        "task_id": "task-002",
                        "title": "required implementation",
                        "status": "failed",
                        "depends_on": ["task-001"],
                        "recovery_rounds": 1,
                        "budget": {"attempts_max": 3, "attempts_used": 3},
                    },
                    {
                        "task_id": "task-003",
                        "title": "dependent implementation",
                        "status": "open",
                        "depends_on": ["task-002"],
                    },
                ],
            }
            (artifact_dir / "run_spec.json").write_text(json.dumps(spec) + "\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_enabled": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_task_recovery_rounds": 1,
                        "max_code_test_loops": 10,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            agent.state.loop_count = 4

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.FAILED)
            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual(persisted["last_stop_reason"], "no_recovery_possible")
            progress_events = (artifact_dir / "spec_progress.jsonl").read_text()
            self.assertIn('"event": "blocked"', progress_events)
            self.assertIn('"failed_prerequisites": ["task-002"]', progress_events)
            self.assertIn('"remaining_loops": 6', progress_events)
            self.assertIn('"stop_reason": "no_recovery_possible"', progress_events)

    def test_spec_mode_relaxes_failed_dependencies_when_budget_remains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            spec = {
                "version": 2,
                "spec_id": "relax-chain",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "title": "failed tactic",
                        "status": "failed",
                        "deliverables": ["target.txt"],
                    },
                    {
                        "task_id": "task-002",
                        "title": "sibling tactic",
                        "status": "open",
                        "depends_on": ["task-001"],
                        "deliverables": ["target.txt"],
                    },
                ],
            }
            (artifact_dir / "run_spec.json").write_text(json.dumps(spec) + "\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_enabled": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_task_recovery_rounds": 0,
                        "spec_relax_failed_dependencies_with_budget": True,
                        "max_code_test_loops": 10,
                        "writable_files": ["target.txt"],
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            agent.state.loop_count = 4

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.TASK_READ)
            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            task = persisted["task_graph"][1]
            self.assertEqual(task["depends_on"], [])
            self.assertEqual(task["status"], "in_progress")
            self.assertEqual(persisted["active_task_id"], "task-002")
            progress_events = (artifact_dir / "spec_progress.jsonl").read_text()
            self.assertIn('"event": "dependencies_relaxed"', progress_events)

    def test_spec_mode_reopens_failed_portfolio_tasks_when_budget_remains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            spec = {
                "version": 2,
                "spec_id": "portfolio-reopen",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "title": "failed tactic",
                        "status": "failed",
                        "deliverables": ["target.txt"],
                        "budget": {"attempts_max": 3, "attempts_used": 3},
                    },
                    {
                        "task_id": "task-002",
                        "title": "another failed tactic",
                        "status": "failed",
                        "deliverables": ["target.txt"],
                        "budget": {"attempts_max": 3, "attempts_used": 3},
                    },
                ],
            }
            (artifact_dir / "run_spec.json").write_text(json.dumps(spec) + "\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_enabled": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_reopen_failed_portfolio_tasks": True,
                        "spec_portfolio_recovery_rounds": 2,
                        "max_code_test_loops": 10,
                        "writable_files": ["target.txt"],
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            agent.state.loop_count = 4

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.TASK_READ)
            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            first = persisted["task_graph"][0]
            self.assertEqual(first["status"], "in_progress")
            self.assertEqual(first["recovery_rounds"], 1)
            self.assertEqual(first["budget"]["attempts_used"], 0)
            self.assertEqual(first["attempts_total"], 3)
            self.assertIn("portfolio_revisit_after_failure", first["decision_hint"])
            progress_events = (artifact_dir / "spec_progress.jsonl").read_text()
            self.assertIn('"event": "portfolio_reopened"', progress_events)

    def test_spec_mode_defers_portfolio_task_when_recovery_rounds_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            spec = {
                "version": 2,
                "spec_id": "portfolio-exhausted",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "title": "repeated failed tactic",
                        "status": "failed",
                        "deliverables": ["target.txt"],
                        "recovery_rounds": 2,
                        "budget": {"attempts_max": 3, "attempts_used": 3},
                    },
                ],
            }
            (artifact_dir / "run_spec.json").write_text(json.dumps(spec) + "\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_enabled": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_reopen_failed_portfolio_tasks": True,
                        "spec_portfolio_recovery_rounds": 2,
                        "max_code_test_loops": 10,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            agent.state.loop_count = 4

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.FAILED)
            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            task = persisted["task_graph"][0]
            self.assertEqual(task["status"], "deferred_portfolio_exhausted")
            self.assertIn("portfolio_recovery_exhausted", task["decision_hint"])
            self.assertEqual(
                persisted["last_stop_reason"],
                "no_runnable_tasks_after_portfolio_exhausted",
            )
            progress_events = (artifact_dir / "spec_progress.jsonl").read_text()
            self.assertIn('"event": "portfolio_exhausted"', progress_events)
            self.assertIn('"event": "blocked"', progress_events)
            self.assertIn('"portfolio_exhausted_tasks": ["task-001"]', progress_events)
            signatures = [
                json.loads(line)
                for line in (artifact_dir / "failure_signatures.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(signatures[0]["failure_class"], "portfolio_exhausted")
            self.assertEqual(
                signatures[0]["issue_code"],
                "portfolio_recovery_budget_exhausted",
            )

    def test_spec_mode_reports_blocked_for_mixed_drift_and_portfolio_exhaustion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            spec = {
                "version": 2,
                "spec_id": "mixed-exhaustion",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "title": "drifted contract",
                        "status": "deferred_contract_drift",
                        "deliverables": ["target.txt"],
                    },
                    {
                        "task_id": "task-002",
                        "title": "repeated failed tactic",
                        "status": "failed",
                        "deliverables": ["target.txt"],
                        "recovery_rounds": 2,
                        "budget": {"attempts_max": 3, "attempts_used": 3},
                    },
                ],
            }
            (artifact_dir / "run_spec.json").write_text(json.dumps(spec) + "\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_enabled": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_reopen_failed_portfolio_tasks": True,
                        "spec_portfolio_recovery_rounds": 2,
                        "max_code_test_loops": 10,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            agent.state.loop_count = 4

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.FAILED)
            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual(persisted["task_graph"][0]["status"], "deferred_contract_drift")
            self.assertEqual(
                persisted["task_graph"][1]["status"],
                "deferred_portfolio_exhausted",
            )
            self.assertEqual(
                persisted["last_stop_reason"],
                "no_runnable_tasks_after_portfolio_exhausted",
            )
            progress_events = (artifact_dir / "spec_progress.jsonl").read_text()
            self.assertIn('"drift_deferred_tasks": ["task-001"]', progress_events)
            self.assertIn('"portfolio_exhausted_tasks": ["task-002"]', progress_events)

    def test_spec_mode_reports_blocked_for_mixed_drift_and_design_invalid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            spec = {
                "version": 2,
                "spec_id": "mixed-design-invalid",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "title": "drifted contract",
                        "status": "deferred_contract_drift",
                        "deliverables": ["target.txt"],
                    },
                    {
                        "task_id": "task-002",
                        "title": "invalid design",
                        "status": "failed_design",
                        "deliverables": ["target.txt"],
                    },
                ],
            }
            (artifact_dir / "run_spec.json").write_text(json.dumps(spec) + "\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_enabled": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "max_code_test_loops": 10,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            agent.state.loop_count = 4

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.FAILED)
            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual(
                persisted["last_stop_reason"],
                "search_frontier_exhausted_after_design_invalid",
            )
            progress_events = (artifact_dir / "spec_progress.jsonl").read_text()
            self.assertIn('"design_failed_tasks": ["task-002"]', progress_events)
            self.assertIn('"drift_deferred_tasks": ["task-001"]', progress_events)

    def test_spec_mode_marks_partial_success_for_mixed_drift_and_design_invalid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            spec = {
                "version": 2,
                "spec_id": "partial-mixed-design-invalid",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "title": "validated task",
                        "status": "closed",
                        "deliverables": ["target.txt"],
                    },
                    {
                        "task_id": "task-002",
                        "title": "drifted contract",
                        "status": "deferred_contract_drift",
                        "deliverables": ["target.txt"],
                    },
                    {
                        "task_id": "task-003",
                        "title": "invalid design",
                        "status": "failed_design",
                        "deliverables": ["target.txt"],
                    },
                ],
            }
            (artifact_dir / "run_spec.json").write_text(json.dumps(spec) + "\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_enabled": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "max_code_test_loops": 10,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            agent.state.loop_count = 4

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.DONE)
            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual(
                persisted["last_stop_reason"],
                "partial_success_search_frontier_exhausted",
            )
            self.assertEqual(persisted["task_graph"][2]["status"], "deferred_design")
            progress_events = (artifact_dir / "spec_progress.jsonl").read_text()
            self.assertIn(
                '"stop_reason": "partial_success_search_frontier_exhausted"',
                progress_events,
            )

    def test_spec_mode_selects_backtrackable_graph_before_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            sidecar_dir = artifact_dir / "spec_graph_candidates"
            sidecar_dir.mkdir(parents=True)
            current_spec = {
                "version": 2,
                "spec_id": "current",
                "search": {"graph_id": "graph-0001"},
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "title": "drifted contract",
                        "status": "deferred_contract_drift",
                        "deliverables": ["target.py"],
                    },
                    {
                        "task_id": "task-002",
                        "title": "invalid design",
                        "status": "deferred_design_invalid",
                        "deliverables": ["target.py"],
                    },
                ],
            }
            sibling_spec = {
                "version": 2,
                "spec_id": "sibling",
                "search": {"graph_id": "graph-0002", "parent_graph_id": "graph-0001"},
                "task_graph": [
                    {
                        "task_id": "task-101",
                        "title": "fresh sibling",
                        "deliverables": ["target.py"],
                        "target_regions": ["target.py::parse_item"],
                    }
                ],
            }
            (artifact_dir / "run_spec.json").write_text(json.dumps(current_spec) + "\n")
            (sidecar_dir / "graph-0002.json").write_text(json.dumps(sibling_spec) + "\n")
            (artifact_dir / "spec_graph_candidates.jsonl").write_text(
                json.dumps(
                    {
                        "schema": "spec_graph_candidate.v1",
                        "event": "candidate_created",
                        "status": "backtrackable",
                        "origin": "test",
                        "graph_id": "graph-0002",
                        "spec_sidecar_path": ".local_micro_agent/spec_graph_candidates/graph-0002.json",
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
                        "spec_mode": True,
                        "run_spec_enabled": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_graph_reseed_attempts": 1,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            agent.state.loop_count = 4

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.SCHEDULE)
            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual(persisted["spec_id"], "sibling")
            graph_events = (artifact_dir / "spec_graph_candidates.jsonl").read_text()
            self.assertIn('"status": "selected_backtrack"', graph_events)
            progress_events = (artifact_dir / "spec_progress.jsonl").read_text()
            self.assertIn('"event": "graph_backtracked"', progress_events)

    def test_spec_mode_selects_lowest_scored_backtrackable_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            sidecar_dir = artifact_dir / "spec_graph_candidates"
            sidecar_dir.mkdir(parents=True)
            current_spec = {
                "version": 2,
                "spec_id": "current",
                "search": {"graph_id": "graph-0001"},
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "title": "invalid design",
                        "status": "deferred_design_invalid",
                        "deliverables": ["target.py"],
                    }
                ],
            }
            cooled_region = "target.py::cooled_probe"
            cooled_hash = hashlib.sha1(cooled_region.encode("utf-8")).hexdigest()[:8]
            first_valid_but_cooled = {
                "version": 2,
                "spec_id": "cooled",
                "search": {"graph_id": "graph-0002", "parent_graph_id": "graph-0001"},
                "task_graph": [
                    {
                        "task_id": "task-101",
                        "title": "cooled sibling",
                        "deliverables": ["target.py"],
                        "target_regions": [cooled_region],
                        "tactic_stage": "local_edit",
                    }
                ],
            }
            later_clean = {
                "version": 2,
                "spec_id": "clean",
                "search": {"graph_id": "graph-0003", "parent_graph_id": "graph-0001"},
                "task_graph": [
                    {
                        "task_id": "task-201",
                        "title": "clean sibling",
                        "deliverables": ["target.py"],
                        "target_regions": ["target.py::clean_probe"],
                        "tactic_stage": "local_edit",
                    }
                ],
            }
            (artifact_dir / "run_spec.json").write_text(json.dumps(current_spec) + "\n")
            (sidecar_dir / "graph-0002.json").write_text(
                json.dumps(first_valid_but_cooled) + "\n"
            )
            (sidecar_dir / "graph-0003.json").write_text(json.dumps(later_clean) + "\n")
            (artifact_dir / "failure_signatures.jsonl").write_text(
                json.dumps(
                    {
                        "schema": "failure_signature.v1",
                        "cooldown_key": f"{cooled_hash}:local_edit:active_task_drift",
                    }
                )
                + "\n"
            )
            (artifact_dir / "spec_graph_candidates.jsonl").write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        {
                            "schema": "spec_graph_candidate.v1",
                            "event": "candidate_created",
                            "status": "backtrackable",
                            "origin": "test",
                            "graph_id": "graph-0002",
                            "spec_sidecar_path": ".local_micro_agent/spec_graph_candidates/graph-0002.json",
                        },
                        {
                            "schema": "spec_graph_candidate.v1",
                            "event": "candidate_created",
                            "status": "backtrackable",
                            "origin": "test",
                            "graph_id": "graph-0003",
                            "spec_sidecar_path": ".local_micro_agent/spec_graph_candidates/graph-0003.json",
                        },
                    )
                )
                + "\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_enabled": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_graph_reseed_attempts": 1,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            agent.state.loop_count = 4

            agent._schedule_spec_task()

            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual(persisted["spec_id"], "clean")
            progress_events = [
                json.loads(line)
                for line in (artifact_dir / "spec_progress.jsonl").read_text().splitlines()
                if line.strip()
            ]
            backtracked = progress_events[-1]
            self.assertEqual(backtracked["event"], "graph_backtracked")
            self.assertEqual(backtracked["selected_graph_id"], "graph-0003")
            self.assertEqual(backtracked["selection_score"]["cooldown_hits"], 0)

    def test_spec_mode_rejects_stale_backtrackable_graph_and_requests_reseed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            sidecar_dir = artifact_dir / "spec_graph_candidates"
            sidecar_dir.mkdir(parents=True)
            current_spec = {
                "version": 2,
                "spec_id": "current",
                "search": {"graph_id": "graph-0001"},
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "title": "invalid design",
                        "status": "deferred_design_invalid",
                        "deliverables": ["target.py"],
                    }
                ],
            }
            stale_spec = {
                "version": 2,
                "spec_id": "stale",
                "search": {"graph_id": "graph-0002", "parent_graph_id": "graph-0001"},
                "task_graph": [],
            }
            (artifact_dir / "run_spec.json").write_text(json.dumps(current_spec) + "\n")
            (sidecar_dir / "graph-0002.json").write_text(json.dumps(stale_spec) + "\n")
            (artifact_dir / "spec_graph_candidates.jsonl").write_text(
                json.dumps(
                    {
                        "schema": "spec_graph_candidate.v1",
                        "event": "candidate_created",
                        "status": "backtrackable",
                        "origin": "test",
                        "graph_id": "graph-0002",
                        "spec_sidecar_path": ".local_micro_agent/spec_graph_candidates/graph-0002.json",
                    }
                )
                + "\n"
            )
            (artifact_dir / "failure_signatures.jsonl").write_text(
                json.dumps(
                    {
                        "schema": "failure_signature.v1",
                        "failure_class": "design_rewrite_invalid",
                        "issue_code": "single_broad_structural_task",
                        "cooldown_key": "abcd1234:structural_probe:single_broad_structural_task",
                        "summary": "broad structural task",
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
                        "spec_mode": True,
                        "run_spec_enabled": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_graph_reseed_attempts": 1,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            agent.state.loop_count = 4

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.SPEC_SYNTH)
            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual(persisted["search"]["reseed_attempts"], 1)
            self.assertIn(
                "abcd1234:structural_probe:single_broad_structural_task",
                agent.state.scratch["spec_rewrite_focus"],
            )
            graph_events = (artifact_dir / "spec_graph_candidates.jsonl").read_text()
            self.assertIn('"status": "rejected_stale"', graph_events)
            progress_events = (artifact_dir / "spec_progress.jsonl").read_text()
            self.assertIn('"event": "graph_reseed_requested"', progress_events)

            reseeded_spec = json.dumps(
                {
                    "version": 2,
                    "spec_id": "reseeded",
                    "search": {"graph_id": "model-supplied-id"},
                    "task_graph": [
                        {
                            "task_id": "task-101",
                            "title": "fresh local probe",
                            "deliverables": ["target.py"],
                            "target_regions": ["target.py::fresh_probe"],
                        }
                    ],
                }
            )
            agent.models = _StaticModelManager(reseeded_spec)

            asyncio.run(agent._maybe_refresh_run_spec(force=True))

            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual(persisted["spec_id"], "reseeded")
            self.assertNotEqual(persisted["search"]["graph_id"], "model-supplied-id")
            self.assertEqual(persisted["search"]["parent_graph_id"], "graph-0001")
            self.assertEqual(persisted["search"]["reseed_attempts"], 1)
            self.assertIn(
                "abcd1234:structural_probe:single_broad_structural_task",
                persisted["search"]["cooldown_keys"],
            )
            graph_events = [
                json.loads(line)
                for line in (artifact_dir / "spec_graph_candidates.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(graph_events[-1]["status"], "selected")
            self.assertEqual(
                graph_events[-1]["origin"],
                "reseed_after_graph_frontier_exhausted",
            )
            self.assertEqual(graph_events[-1]["parent_graph_id"], "graph-0001")

    def test_spec_graph_reseed_focus_caps_exhausted_task_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_graph_reseed_task_summary_limit": 2,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            tasks = [
                {
                    "task_id": f"task-{index:03d}",
                    "title": f"Task {index}",
                    "status": "deferred_design_invalid",
                    "target_regions": [f"target.py::region_{index}"],
                }
                for index in range(5)
            ]

            focus = agent._spec_graph_reseed_focus(
                {"version": 2, "spec_id": "current", "task_graph": tasks},
                tasks,
                reseed_attempt=1,
                reseed_attempts_max=2,
                cooldown_keys=[],
            )

            self.assertNotIn("task-000", focus)
            self.assertNotIn("task-002", focus)
            self.assertIn("task-003", focus)
            self.assertIn("task-004", focus)

    def test_spec_graph_reseed_focus_includes_model_suggested_regions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (repo / "target.py").write_text(
                "def parse_item(value):\n"
                "    return value.strip()\n\n"
                "def helper(value):\n"
                "    return value\n"
            )
            records = [
                {
                    "failure_class": "active_task_drift",
                    "drift_attempted_regions": ["target.py::helper"],
                    "drift_region_pairs": [
                        {
                            "declared": "target.py::parse_item",
                            "attempted": "target.py::helper",
                        }
                    ],
                }
                for _ in range(2)
            ]
            (artifact_dir / "candidates.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                        "writable_files": ["target.py"],
                        "spec_model_suggested_region_min_count": 2,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )

            focus = agent._spec_graph_reseed_focus(
                {"version": 2, "spec_id": "current", "task_graph": []},
                [],
                reseed_attempt=1,
                reseed_attempts_max=2,
                cooldown_keys=[],
            )

            self.assertIn("Model-suggested regions from repeated drift", focus)
            self.assertIn("target.py::helper", focus)
            self.assertIn("target.py::parse_item", focus)

    def test_spec_graph_score_counts_drift_saturation_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                        "spec_drift_saturation_threshold": 2,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            cooled_hash = agent._failure_signature_target_region_hash(
                ["target.py::parse_item"]
            )
            records = [
                {
                    "failure_class": "active_task_drift",
                    "drift_cooldown_key": f"{cooled_hash}:local_edit:same_drift",
                }
                for _ in range(2)
            ]
            (artifact_dir / "candidates.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n"
            )
            spec = {
                "version": 2,
                "spec_id": "candidate",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "status": "open",
                        "target_regions": ["target.py::parse_item"],
                        "tactic_stage": "local_edit",
                    }
                ],
            }

            score = agent._spec_graph_candidate_score(spec)

            self.assertEqual(score["drift_saturation_hits"], 1)

    def test_spec_rewrite_respects_reseed_reserved_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_synth_call_budget": 5,
                        "spec_reseed_reserved_synth_calls": 2,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            agent.state.scratch["run_spec"] = {
                "version": 2,
                "spec_id": "reserved-drift",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "status": "needs_contract_rewrite",
                        "target_regions": ["target.py::parse_item"],
                        "tactic_stage": "local_edit",
                        "contract_rewrite": {"reason": "repeated_active_task_drift"},
                    }
                ],
            }
            agent.state.scratch["spec_rewrite_target_task_id"] = "task-001"
            agent.state.scratch["spec_synth_call_count"] = 3

            rewrite_allowed = agent._consume_spec_synth_call_budget(
                "spec_synth_rewrite"
            )
            self.assertFalse(rewrite_allowed)
            self.assertEqual(
                agent.state.scratch["spec_synth_budget_reserved_at"][
                    "reserved_for_reseed"
                ],
                2,
            )
            reseed_allowed = agent._consume_spec_synth_call_budget(
                "spec_synth_reseed"
            )

            self.assertTrue(reseed_allowed)
            self.assertEqual(agent.state.scratch["spec_synth_call_count"], 4)
            self.assertNotIn("spec_synth_budget_reserved_at", agent.state.scratch)

    def test_reseed_reserved_budget_does_not_block_non_drift_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_synth_call_budget": 5,
                        "spec_reseed_reserved_synth_calls": 2,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            agent.state.scratch["run_spec"] = {
                "version": 2,
                "spec_id": "reserved-design",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "status": "needs_design",
                        "target_regions": ["target.py::parse_item"],
                        "tactic_stage": "local_edit",
                        "design_contract": {"status": "rejected"},
                    }
                ],
            }
            agent.state.scratch["spec_rewrite_target_task_id"] = "task-001"
            agent.state.scratch["spec_synth_call_count"] = 3

            rewrite_allowed = agent._consume_spec_synth_call_budget(
                "spec_synth_rewrite"
            )

            self.assertTrue(rewrite_allowed)
            self.assertEqual(agent.state.scratch["spec_synth_call_count"], 4)
            self.assertNotIn("spec_synth_budget_reserved_at", agent.state.scratch)

    def test_reseed_reserved_budget_defers_targeted_drift_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_progress_path": ".local_micro_agent/spec_progress.jsonl",
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            previous_spec = {
                "version": 2,
                "spec_id": "reserve",
                "active_task_id": "task-001",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "status": "needs_contract_rewrite",
                        "target_regions": ["target.py::parse_item"],
                        "tactic_stage": "local_edit",
                    }
                ],
            }
            agent.state.scratch["spec_synth_budget_reserved_at"] = {
                "call_site": "spec_synth_rewrite",
                "used": 20,
                "budget": 24,
                "reserved_for_reseed": 4,
            }

            agent._defer_targeted_rewrite_for_reseed_reserve(
                previous_spec,
                "task-001",
            )

            self.assertEqual(agent.state.current, AgentStateName.SCHEDULE)
            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            task = persisted["task_graph"][0]
            self.assertEqual(task["status"], "deferred_contract_drift")
            self.assertEqual(task["contract_rewrite"]["reason"], "reseed_budget_reserved")
            progress = json.loads(
                (artifact_dir / "spec_progress.jsonl").read_text().splitlines()[-1]
            )
            self.assertEqual(progress["action"], "rewrite_rejected_reseed_budget_reserved")
            signature = json.loads(
                (artifact_dir / "failure_signatures.jsonl").read_text().splitlines()[-1]
            )
            self.assertEqual(signature["issue_code"], "reseed_budget_reserved")
            self.assertEqual(signature["failure_class"], "active_task_drift")

    def test_spec_jsonl_reader_keeps_recent_records_when_limited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text(
                "\n".join(json.dumps({"index": index}) for index in range(1005))
                + "\n"
            )

            capped = MicroAgent._read_spec_jsonl(path, limit=1000)
            uncapped = MicroAgent._read_spec_jsonl(path, limit=None)
            default = MicroAgent._read_spec_jsonl(path)

            self.assertEqual(len(capped), 1000)
            self.assertEqual(capped[0]["index"], 5)
            self.assertEqual(capped[-1]["index"], 1004)
            self.assertEqual(len(uncapped), 1005)
            self.assertEqual(len(default), 1005)

    def test_spec_reseed_candidate_count_records_backtrackable_variants(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                artifact_dir = repo / ".local_micro_agent"
                artifact_dir.mkdir()
                current_spec = {
                    "version": 2,
                    "spec_id": "current",
                    "search": {"graph_id": "graph-0001"},
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "invalid design",
                            "status": "deferred_design_invalid",
                            "deliverables": ["target.py"],
                        }
                    ],
                }
                (artifact_dir / "run_spec.json").write_text(
                    json.dumps(current_spec) + "\n"
                )
                selected = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "selected",
                        "task_graph": [
                            {
                                "task_id": "task-101",
                                "title": "selected target",
                                "deliverables": ["target.py"],
                                "target_regions": ["target.py::selected"],
                            }
                        ],
                    }
                )
                duplicate = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "duplicate",
                        "task_graph": [
                            {
                                "task_id": "task-201",
                                "title": "duplicate target",
                                "deliverables": ["target.py"],
                                "target_regions": ["target.py::selected"],
                            }
                        ],
                    }
                )
                sibling = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "sibling",
                        "task_graph": [
                            {
                                "task_id": "task-301",
                                "title": "sibling target",
                                "deliverables": ["target.py"],
                                "target_regions": ["target.py::sibling"],
                            }
                        ],
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {"reasoner": "sequence", "default": "sequence"},
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_graph_reseed_attempts": 1,
                            "spec_graph_candidate_count": 3,
                            "spec_synth_call_budget": 3,
                        },
                    },
                    AgentState(repo_root=repo, user_request="test", max_loops=10),
                )
                agent.models = _SequenceModelManager([selected, duplicate, sibling])
                agent.state.loop_count = 4

                agent._schedule_spec_task()
                await agent._maybe_refresh_run_spec(force=True)

                persisted = json.loads((artifact_dir / "run_spec.json").read_text())
                self.assertEqual(persisted["spec_id"], "selected")
                self.assertEqual(persisted["search"]["graph_id"], "graph-0002")
                events = [
                    json.loads(line)
                    for line in (
                        artifact_dir / "spec_graph_candidates.jsonl"
                    ).read_text().splitlines()
                    if line.strip()
                ]
                self.assertEqual(
                    [event["status"] for event in events],
                    ["selected", "duplicate_variant", "backtrackable"],
                )
                self.assertEqual(events[-1]["parent_graph_id"], "graph-0001")
                self.assertEqual(agent._spec_synth_call_count(), 3)
                self.assertTrue(
                    (
                        artifact_dir
                        / "spec_graph_candidates"
                        / f"{events[-1]['graph_id']}.json"
                    ).exists()
                )

        asyncio.run(run_case())

    def test_spec_mode_marks_backtrackable_graph_with_missing_sidecar_stale(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            current_spec = {
                "version": 2,
                "spec_id": "current",
                "search": {"graph_id": "graph-0001"},
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "status": "deferred_design_invalid",
                        "deliverables": ["target.py"],
                    }
                ],
            }
            (artifact_dir / "run_spec.json").write_text(json.dumps(current_spec) + "\n")
            (artifact_dir / "spec_graph_candidates.jsonl").write_text(
                json.dumps(
                    {
                        "schema": "spec_graph_candidate.v1",
                        "event": "candidate_created",
                        "status": "backtrackable",
                        "origin": "test",
                        "graph_id": "graph-0002",
                        "graph_signature": ["target.py::missing:local_edit"],
                        "spec_sidecar_path": ".local_micro_agent/spec_graph_candidates/graph-0002.json",
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
                        "spec_mode": True,
                        "run_spec_enabled": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_graph_reseed_attempts": 1,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            agent.state.loop_count = 4

            agent._schedule_spec_task()

            events = [
                json.loads(line)
                for line in (
                    artifact_dir / "spec_graph_candidates.jsonl"
                ).read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(events[-1]["status"], "rejected_stale")
            self.assertEqual(events[-1]["issue_codes"], ["missing_or_invalid_sidecar"])
            self.assertEqual(agent.state.current, AgentStateName.SPEC_SYNTH)

    def test_spec_reseed_candidate_count_respects_spec_synth_budget(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                artifact_dir = repo / ".local_micro_agent"
                artifact_dir.mkdir()
                current_spec = {
                    "version": 2,
                    "spec_id": "current",
                    "search": {"graph_id": "graph-0001"},
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "status": "deferred_design_invalid",
                            "deliverables": ["target.py"],
                        }
                    ],
                }
                (artifact_dir / "run_spec.json").write_text(
                    json.dumps(current_spec) + "\n"
                )
                selected = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "selected",
                        "task_graph": [
                            {
                                "task_id": "task-101",
                                "deliverables": ["target.py"],
                                "target_regions": ["target.py::selected"],
                            }
                        ],
                    }
                )
                sibling = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "sibling",
                        "task_graph": [
                            {
                                "task_id": "task-201",
                                "deliverables": ["target.py"],
                                "target_regions": ["target.py::sibling"],
                            }
                        ],
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {"reasoner": "sequence", "default": "sequence"},
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_graph_reseed_attempts": 1,
                            "spec_graph_candidate_count": 3,
                            "spec_synth_call_budget": 2,
                        },
                    },
                    AgentState(repo_root=repo, user_request="test", max_loops=10),
                )
                agent.models = _SequenceModelManager([selected, sibling])
                agent.state.loop_count = 4

                agent._schedule_spec_task()
                await agent._maybe_refresh_run_spec(force=True)

                events = [
                    json.loads(line)
                    for line in (
                        artifact_dir / "spec_graph_candidates.jsonl"
                    ).read_text().splitlines()
                    if line.strip()
                ]
                self.assertEqual(
                    [event["status"] for event in events],
                    ["selected", "backtrackable"],
                )
                self.assertEqual(agent._spec_synth_call_count(), 2)
                self.assertTrue(agent.state.scratch["spec_synth_budget_exhausted"])

        asyncio.run(run_case())

    def test_spec_mode_reports_graph_reseed_exhaustion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            spec = {
                "version": 2,
                "spec_id": "reseed-exhausted",
                "search": {
                    "graph_id": "graph-0001",
                    "reseed_attempts": 1,
                    "reseed_attempts_max": 1,
                },
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "title": "invalid design",
                        "status": "deferred_design_invalid",
                        "deliverables": ["target.py"],
                    }
                ],
            }
            (artifact_dir / "run_spec.json").write_text(json.dumps(spec) + "\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_enabled": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_graph_reseed_attempts": 1,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            agent.state.loop_count = 4

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.FAILED)
            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual(
                persisted["last_stop_reason"],
                "search_frontier_exhausted_after_graph_reseed_exhausted",
            )
            progress_events = (artifact_dir / "spec_progress.jsonl").read_text()
            self.assertIn(
                '"stop_reason": "search_frontier_exhausted_after_graph_reseed_exhausted"',
                progress_events,
            )

    def test_spec_mode_preserves_partial_success_after_graph_reseed_exhaustion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            spec = {
                "version": 2,
                "spec_id": "partial-reseed-exhausted",
                "search": {
                    "graph_id": "graph-0001",
                    "reseed_attempts": 1,
                    "reseed_attempts_max": 1,
                },
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "title": "closed task",
                        "status": "closed",
                        "deliverables": ["target.py"],
                    },
                    {
                        "task_id": "task-002",
                        "title": "invalid design",
                        "status": "deferred_design_invalid",
                        "deliverables": ["target.py"],
                    },
                ],
            }
            (artifact_dir / "run_spec.json").write_text(json.dumps(spec) + "\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_enabled": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_graph_reseed_attempts": 1,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            agent.state.loop_count = 4

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.DONE)
            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual(
                persisted["last_stop_reason"],
                "partial_success_search_frontier_exhausted",
            )
            self.assertEqual(persisted["task_graph"][1]["status"], "deferred_design")
            progress_events = (artifact_dir / "spec_progress.jsonl").read_text()
            self.assertIn('"graph_reseed_exhausted": true', progress_events)

    def test_spec_mode_schedules_open_sibling_without_reopening_exhausted_portfolio_task(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            spec = {
                "version": 2,
                "spec_id": "portfolio-sibling",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "title": "exhausted failed tactic",
                        "status": "failed",
                        "deliverables": ["target.txt"],
                        "recovery_rounds": 2,
                        "budget": {"attempts_max": 3, "attempts_used": 3},
                    },
                    {
                        "task_id": "task-002",
                        "title": "fresh sibling tactic",
                        "status": "open",
                        "deliverables": ["target.txt"],
                        "budget": {"attempts_max": 3, "attempts_used": 0},
                    },
                ],
            }
            (artifact_dir / "run_spec.json").write_text(json.dumps(spec) + "\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_enabled": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_reopen_failed_portfolio_tasks": True,
                        "spec_portfolio_recovery_rounds": 2,
                        "max_code_test_loops": 10,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            agent.state.loop_count = 4

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.TASK_READ)
            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual(persisted["active_task_id"], "task-002")
            self.assertEqual(persisted["task_graph"][0]["status"], "failed")
            self.assertEqual(persisted["task_graph"][1]["status"], "in_progress")
            progress_events = (artifact_dir / "spec_progress.jsonl").read_text()
            self.assertIn('"event": "scheduled"', progress_events)
            self.assertNotIn('"event": "portfolio_reopened"', progress_events)

    def test_spec_mode_cold_start_synthesizes_v2_run_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("value = 'old'\n")
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            model_output = json.dumps(
                {
                    "version": 2,
                    "spec_id": "cold-start",
                    "objective": "Update target value.",
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "write target",
                            "deliverables": ["target.py"],
                            "read_hints": ["target.py"],
                            "acceptance": {
                                "kind": "command",
                                "commands": [
                                    "python3 -c \"from pathlib import Path; assert Path('target.py').read_text() == 'done'\""
                                ],
                            },
                            "budget": {"attempts_max": 2},
                        }
                    ],
                }
            )
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "plan_markdown": "seeded",
                    "seed_files": ["target.py"],
                    "spec_mode": True,
                    "run_spec_enabled": True,
                    "run_spec_path": ".local_micro_agent/run_spec.json",
                    "max_code_test_loops": 3,
                    "writable_files": ["target.py"],
                    "seed_changes": [{"path": "target.py", "content": "done"}],
                    "deterministic_test_decision": True,
                },
            }
            state = AgentState(repo_root=repo, user_request="test", max_loops=3)
            agent = MicroAgent(config, state)
            agent.models = _StaticModelManager(model_output)

            result = asyncio.run(agent.run())

            self.assertEqual(result.current, AgentStateName.DONE)
            self.assertIn("attempting SPEC_SYNTH", "\n".join(result.notes))
            spec = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual(spec["version"], 2)
            self.assertEqual(spec["progress"], {"total": 1, "closed": 1, "deferred": 0, "failed": 0})
            self.assertEqual(spec["task_graph"][0]["status"], "closed")

    def test_spec_mode_defers_task_after_budget_exhaustion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "run_spec.json").write_text(
                json.dumps(
                    {
                        "version": 2,
                        "spec_id": "blocked",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "write missing value",
                                "deliverables": ["a.txt"],
                                "acceptance": {
                                    "kind": "command",
                                    "commands": [
                                        "python3 -c \"from pathlib import Path; assert Path('a.txt').read_text() == 'expected'\""
                                    ],
                                },
                                "budget": {"attempts_max": 1},
                            }
                        ],
                    }
                )
                + "\n"
            )

            result = run_agent(
                repo,
                {
                    "spec_mode": True,
                    "run_spec_enabled": True,
                    "run_spec_path": ".local_micro_agent/run_spec.json",
                    "max_code_test_loops": 3,
                    "writable_files": ["a.txt"],
                    "seed_files": [],
                    "seed_changes": [{"path": "a.txt", "content": "wrong"}],
                },
            )

            self.assertEqual(result.current, AgentStateName.FAILED)
            spec = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual(spec["task_graph"][0]["status"], "failed")
            self.assertEqual(spec["progress"], {"total": 1, "closed": 0, "deferred": 0, "failed": 1})

    def test_spec_mode_does_not_count_patch_miss_against_task_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            task = {
                "task_id": "task-001",
                "title": "patch target",
                "status": "in_progress",
                "deliverables": ["target.py"],
                "acceptance": {"kind": "command", "commands": []},
                "budget": {"attempts_max": 1, "attempts_used": 0},
            }
            spec = {
                "version": 2,
                "spec_id": "patch-miss",
                "task_graph": [task],
            }
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_enabled": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=3),
            )
            agent.state.scratch["run_spec"] = spec
            agent.state.scratch["current_spec_task"] = task
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "last_non_budget_attempt": {
                    "loop": 0,
                    "budget_counted": False,
                    "failure_class": "patch_miss",
                },
            }
            agent.state.test_results = [
                TestResult(command="preflight", exit_code=1, stderr="No code changes were applied")
            ]

            asyncio.run(agent._handle_spec_task_test_result(failed=True))

            self.assertEqual(task["budget"]["attempts_used"], 0)
            self.assertEqual(task["status"], "in_progress")
            self.assertEqual(agent.state.loop_count, 1)
            self.assertNotEqual(agent.state.current, AgentStateName.SCHEDULE)

    def test_spec_mode_forces_deterministic_task_closure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "run_spec.json").write_text(
                json.dumps(
                    {
                        "version": 2,
                        "spec_id": "deterministic",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "write target",
                                "deliverables": ["target.txt"],
                                "acceptance": {
                                    "kind": "command",
                                    "commands": [
                                        "python3 -c \"from pathlib import Path; assert Path('target.txt').read_text() == 'done'\""
                                    ],
                                },
                            }
                        ],
                    }
                )
                + "\n"
            )

            result = run_agent(
                repo,
                {
                    "spec_mode": True,
                    "run_spec_enabled": True,
                    "run_spec_path": ".local_micro_agent/run_spec.json",
                    "deterministic_test_decision": False,
                    "max_code_test_loops": 2,
                    "writable_files": ["target.txt"],
                    "seed_files": [],
                    "seed_changes": [{"path": "target.txt", "content": "done"}],
                },
            )

            self.assertEqual(result.current, AgentStateName.DONE)
            spec = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual(spec["task_graph"][0]["status"], "closed")

    def test_spec_mode_global_loop_cap_blocks_remaining_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "run_spec.json").write_text(
                json.dumps(
                    {
                        "version": 2,
                        "spec_id": "loop-cap",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "write a",
                                "deliverables": ["a.txt"],
                                "acceptance": {
                                    "kind": "command",
                                    "commands": [
                                        "python3 -c \"from pathlib import Path; assert Path('a.txt').read_text() == 'done-a'\""
                                    ],
                                },
                            },
                            {
                                "task_id": "task-002",
                                "title": "write b",
                                "depends_on": ["task-001"],
                                "deliverables": ["b.txt"],
                                "acceptance": {
                                    "kind": "command",
                                    "commands": [
                                        "python3 -c \"from pathlib import Path; assert Path('b.txt').read_text() == 'done-b'\""
                                    ],
                                },
                            },
                        ],
                    }
                )
                + "\n"
            )

            result = run_agent(
                repo,
                {
                    "spec_mode": True,
                    "run_spec_enabled": True,
                    "run_spec_path": ".local_micro_agent/run_spec.json",
                    "max_code_test_loops": 1,
                    "writable_files": ["*.txt"],
                    "seed_files": [],
                    "seed_changes": [
                        {"path": "a.txt", "content": "done-a"},
                        {"path": "b.txt", "content": "done-b"},
                    ],
                },
            )

            self.assertEqual(result.current, AgentStateName.FAILED)
            self.assertIn("max_code_test_loops=1", "\n".join(result.notes))
            spec = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual([task["status"] for task in spec["task_graph"]], ["closed", "open"])
            self.assertEqual(spec["last_stop_reason"], "max_code_test_loops")
            report = (artifact_dir / "spec_report.md").read_text()
            self.assertIn("stop_reason: `max_code_test_loops`", report)

    def test_spec_mode_synthesizes_and_freezes_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "run_spec.json").write_text(
                json.dumps(
                    {
                        "version": 2,
                        "spec_id": "acceptance",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "write target",
                                "deliverables": ["target.py"],
                                "acceptance": {"kind": "synthesized"},
                            }
                        ],
                    }
                )
                + "\n"
            )
            model_output = json.dumps(
                {
                    "files": [
                        {
                            "path": "test_task.py",
                            "content": (
                                "import unittest\n"
                                "from pathlib import Path\n\n"
                                "class TaskTest(unittest.TestCase):\n"
                                "    def test_target(self):\n"
                                "        self.assertEqual(Path('target.py').read_text(), 'done')\n"
                            ),
                        }
                    ],
                    "commands": [
                        "python3 -c \"from pathlib import Path; Path('pwned').write_text('yes')\" # .lma_acceptance/task-001"
                    ],
                }
            )
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "plan_markdown": "seeded",
                    "seed_files": [],
                    "spec_mode": True,
                    "run_spec_enabled": True,
                    "run_spec_path": ".local_micro_agent/run_spec.json",
                    "max_code_test_loops": 3,
                    "writable_files": ["target.py", ".lma_acceptance/**"],
                    "seed_changes": [{"path": "target.py", "content": "done"}],
                    "deterministic_test_decision": True,
                },
            }
            state = AgentState(repo_root=repo, user_request="test", max_loops=3)
            agent = MicroAgent(config, state)
            agent.models = _StaticModelManager(model_output)

            result = asyncio.run(agent.run())

            self.assertEqual(result.current, AgentStateName.DONE)
            spec = json.loads((artifact_dir / "run_spec.json").read_text())
            acceptance = spec["task_graph"][0]["acceptance"]
            self.assertEqual(acceptance["kind"], "synthesized")
            self.assertTrue(acceptance["frozen_sha256"])
            self.assertEqual(
                acceptance["test_paths"],
                [".lma_acceptance/task-001/test_task.py"],
            )
            self.assertEqual(
                acceptance["commands"],
                [
                    f"{shlex.quote(sys.executable)} -m unittest discover "
                    "-s .lma_acceptance/task-001 -p 'test*.py'"
                ],
            )
            self.assertFalse((repo / "pwned").exists())

    def test_spec_mode_rejects_vacuous_synthesized_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("value = 'old'\n")
            model_output = json.dumps(
                {
                    "files": [
                        {
                            "path": "check_task.py",
                            "content": (
                                "import unittest\n\n"
                                "class TaskTest(unittest.TestCase):\n"
                                "    def test_placeholder(self):\n"
                                "        self.assertTrue(True)\n"
                            ),
                        }
                    ]
                }
            )
            task = {
                "task_id": "task-001",
                "title": "write target",
                "deliverables": ["target.py"],
                "acceptance": {"kind": "synthesized"},
            }
            agent = MicroAgent(
                {
                    "models": {"default": "static"},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_acceptance_synth_retries": 0,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.models = _StaticModelManager(model_output)

            async def exercise() -> bool:
                await agent.mcp.start()
                try:
                    return await agent._synthesize_and_freeze_acceptance(task)
                finally:
                    await agent.mcp.close()

            ok = asyncio.run(exercise())

            self.assertFalse(ok)
            self.assertNotEqual(task.get("status"), "closed")
            self.assertNotIn("frozen_sha256", task.get("acceptance", {}))
            self.assertIn("empty files or commands", "\n".join(agent.state.notes))
            self.assertTrue(
                agent._acceptance_results_ran_zero_tests(
                    [TestResult(command="unittest", exit_code=0, stderr="Ran 0 tests in 0.000s")]
                )
            )

    def test_spec_mode_blocks_writes_to_frozen_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            acceptance_dir = repo / ".lma_acceptance" / "task-001"
            artifact_dir.mkdir()
            acceptance_dir.mkdir(parents=True)
            test_path = acceptance_dir / "test_task.py"
            test_path.write_text("assert False\n")
            agent_for_hash = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=repo, user_request="test"),
            )
            frozen = agent_for_hash._hash_acceptance_files(
                [".lma_acceptance/task-001/test_task.py"]
            )
            (artifact_dir / "run_spec.json").write_text(
                json.dumps(
                    {
                        "version": 2,
                        "spec_id": "blocked-write",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "try to edit tests",
                                "deliverables": ["target.py", ".lma_acceptance/task-001/test_task.py"],
                                "acceptance": {
                                    "kind": "synthesized",
                                    "test_paths": [".lma_acceptance/task-001/test_task.py"],
                                    "commands": ["python3 .lma_acceptance/task-001/test_task.py"],
                                    "frozen_sha256": frozen,
                                },
                                "budget": {"attempts_max": 1},
                            }
                        ],
                    }
                )
                + "\n"
            )

            result = run_agent(
                repo,
                {
                    "spec_mode": True,
                    "run_spec_enabled": True,
                    "run_spec_path": ".local_micro_agent/run_spec.json",
                    "max_code_test_loops": 1,
                    "writable_files": ["target.py", ".lma_acceptance/**"],
                    "seed_files": [],
                    "seed_changes": [
                        {"path": ".lma_acceptance/task-001/test_task.py", "content": "assert True\n"},
                        {"path": "target.py", "content": "done"},
                    ],
                },
            )

            self.assertEqual(result.current, AgentStateName.FAILED)
            self.assertEqual(test_path.read_text(), "assert False\n")
            self.assertTrue(
                any("Rejected out-of-plan change" in note for note in result.notes)
            )

    def test_spec_regression_gate_keeps_changes_for_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "a.txt").write_text("done")
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "run_spec.json").write_text(
                json.dumps(
                    {
                        "version": 2,
                        "spec_id": "regression-repair",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "keep a done",
                                "deliverables": ["a.txt"],
                                "status": "closed",
                                "acceptance": {
                                    "kind": "command",
                                    "commands": [
                                        "python3 -c \"from pathlib import Path; assert Path('a.txt').read_text() == 'done'\""
                                    ],
                                },
                            },
                            {
                                "task_id": "task-002",
                                "title": "write b without breaking a",
                                "depends_on": ["task-001"],
                                "deliverables": ["a.txt", "b.txt"],
                                "acceptance": {
                                    "kind": "command",
                                    "commands": [
                                        "python3 -c \"from pathlib import Path; assert Path('b.txt').read_text() == 'done'\""
                                    ],
                                },
                                "budget": {"attempts_max": 2},
                            },
                        ],
                    }
                )
                + "\n"
            )
            outputs = [
                json.dumps(
                    {
                        "changes": [
                            {"path": "a.txt", "content": "broken", "reason": "bad regression"},
                            {"path": "b.txt", "content": "done", "reason": "task output"},
                        ]
                    }
                ),
                json.dumps(
                    {
                        "changes": [
                            {
                                "path": "a.txt",
                                "target": "broken",
                                "replacement": "done",
                                "reason": "repair closed task regression",
                            }
                        ]
                    }
                ),
            ]
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "plan_markdown": "seeded",
                    "seed_files": [],
                    "spec_mode": True,
                    "run_spec_enabled": True,
                    "run_spec_path": ".local_micro_agent/run_spec.json",
                    "max_code_test_loops": 4,
                    "writable_files": ["*.txt"],
                    "deterministic_test_decision": True,
                    "retry_rejected_candidates": True,
                    "spec_regression_scope": "all",
                },
            }
            state = AgentState(repo_root=repo, user_request="test", max_loops=4)
            agent = MicroAgent(config, state)
            agent.models = _SequenceModelManager(outputs)

            result = asyncio.run(agent.run())

            self.assertEqual(result.current, AgentStateName.DONE)
            self.assertEqual((repo / "a.txt").read_text(), "done")
            self.assertEqual((repo / "b.txt").read_text(), "done")
            self.assertTrue(
                any("Keeping current task changes after regression gate failure" in note for note in result.notes)
            )
            spec = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual([task["status"] for task in spec["task_graph"]], ["closed", "closed"])

    def test_spec_task_boundary_snapshot_restores_on_budget_exhaustion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "a.txt").write_text("done")
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "run_spec.json").write_text(
                json.dumps(
                    {
                        "version": 2,
                        "spec_id": "regression-rollback",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "keep a done",
                                "deliverables": ["a.txt"],
                                "status": "closed",
                                "acceptance": {
                                    "kind": "command",
                                    "commands": [
                                        "python3 -c \"from pathlib import Path; assert Path('a.txt').read_text() == 'done'\""
                                    ],
                                },
                            },
                            {
                                "task_id": "task-002",
                                "title": "write b but regression remains",
                                "depends_on": ["task-001"],
                                "deliverables": ["a.txt", "b.txt"],
                                "acceptance": {
                                    "kind": "command",
                                    "commands": [
                                        "python3 -c \"from pathlib import Path; assert Path('b.txt').read_text() == 'done'\""
                                    ],
                                },
                                "budget": {"attempts_max": 1},
                            },
                        ],
                    }
                )
                + "\n"
            )

            result = run_agent(
                repo,
                {
                    "spec_mode": True,
                    "run_spec_enabled": True,
                    "run_spec_path": ".local_micro_agent/run_spec.json",
                    "max_code_test_loops": 2,
                    "writable_files": ["*.txt"],
                    "seed_files": [],
                    "seed_changes": [
                        {"path": "a.txt", "content": "broken"},
                        {"path": "b.txt", "content": "done"},
                    ],
                    "spec_regression_scope": "all",
                },
            )

            self.assertEqual(result.current, AgentStateName.FAILED)
            self.assertEqual((repo / "a.txt").read_text(), "done")
            self.assertFalse((repo / "b.txt").exists())
            self.assertTrue(
                any("Restored spec task boundary snapshot: task-002" in note for note in result.notes)
            )
            report = (repo / ".local_micro_agent" / "spec_report.md").read_text()
            self.assertIn("status: `failed`", report)
            self.assertIn("task-002", report)

    def test_spec_mode_resume_skips_closed_tasks_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "a.txt").write_text("done-a")
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "run_spec.json").write_text(
                json.dumps(
                    {
                        "version": 2,
                        "spec_id": "resume",
                        "objective": "resume from task two",
                        "search": {
                            "graph_id": "graph-0042",
                            "origin": "prior_run",
                        },
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "already done",
                                "deliverables": ["a.txt"],
                                "status": "closed",
                                "acceptance": {
                                    "kind": "command",
                                    "commands": [
                                        "python3 -c \"from pathlib import Path; assert Path('a.txt').read_text() == 'done-a'\""
                                    ],
                                },
                            },
                            {
                                "task_id": "task-002",
                                "title": "finish b",
                                "depends_on": ["task-001"],
                                "deliverables": ["b.txt"],
                                "status": "open",
                                "acceptance": {
                                    "kind": "command",
                                    "commands": [
                                        "python3 -c \"from pathlib import Path; assert Path('b.txt').read_text() == 'done-b'\""
                                    ],
                                },
                            },
                        ],
                    }
                )
                + "\n"
            )
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "plan_markdown": "seeded",
                    "seed_files": [],
                    "spec_mode": True,
                    "run_spec_enabled": True,
                    "run_spec_after_read": True,
                    "spec_resume": True,
                    "run_spec_path": ".local_micro_agent/run_spec.json",
                    "max_code_test_loops": 3,
                    "writable_files": ["*.txt"],
                    "seed_changes": [{"path": "b.txt", "content": "done-b"}],
                    "deterministic_test_decision": True,
                },
            }
            state = AgentState(repo_root=repo, user_request="test", max_loops=3)
            agent = MicroAgent(config, state)
            agent.models = _FailingModelManager()

            result = asyncio.run(agent.run())

            self.assertEqual(result.current, AgentStateName.DONE)
            self.assertGreater(result.fsm_step_count, result.loop_count)
            self.assertEqual((repo / "a.txt").read_text(), "done-a")
            self.assertEqual((repo / "b.txt").read_text(), "done-b")
            self.assertIn("Resumed run spec", "\n".join(result.notes))
            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual(persisted["search"]["graph_id"], "graph-0042")
            self.assertFalse((artifact_dir / "spec_graph_candidates.jsonl").exists())
            report = (artifact_dir / "spec_report.md").read_text()
            self.assertIn("status: `done`", report)
            self.assertIn("progress: 2/2 closed", report)
            self.assertIn("code_test_loop_count: 1", report)
            self.assertIn("fsm_step_count:", report)
            self.assertIn("max_code_test_loops: 3", report)
            self.assertIn("task-001", report)
            self.assertIn("task-002", report)
            progress_events = (artifact_dir / "spec_progress.jsonl").read_text()
            self.assertIn('"event": "scheduled"', progress_events)
            self.assertIn('"event": "closed"', progress_events)
            self.assertIn('"event": "done"', progress_events)
            self.assertIn('"fsm_step":', progress_events)

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

    def test_run_spec_generation_failure_does_not_keep_old_artifact_in_prompt(self) -> None:
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
            state = AgentState(repo_root=repo, user_request="new request")
            config = {
                "models": {"default": "failing"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "seed_files": ["target.py"],
                    "run_spec_after_read": True,
                    "run_spec_path": ".local_micro_agent/run_spec.json",
                },
            }
            agent = MicroAgent(config, state)
            agent.models = _FailingModelManager()

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
            self.assertIn("Run spec model call failed", "\n".join(state.notes))

    def test_spec_prompt_requests_task_graph_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            state.plan_markdown = "plan"
            state.file_context = [FileSnapshot("target.py", "value = 1\n")]

            messages = spec_prompt(state)

            self.assertIn("SPEC node", messages[0]["content"])
            self.assertIn("task_graph", messages[0]["content"])
            self.assertIn("target.py", messages[1]["content"])

    def test_spec_force_default_acceptance_kind_overrides_model_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_default_acceptance_kind": "metric",
                        "spec_force_default_acceptance_kind": True,
                    },
                },
                AgentState(repo_root=Path(tmp), user_request="test"),
            )

            spec = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "perf",
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "optimize metric",
                            "deliverables": ["target.py"],
                            "acceptance": {"kind": "synthesized", "commands": []},
                        }
                    ],
                }
            )

            self.assertIsNotNone(spec)
            task = spec["task_graph"][0]
            self.assertEqual(task["acceptance"]["kind"], "metric")
            self.assertEqual(task["acceptance"]["commands"], [])

    def test_metric_tactic_portfolio_forces_metric_and_strips_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "metric_regex": r"CYCLES:\s*(\d+)",
                        "spec_metric_requires_improvement": True,
                        "spec_tactic_portfolio": True,
                    },
                },
                AgentState(repo_root=Path(tmp), user_request="test"),
            )

            spec = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "portfolio",
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "alternate tactic",
                            "depends_on": ["task-000"],
                            "deliverables": ["target.py"],
                            "acceptance": {"kind": "synthesized"},
                        }
                    ],
                }
            )

            task = spec["task_graph"][0]
            self.assertEqual(task["depends_on"], [])
            self.assertEqual(task["acceptance"]["kind"], "metric")
            self.assertEqual(task["acceptance"]["commands"], [])

    def test_run_spec_normalization_preserves_design_contract_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {},
                },
                AgentState(repo_root=Path(tmp), user_request="test"),
            )

            spec = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "design-contract",
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "guard parser branch",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["existing parse outputs stay unchanged"],
                            "edit_scope": "Change one guarded branch in parse_item.",
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The fallback branch is unchanged.",
                            "fallback_plan": "Revert the guarded branch.",
                        }
                    ],
                }
            )

            task = spec["task_graph"][0]
            self.assertEqual(task["target_symbols"], ["parse_item"])
            self.assertEqual(task["target_regions"], ["target.py::parse_item"])
            self.assertEqual(task["preserved_invariants"], ["existing parse outputs stay unchanged"])
            self.assertEqual(task["validator"]["failure_condition"], "pytest fails")
            self.assertIn("fallback branch", task["correctness_rationale"])

    def test_spec_design_contract_gate_routes_abstract_task_to_spec_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["run_spec"] = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "perf",
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "Optimize hot path",
                            "deliverables": ["target.py"],
                            "acceptance": {"kind": "metric"},
                        }
                    ],
                }
            )

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.SPEC_SYNTH)
            persisted = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
            task = persisted["task_graph"][0]
            self.assertEqual(task["status"], "needs_design")
            self.assertIn("missing target_symbols", task["design_contract"]["issues"][0])
            self.assertIn("Spec rewrite focus", agent._spec_rewrite_focus_context())

    def test_spec_design_soft_fallback_schedules_one_attempt_before_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "active_todo_path": ".local_micro_agent/active_todo.json",
                        "spec_design_contract_gate": True,
                        "spec_design_contract_rewrite_attempts": 0,
                        "spec_gate_soft_fallback": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test", loop_count=0),
            )
            agent.state.scratch["run_spec"] = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "soft-design",
                    "invariants": ["public behavior stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "Improve component broadly",
                            "deliverables": ["target.py"],
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            agent._schedule_spec_task()
            self.assertEqual(agent.state.current, AgentStateName.SCHEDULE)
            spec = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
            task = spec["task_graph"][0]
            self.assertEqual(task["status"], "open")
            self.assertTrue(task["design_contract_advisory_once"])

            agent.state.scratch["run_spec"] = spec
            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.TASK_READ)
            scheduled = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
            self.assertEqual(scheduled["task_graph"][0]["status"], "in_progress")
            self.assertEqual(
                scheduled["task_graph"][0]["design_contract"]["status"],
                "soft_fallback_advisory",
            )

    def test_design_contract_exhaustion_skips_task_and_schedules_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_design_contract_rewrite_attempts": 0,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["run_spec"] = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "portable-spec",
                    "invariants": ["public behavior stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "Improve component broadly",
                            "deliverables": ["target.py"],
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        },
                        {
                            "task_id": "task-002",
                            "title": "guard parser branch",
                            "strategy_axis": "general_edit",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["existing parse outputs stay unchanged"],
                            "edit_scope": "Change one guarded branch in parse_item.",
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The fallback branch is unchanged.",
                            "fallback_plan": "Revert the guarded branch.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        },
                    ],
                }
            )

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.SCHEDULE)
            persisted = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
            self.assertEqual(persisted["task_graph"][0]["status"], "failed_design")

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.TASK_READ)
            self.assertEqual(agent.state.scratch["active_todo"]["spec_task_id"], "task-002")

    def test_all_design_contract_exhausted_tasks_stop_as_spec_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_design_contract_rewrite_attempts": 0,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["run_spec"] = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "portable-spec",
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "Improve component broadly",
                            "deliverables": ["target.py"],
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            agent._schedule_spec_task()
            self.assertEqual(agent.state.current, AgentStateName.SCHEDULE)

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.FAILED)
            persisted = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
            self.assertEqual(persisted["task_graph"][0]["status"], "failed_design")
            self.assertEqual(persisted["last_stop_reason"], "spec_design_contract_incomplete")

    def test_spec_scheduler_marks_partial_success_when_remaining_design_is_deferred(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_progress_path": ".local_micro_agent/spec_progress.jsonl",
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["run_spec"] = {
                "version": 2,
                "spec_id": "partial",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "status": "closed",
                        "title": "validated local edit",
                        "deliverables": ["target.py"],
                    },
                    {
                        "task_id": "task-002",
                        "status": "failed_design",
                        "title": "broad structural rewrite",
                        "deliverables": ["target.py"],
                    },
                ],
            }

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.DONE)
            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual(
                persisted["last_stop_reason"],
                "partial_success_design_deferred",
            )
            self.assertEqual(persisted["task_graph"][1]["status"], "deferred_design")
            self.assertEqual(persisted["progress"]["closed"], 1)
            self.assertEqual(persisted["progress"]["deferred"], 1)
            self.assertEqual(persisted["progress"]["failed"], 0)
            progress_event = json.loads(
                (artifact_dir / "spec_progress.jsonl").read_text().splitlines()[-1]
            )
            self.assertEqual(
                progress_event["stop_reason"],
                "partial_success_design_deferred",
            )

    def test_spec_scheduler_keeps_design_incomplete_failed_without_closed_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["run_spec"] = {
                "version": 2,
                "spec_id": "no-success",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "status": "failed_design",
                        "title": "broad structural rewrite",
                        "deliverables": ["target.py"],
                    }
                ],
            }

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.FAILED)
            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            self.assertEqual(
                persisted["last_stop_reason"],
                "spec_design_contract_incomplete",
            )
            self.assertEqual(persisted["task_graph"][0]["status"], "failed_design")

    def test_spec_blocked_reason_does_not_mark_partial_when_open_task_remains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=Path(tmp), user_request="test"),
            )
            tasks = [
                {"task_id": "task-001", "status": "closed"},
                {"task_id": "task-002", "status": "failed_design"},
                {"task_id": "task-003", "status": "open"},
            ]

            extra = agent._spec_blocked_event_extra(tasks)

            self.assertNotEqual(
                extra["stop_reason"],
                "partial_success_design_deferred",
            )

    def test_loop_cap_defers_spec_rewrite_instead_of_resetting_run_spec(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                task = {
                    "task_id": "task-001",
                    "title": "guard parser branch",
                    "strategy_axis": "general_edit",
                    "status": "in_progress",
                    "deliverables": ["target.py"],
                    "target_symbols": ["parse_item"],
                    "target_regions": ["target.py::parse_item"],
                    "preserved_invariants": ["existing parse outputs stay unchanged"],
                    "edit_scope": "Change one guarded branch in parse_item.",
                    "validator": {
                        "kind": "command",
                        "failure_condition": "pytest fails",
                    },
                    "correctness_rationale": "The fallback branch is unchanged.",
                    "fallback_plan": "Revert the guarded branch.",
                    "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                    "budget": {"attempts_max": 4, "attempts_used": 0},
                }
                spec = {
                    "version": 2,
                    "spec_id": "portable-spec",
                    "active_task_id": "task-001",
                    "task_graph": [task],
                }
                agent = MicroAgent(
                    {
                        "models": {},
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_design_contract_gate": True,
                            "spec_redesign_after_correctness_failures": 1,
                        },
                    },
                    AgentState(repo_root=repo, user_request="test", max_loops=1),
                )
                agent.state.scratch["run_spec"] = spec
                agent.state.scratch["current_spec_task"] = task
                agent.state.scratch["last_candidate_observation"] = {
                    "failure_class": "correctness_failure",
                    "summary": "assertion failed",
                }
                agent.state.test_results = [
                    TestResult(command="python -m pytest", exit_code=1, stderr="assertion failed")
                ]

                await agent._handle_spec_task_test_result(True)

                self.assertEqual(agent.state.current, AgentStateName.FAILED)
                self.assertNotEqual(agent.state.current, AgentStateName.SPEC_SYNTH)
                persisted = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
                self.assertEqual(persisted["spec_id"], "portable-spec")
                self.assertEqual(persisted["last_stop_reason"], "max_code_test_loops")
                self.assertEqual(persisted["task_graph"][0]["status"], "needs_design")
                self.assertEqual(
                    persisted["pending_spec_rewrite_reason"]["reason"],
                    "repeated_correctness_failure",
                )
                progress_events = (repo / ".local_micro_agent" / "spec_progress.jsonl").read_text()
                self.assertIn('"pending_spec_rewrite": true', progress_events)

        asyncio.run(run_case())

    def test_repeated_active_task_drift_routes_to_contract_rewrite(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                artifact_dir = repo / ".local_micro_agent"
                artifact_dir.mkdir()
                task = {
                    "task_id": "task-001",
                    "title": "guard parser branch",
                    "strategy_axis": "general_edit",
                    "status": "in_progress",
                    "deliverables": ["target.py"],
                    "target_symbols": ["parse_item"],
                    "target_regions": ["target.py::parse_item"],
                    "preserved_invariants": ["existing parse outputs stay unchanged"],
                    "edit_scope": "Change one guarded branch in parse_item.",
                    "validator": {"kind": "command", "failure_condition": "pytest fails"},
                    "budget": {"attempts_max": 4, "attempts_used": 0},
                }
                spec = {
                    "version": 2,
                    "spec_id": "drift-spec",
                    "active_task_id": "task-001",
                    "task_graph": [task],
                }
                active_todo = {
                    "todo_id": "task-001",
                    "spec_task_id": "task-001",
                    "status": "active",
                    "strategy_axis": "general_edit",
                    "source": "spec_scheduler",
                }
                attempts = [
                    {
                        "loop": index,
                        "todo_id": "task-001",
                        "status": "rejected_todo_scope_drift",
                        "failure_class": "active_task_drift",
                        "budget_counted": False,
                        "fingerprint": f"drift-{index}",
                        "summary": "outside active task",
                        "changes": [
                            {
                                "path": "target.py",
                                "target_region": "target.py::Parser.build",
                            }
                        ],
                    }
                    for index in range(3)
                ]
                (artifact_dir / "todo_attempts.jsonl").write_text(
                    "\n".join(json.dumps(attempt) for attempt in attempts) + "\n"
                )
                agent = MicroAgent(
                    {
                        "models": {},
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_progress_path": ".local_micro_agent/spec_progress.jsonl",
                            "spec_active_task_drift_streak_limit": 3,
                            "spec_active_task_drift_same_fingerprint_limit": 99,
                            "spec_active_task_drift_rewrite_attempts": 1,
                        },
                    },
                    AgentState(repo_root=repo, user_request="test", loop_count=2, max_loops=10),
                )
                agent.state.scratch["run_spec"] = spec
                agent.state.scratch["current_spec_task"] = task
                agent.state.scratch["active_todo"] = active_todo
                agent.state.scratch["last_candidate_observation"] = attempts[-1]
                agent.state.test_results = [
                    TestResult(command="python -m pytest", exit_code=1, stderr="scope drift")
                ]

                await agent._handle_spec_task_test_result(True)

                self.assertEqual(agent.state.current, AgentStateName.SPEC_SYNTH)
                self.assertEqual(agent.state.loop_count, 3)
                persisted = json.loads((artifact_dir / "run_spec.json").read_text())
                persisted_task = persisted["task_graph"][0]
                self.assertEqual(persisted_task["status"], "needs_contract_rewrite")
                self.assertEqual(
                    persisted_task["decision_hint"],
                    "repeated_active_task_drift_requires_contract_rewrite: CODE "
                    "kept violating or over-broadening this active task contract. "
                    "Rewrite the task as a smaller executable probe, or retire it "
                    "in favor of a different runnable task.",
                )
                self.assertEqual(
                    persisted_task["contract_rewrite"]["reason"],
                    "repeated_active_task_drift",
                )
                self.assertEqual(persisted_task["contract_rewrite"]["rewrite_attempts"], 1)
                progress_event = json.loads(
                    (artifact_dir / "spec_progress.jsonl").read_text().splitlines()[-1]
                )
                self.assertEqual(progress_event["event"], "drift_recovery")
                self.assertEqual(progress_event["action"], "rewrite")
                signature = json.loads(
                    (artifact_dir / "failure_signatures.jsonl").read_text().splitlines()[-1]
                )
                self.assertEqual(signature["failure_class"], "active_task_drift")
                self.assertEqual(signature["status"], "needs_contract_rewrite")
                self.assertEqual(signature["target_region_hash"], "ece45896")
                self.assertIn("drift_2", signature["cooldown_key"])
                self.assertNotIn("task-001", signature["cooldown_key"])
                self.assertEqual(
                    signature["drift_declared_regions"],
                    ["target.py::parse_item"],
                )
                self.assertEqual(
                    signature["drift_attempted_regions"],
                    ["target.py::Parser.build"],
                )
                self.assertEqual(
                    signature["drift_region_pairs"],
                    [
                        {
                            "declared": "target.py::parse_item",
                            "attempted": "target.py::Parser.build",
                        }
                    ],
                )

        asyncio.run(run_case())

    def test_active_task_drift_streak_resets_on_other_failure(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                artifact_dir = repo / ".local_micro_agent"
                artifact_dir.mkdir()
                task = {
                    "task_id": "task-001",
                    "status": "in_progress",
                    "title": "local task",
                    "strategy_axis": "general_edit",
                    "deliverables": ["target.py"],
                    "budget": {"attempts_max": 5, "attempts_used": 0},
                }
                spec = {
                    "version": 2,
                    "spec_id": "drift-reset",
                    "active_task_id": "task-001",
                    "task_graph": [task],
                }
                attempts = [
                    {
                        "loop": 0,
                        "todo_id": "task-001",
                        "status": "rejected_todo_scope_drift",
                        "failure_class": "active_task_drift",
                        "budget_counted": False,
                        "fingerprint": "drift-a",
                    },
                    {
                        "loop": 1,
                        "todo_id": "task-001",
                        "status": "rejected_correctness",
                        "failure_class": "correctness_failure",
                        "budget_counted": True,
                    },
                    {
                        "loop": 2,
                        "todo_id": "task-001",
                        "status": "rejected_todo_scope_drift",
                        "failure_class": "active_task_drift",
                        "budget_counted": False,
                        "fingerprint": "drift-a",
                    },
                ]
                (artifact_dir / "todo_attempts.jsonl").write_text(
                    "\n".join(json.dumps(attempt) for attempt in attempts) + "\n"
                )
                agent = MicroAgent(
                    {
                        "models": {},
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_progress_path": ".local_micro_agent/spec_progress.jsonl",
                            "spec_active_task_drift_streak_limit": 2,
                            "spec_active_task_drift_same_fingerprint_limit": 2,
                            "reflect_before_retry": False,
                        },
                    },
                    AgentState(repo_root=repo, user_request="test", loop_count=2, max_loops=10),
                )
                agent.state.scratch["run_spec"] = spec
                agent.state.scratch["current_spec_task"] = task
                agent.state.scratch["active_todo"] = {
                    "todo_id": "task-001",
                    "spec_task_id": "task-001",
                    "status": "active",
                    "source": "spec_scheduler",
                }
                agent.state.scratch["last_candidate_observation"] = attempts[-1]
                agent.state.test_results = [
                    TestResult(command="python -m pytest", exit_code=1, stderr="scope drift")
                ]

                await agent._handle_spec_task_test_result(True)

                self.assertEqual(agent.state.current, AgentStateName.CODE)
                persisted = json.loads((artifact_dir / "run_spec.json").read_text())
                self.assertEqual(persisted["task_graph"][0]["status"], "in_progress")
                progress_event = json.loads(
                    (artifact_dir / "spec_progress.jsonl").read_text().splitlines()[-1]
                )
                self.assertEqual(progress_event["event"], "retry")

        asyncio.run(run_case())

    def test_repeated_active_task_drift_defers_after_rewrite_budget(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                artifact_dir = repo / ".local_micro_agent"
                artifact_dir.mkdir()
                task = {
                    "task_id": "task-001",
                    "status": "in_progress",
                    "title": "local task",
                    "strategy_axis": "general_edit",
                    "deliverables": ["target.py"],
                    "contract_rewrite": {
                        "rewrite_attempt_key": "task-001",
                        "rewrite_attempts": 1,
                        "rewrite_attempts_max": 1,
                    },
                    "budget": {"attempts_max": 5, "attempts_used": 0},
                }
                spec = {
                    "version": 2,
                    "spec_id": "drift-defer",
                    "active_task_id": "task-001",
                    "task_graph": [task],
                }
                attempts = [
                    {
                        "loop": index,
                        "todo_id": "task-001",
                        "status": "rejected_todo_scope_drift",
                        "failure_class": "active_task_drift",
                        "budget_counted": False,
                        "fingerprint": "same-drift",
                    }
                    for index in range(2)
                ]
                (artifact_dir / "todo_attempts.jsonl").write_text(
                    "\n".join(json.dumps(attempt) for attempt in attempts) + "\n"
                )
                agent = MicroAgent(
                    {
                        "models": {},
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_progress_path": ".local_micro_agent/spec_progress.jsonl",
                            "spec_active_task_drift_streak_limit": 10,
                            "spec_active_task_drift_same_fingerprint_limit": 2,
                            "spec_active_task_drift_rewrite_attempts": 1,
                        },
                    },
                    AgentState(repo_root=repo, user_request="test", loop_count=1, max_loops=10),
                )
                agent.state.scratch["run_spec"] = spec
                agent.state.scratch["current_spec_task"] = task
                agent.state.scratch["active_todo"] = {
                    "todo_id": "task-001",
                    "spec_task_id": "task-001",
                    "status": "active",
                    "source": "spec_scheduler",
                }
                agent.state.scratch["last_candidate_observation"] = attempts[-1]
                agent.state.test_results = [
                    TestResult(command="python -m pytest", exit_code=1, stderr="scope drift")
                ]

                await agent._handle_spec_task_test_result(True)

                self.assertEqual(agent.state.current, AgentStateName.SCHEDULE)
                persisted = json.loads((artifact_dir / "run_spec.json").read_text())
                self.assertEqual(
                    persisted["task_graph"][0]["status"],
                    "deferred_contract_drift",
                )
                progress_event = json.loads(
                    (artifact_dir / "spec_progress.jsonl").read_text().splitlines()[-1]
                )
                self.assertEqual(progress_event["event"], "drift_recovery")
                self.assertEqual(progress_event["action"], "defer")
                signature = json.loads(
                    (artifact_dir / "failure_signatures.jsonl").read_text().splitlines()[-1]
                )
                self.assertEqual(signature["failure_class"], "active_task_drift")
                self.assertEqual(signature["status"], "deferred_contract_drift")
                self.assertEqual(signature["target_region_hash"], "b3652dd1")
                self.assertIn("same_drift", signature["cooldown_key"])
                self.assertNotIn("task-001", signature["cooldown_key"])

                agent._schedule_spec_task()

                blocked = json.loads(
                    (artifact_dir / "spec_progress.jsonl").read_text().splitlines()[-1]
                )
                self.assertEqual(agent.state.current, AgentStateName.FAILED)
                self.assertEqual(
                    blocked["stop_reason"],
                    "no_runnable_tasks_after_drift_deferred",
                )

        asyncio.run(run_case())

    def test_active_task_drift_saturation_defers_without_spending_rewrite(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                artifact_dir = repo / ".local_micro_agent"
                artifact_dir.mkdir()
                task = {
                    "task_id": "task-001",
                    "status": "in_progress",
                    "title": "local parser task",
                    "strategy_axis": "general_edit",
                    "deliverables": ["target.py"],
                    "target_regions": ["target.py::parse_item"],
                    "target_symbols": ["parse_item"],
                    "tactic_stage": "local_edit",
                    "validator": {"kind": "command"},
                    "budget": {"attempts_max": 5, "attempts_used": 0},
                }
                spec = {
                    "version": 2,
                    "spec_id": "drift-saturated",
                    "active_task_id": "task-001",
                    "task_graph": [task],
                }
                attempts = [
                    {
                        "loop": index,
                        "todo_id": "task-001",
                        "status": "rejected_todo_scope_drift",
                        "failure_class": "active_task_drift",
                        "budget_counted": False,
                        "fingerprint": "same-drift",
                        "tactic_stage": "local_edit",
                        "drift_cooldown_key": "ece45896:local_edit:same_drift",
                        "drift_attempted_regions": ["target.py::helper"],
                        "drift_region_pairs": [
                            {
                                "declared": "target.py::parse_item",
                                "attempted": "target.py::helper",
                            }
                        ],
                        "changes": [
                            {
                                "path": "target.py",
                                "target_region": "target.py::helper",
                            }
                        ],
                    }
                    for index in range(3)
                ]
                (artifact_dir / "todo_attempts.jsonl").write_text(
                    "\n".join(json.dumps(attempt) for attempt in attempts) + "\n"
                )
                agent = MicroAgent(
                    {
                        "models": {},
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_progress_path": ".local_micro_agent/spec_progress.jsonl",
                            "spec_active_task_drift_streak_limit": 10,
                            "spec_active_task_drift_same_fingerprint_limit": 2,
                            "spec_active_task_drift_rewrite_attempts": 1,
                            "spec_drift_saturation_threshold": 2,
                        },
                    },
                    AgentState(repo_root=repo, user_request="test", loop_count=2, max_loops=10),
                )
                agent.state.scratch["run_spec"] = spec
                agent.state.scratch["current_spec_task"] = task
                agent.state.scratch["active_todo"] = {
                    "todo_id": "task-001",
                    "spec_task_id": "task-001",
                    "status": "active",
                    "source": "spec_scheduler",
                }
                agent.state.scratch["last_candidate_observation"] = attempts[-1]
                agent.state.test_results = [
                    TestResult(command="python -m pytest", exit_code=1, stderr="scope drift")
                ]

                await agent._handle_spec_task_test_result(True)

                self.assertEqual(agent.state.current, AgentStateName.SCHEDULE)
                persisted = json.loads((artifact_dir / "run_spec.json").read_text())
                persisted_task = persisted["task_graph"][0]
                self.assertEqual(persisted_task["status"], "deferred_contract_drift")
                self.assertEqual(
                    persisted_task["contract_rewrite"]["reason"],
                    "drift_saturation",
                )
                self.assertEqual(persisted_task["contract_rewrite"]["rewrite_attempts"], 0)
                self.assertEqual(
                    persisted_task["contract_rewrite"]["drift_saturation"]["count"],
                    3,
                )
                progress_event = json.loads(
                    (artifact_dir / "spec_progress.jsonl").read_text().splitlines()[-1]
                )
                self.assertEqual(
                    progress_event["action"],
                    "rewrite_rejected_duplicate_drift",
                )
                self.assertEqual(progress_event["spec_budget_saved_by_drift_backoff"], 1)
                signature = json.loads(
                    (artifact_dir / "failure_signatures.jsonl").read_text().splitlines()[-1]
                )
                self.assertEqual(signature["reason"], "drift_saturation")
                self.assertEqual(signature["spec_budget_saved_by_drift_backoff"], 1)

        asyncio.run(run_case())

    def test_terminal_report_summarizes_active_task_drift_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_progress_path": ".local_micro_agent/spec_progress.jsonl",
                        "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                        "spec_terminal_state_path": ".local_micro_agent/terminal_state.json",
                    },
                },
                AgentState(repo_root=repo, user_request="test", loop_count=3),
            )
            spec = {
                "version": 2,
                "spec_id": "drift-report",
                "last_stop_reason": "no_runnable_tasks_after_drift_deferred",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "status": "deferred_contract_drift",
                        "title": "local task",
                    }
                ],
            }
            agent.state.scratch["run_spec"] = spec
            (artifact_dir / "candidates.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        {
                            "loop": index,
                            "todo_id": "task-001",
                            "status": "rejected_todo_scope_drift",
                            "failure_class": "active_task_drift",
                            "budget_counted": False,
                            "drift_attempted_regions": ["target.py::helper"],
                            "drift_region_pairs": [
                                {
                                    "declared": "target.py::parse_item",
                                    "attempted": "target.py::helper",
                                }
                            ],
                            "drift_cooldown_key": "ece45896:local_edit:scope_drift",
                        }
                    )
                    for index in range(3)
                )
                + "\n"
            )
            (artifact_dir / "spec_progress.jsonl").write_text(
                json.dumps(
                    {
                        "event": "drift_recovery",
                        "task_id": "task-001",
                        "action": "defer",
                    }
                )
                + "\n"
            )

            agent._persist_spec_report()

            terminal = json.loads((artifact_dir / "terminal_state.json").read_text())
            self.assertEqual(terminal["active_task_drift_count"], 3)
            self.assertEqual(terminal["max_active_task_drift_streak"], 3)
            self.assertEqual(terminal["drift_recovery_count"], 1)
            self.assertEqual(terminal["drift_deferred_task_ids"], ["task-001"])
            self.assertEqual(
                terminal["active_task_drift_attempted_region_counts"],
                {"target.py::helper": 3},
            )
            self.assertEqual(
                terminal["active_task_drift_region_pair_counts"],
                {"target.py::parse_item -> target.py::helper": 3},
            )
            self.assertEqual(terminal["same_region_drift_saturation_count"], 1)
            self.assertEqual(
                terminal["same_region_drift_saturated_keys"],
                {"ece45896:local_edit:scope_drift": 3},
            )
            self.assertEqual(terminal["targeted_rewrite_rejected_duplicate_drift"], 0)
            self.assertEqual(terminal["spec_budget_saved_by_drift_backoff"], 0)

    def test_terminal_report_summarizes_portfolio_recovery_exhaustion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_progress_path": ".local_micro_agent/spec_progress.jsonl",
                        "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                        "spec_terminal_state_path": ".local_micro_agent/terminal_state.json",
                        "spec_portfolio_recovery_rounds": 2,
                    },
                },
                AgentState(repo_root=repo, user_request="test", loop_count=6),
            )
            spec = {
                "version": 2,
                "spec_id": "portfolio-report",
                "last_stop_reason": "no_runnable_tasks_after_portfolio_exhausted",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "status": "deferred_portfolio_exhausted",
                        "title": "repeated local tactic",
                        "recovery_rounds": 2,
                    }
                ],
            }
            agent.state.scratch["run_spec"] = spec
            (artifact_dir / "spec_progress.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "event": "portfolio_reopened",
                                "reopened_tasks": ["task-001"],
                            }
                        ),
                        json.dumps(
                            {
                                "event": "portfolio_reopened",
                                "reopened_tasks": ["task-001"],
                            }
                        ),
                        json.dumps(
                            {
                                "event": "portfolio_exhausted",
                                "exhausted_tasks": ["task-001"],
                            }
                        ),
                    ]
                )
                + "\n"
            )

            agent._persist_spec_report()

            terminal = json.loads((artifact_dir / "terminal_state.json").read_text())
            self.assertEqual(terminal["portfolio_reopened_count"], 2)
            self.assertEqual(terminal["portfolio_exhausted_count"], 1)
            self.assertEqual(terminal["portfolio_exhausted_task_ids"], ["task-001"])
            self.assertEqual(terminal["max_portfolio_recovery_rounds"], 2)

    def test_needs_contract_rewrite_is_schedulable_without_design_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=repo, user_request="test"),
            )
            tasks = [
                {
                    "task_id": "task-001",
                    "status": "needs_contract_rewrite",
                    "title": "rewrite drifted contract",
                }
            ]

            schedulable = agent._schedulable_spec_tasks(tasks)

            self.assertEqual([task["task_id"] for task in schedulable], ["task-001"])

    def test_targeted_rewrite_rejects_repeated_drift_material_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {"spec_mode": True},
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            previous_spec = {
                "version": 2,
                "spec_id": "previous",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "status": "needs_contract_rewrite",
                        "target_regions": ["target.py::parse_item"],
                        "target_symbols": ["parse_item"],
                        "tactic_stage": "local_edit",
                        "deliverables": ["target.py"],
                        "validator": {"kind": "command"},
                    },
                    {
                        "task_id": "task-002",
                        "status": "open",
                        "target_regions": ["target.py::format_item"],
                    },
                ],
            }
            rewrite_spec = {
                "version": 2,
                "spec_id": "rewrite",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "status": "open",
                        "target_regions": ["target.py::parse_item"],
                        "target_symbols": ["parse_item"],
                        "tactic_stage": "local_edit",
                        "deliverables": ["target.py"],
                        "validator": {"kind": "command"},
                    },
                    {
                        "task_id": "task-002",
                        "status": "open",
                        "target_regions": ["target.py::format_item"],
                    },
                ],
            }

            issues = agent._spec_rewrite_graph_contract_issues(
                previous_spec,
                rewrite_spec,
                "task-001",
            )

            self.assertIn(
                "targeted SPEC rewrite repeated active-task drift material axes "
                "(target_regions, tactic_stage, validator.kind, deliverables) "
                "without a structurally different contract",
                issues,
            )

    def test_targeted_rewrite_allows_materially_different_drift_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {"spec_mode": True},
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            previous_spec = {
                "version": 2,
                "spec_id": "previous",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "status": "needs_contract_rewrite",
                        "target_regions": ["target.py::parse_item"],
                        "tactic_stage": "local_edit",
                        "deliverables": ["target.py"],
                        "validator": {"kind": "command"},
                    },
                    {
                        "task_id": "task-002",
                        "status": "open",
                        "target_regions": ["target.py::format_item"],
                    },
                ],
            }
            rewrite_spec = {
                "version": 2,
                "spec_id": "rewrite",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "status": "open",
                        "target_regions": ["target.py::helper"],
                        "tactic_stage": "local_edit",
                        "deliverables": ["target.py"],
                        "validator": {"kind": "command"},
                    },
                    {
                        "task_id": "task-002",
                        "status": "open",
                        "target_regions": ["target.py::format_item"],
                    },
                ],
            }

            issues = agent._spec_rewrite_graph_contract_issues(
                previous_spec,
                rewrite_spec,
                "task-001",
            )

            self.assertNotIn(
                "targeted SPEC rewrite repeated active-task drift material axes "
                "(target_regions, tactic_stage, validator.kind, deliverables) "
                "without a structurally different contract",
                issues,
            )

    def test_terminal_drift_streak_resets_on_non_drift_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_progress_path": ".local_micro_agent/spec_progress.jsonl",
                        "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                        "spec_terminal_state_path": ".local_micro_agent/terminal_state.json",
                    },
                },
                AgentState(repo_root=repo, user_request="test", loop_count=4),
            )
            agent.state.scratch["run_spec"] = {
                "version": 2,
                "spec_id": "drift-report",
                "task_graph": [{"task_id": "task-001", "status": "open"}],
            }
            events = [
                {
                    "loop": 0,
                    "todo_id": "task-001",
                    "status": "rejected_todo_scope_drift",
                    "failure_class": "active_task_drift",
                    "budget_counted": False,
                },
                {
                    "loop": 1,
                    "todo_id": "task-001",
                    "status": "rejected_correctness",
                    "failure_class": "correctness_failure",
                },
                {
                    "loop": 2,
                    "todo_id": "task-001",
                    "status": "rejected_todo_scope_drift",
                    "failure_class": "active_task_drift",
                    "budget_counted": False,
                },
                {
                    "loop": 3,
                    "todo_id": "task-001",
                    "status": "rejected_todo_scope_drift",
                    "failure_class": "active_task_drift",
                    "budget_counted": False,
                },
            ]
            (artifact_dir / "candidates.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n"
            )

            agent._persist_spec_report()

            terminal = json.loads((artifact_dir / "terminal_state.json").read_text())
            self.assertEqual(terminal["active_task_drift_count"], 3)
            self.assertEqual(terminal["max_active_task_drift_streak"], 2)

    def test_targeted_spec_rewrite_preserves_omitted_sibling_tasks(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                agent = MicroAgent(
                    {
                        "models": {},
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_design_contract_gate": True,
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                previous_spec = agent._normalize_run_spec(
                    {
                        "version": 2,
                        "spec_id": "previous",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "Rejected target",
                                "status": "needs_design",
                                "deliverables": ["target.py"],
                                "target_symbols": ["target"],
                                "target_regions": ["target.py::target"],
                                "preserved_invariants": ["existing behavior"],
                                "edit_scope": "Rewrite the whole target function.",
                                "risk_level": "structural",
                                "tactic_stage": "structural_probe",
                                "risk_evidence": {
                                    "field": "edit_scope",
                                    "quote": "Rewrite the whole target function.",
                                    "explanation": "Structural edit scope.",
                                },
                                "probe_plan": "Try a smaller guard.",
                                "invariant_evidence": ["tests pass"],
                                "validator": {
                                    "kind": "command",
                                    "failure_condition": "pytest fails",
                                },
                                "correctness_rationale": "Preserve behavior.",
                                "fallback_plan": "Revert.",
                                "rollback_or_shrink_plan": "Shrink to one branch.",
                                "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                            },
                            {
                                "task_id": "task-002",
                                "title": "Sibling local edit",
                                "status": "open",
                                "deliverables": ["target.py"],
                                "target_symbols": ["parse_item"],
                                "target_regions": ["target.py::parse_item"],
                                "preserved_invariants": ["existing parse outputs"],
                                "edit_scope": "Cache one local value in parse_item.",
                                "risk_level": "local",
                                "tactic_stage": "local_edit",
                                "risk_evidence": {
                                    "field": "edit_scope",
                                    "quote": "Cache one local value in parse_item.",
                                    "explanation": "Local binding only.",
                                },
                                "validator": {
                                    "kind": "command",
                                    "failure_condition": "pytest fails",
                                },
                                "correctness_rationale": "No behavior change.",
                                "fallback_plan": "Remove the binding.",
                                "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                            },
                        ],
                    }
                )
                replacement_spec = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "replacement",
                        "task_graph": [
                            {
                                "task_id": "task-099",
                                "replaces_task_id": "task-001",
                                "title": "Narrow replacement",
                                "deliverables": ["target.py"],
                                "target_symbols": ["target"],
                                "target_regions": ["target.py::target"],
                                "preserved_invariants": ["existing behavior"],
                                "edit_scope": "Change one guarded branch in target.",
                                "risk_level": "local",
                                "tactic_stage": "local_edit",
                                "risk_evidence": {
                                    "field": "edit_scope",
                                    "quote": "Change one guarded branch in target.",
                                    "explanation": "One branch only.",
                                },
                                "validator": {
                                    "kind": "command",
                                    "failure_condition": "pytest fails",
                                },
                                "correctness_rationale": "The fallback branch remains.",
                                "fallback_plan": "Revert the guarded branch.",
                                "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                            }
                        ],
                    }
                )
                models = _RoleModelManager({"reasoner": replacement_spec})
                agent.models = models
                agent.state.scratch["run_spec"] = previous_spec
                agent.state.scratch["spec_rewrite_focus"] = "rewrite task-001"
                agent.state.scratch["spec_rewrite_target_task_id"] = "task-001"

                await agent._maybe_refresh_run_spec(force=True)

                persisted = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
                task_ids = [task["task_id"] for task in persisted["task_graph"]]
                self.assertEqual(task_ids, ["task-099", "task-002"])
                self.assertEqual(persisted["task_graph"][0]["replaces_task_id"], "task-001")
                self.assertEqual(persisted["task_graph"][1]["status"], "open")
                self.assertTrue(persisted["task_graph"][1]["portfolio_preserved_after_rewrite"])
                progress = (repo / ".local_micro_agent" / "spec_progress.jsonl").read_text()
                self.assertIn('"event": "rewrite_merged"', progress)
                prompt = models.seen["reasoner"][0][1]["content"]
                self.assertIn("Existing task graph before this targeted SPEC rewrite", prompt)
                self.assertIn("task-002", prompt)
                self.assertIn("rewrite target is task-001", prompt)

        asyncio.run(run_case())

    def test_replacement_task_inherits_design_rewrite_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_design_contract_rewrite_attempts": 2,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["spec_design_contract_rewrite_attempts_by_task"] = {
                "task-001": 2
            }
            agent.state.scratch["run_spec"] = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "portfolio",
                    "task_graph": [
                        {
                            "task_id": "task-099",
                            "replaces_task_id": "task-001",
                            "title": "Broad replacement",
                            "deliverables": ["target.py"],
                            "target_symbols": ["target"],
                            "target_regions": ["target.py::target"],
                            "preserved_invariants": ["existing behavior"],
                            "edit_scope": "Replace the whole target function.",
                            "risk_level": "structural",
                            "tactic_stage": "structural_probe",
                            "risk_evidence": {
                                "field": "edit_scope",
                                "quote": "Replace the whole target function.",
                                "explanation": "Structural scope.",
                            },
                            "probe_plan": "Try the replacement.",
                            "invariant_evidence": ["tests pass"],
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "Preserve behavior.",
                            "fallback_plan": "Revert.",
                            "rollback_or_shrink_plan": "Shrink.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        },
                        {
                            "task_id": "task-002",
                            "title": "Sibling local edit",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["existing parse outputs"],
                            "edit_scope": "Cache one local value in parse_item.",
                            "risk_level": "local",
                            "tactic_stage": "local_edit",
                            "risk_evidence": {
                                "field": "edit_scope",
                                "quote": "Cache one local value in parse_item.",
                                "explanation": "Local binding only.",
                            },
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "No behavior change.",
                            "fallback_plan": "Remove the binding.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        },
                    ],
                }
            )

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.SCHEDULE)
            persisted = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
            self.assertEqual(persisted["task_graph"][0]["status"], "failed_design")
            self.assertEqual(
                persisted["task_graph"][0]["design_contract"]["rewrite_attempt_key"],
                "task-001",
            )

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.TASK_READ)
            self.assertEqual(agent.state.scratch["active_todo"]["spec_task_id"], "task-002")

    def test_targeted_spec_rewrite_rejects_single_broad_structural_graph(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                agent = MicroAgent(
                    {
                        "models": {},
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_design_contract_gate": True,
                            "spec_tactic_portfolio": True,
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                previous_spec = agent._normalize_run_spec(
                    {
                        "version": 2,
                        "spec_id": "previous",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "Rejected target",
                                "status": "needs_design",
                                "deliverables": ["target.py"],
                                "target_symbols": ["target"],
                                "target_regions": ["target.py::target"],
                                "preserved_invariants": ["existing behavior"],
                                "edit_scope": "Rewrite the whole target function.",
                                "risk_level": "structural",
                                "tactic_stage": "structural_probe",
                                "risk_evidence": {
                                    "field": "edit_scope",
                                    "quote": "Rewrite the whole target function.",
                                    "explanation": "Structural edit scope.",
                                },
                                "probe_plan": "Try a smaller guard.",
                                "invariant_evidence": ["tests pass"],
                                "validator": {
                                    "kind": "command",
                                    "failure_condition": "pytest fails",
                                },
                                "correctness_rationale": "Preserve behavior.",
                                "fallback_plan": "Revert.",
                                "rollback_or_shrink_plan": "Shrink to one branch.",
                                "acceptance": {"kind": "metric"},
                            }
                        ],
                    }
                )
                broad_rewrite = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "collapsed",
                        "task_graph": [
                            {
                                "task_id": "task-999",
                                "replaces_task_id": "task-001",
                                "title": "Broad structural replacement",
                                "deliverables": ["target.py"],
                                "target_symbols": ["target"],
                                "target_regions": ["target.py::target"],
                                "preserved_invariants": ["existing behavior"],
                                "edit_scope": "Replace the whole target function.",
                                "risk_level": "structural",
                                "tactic_stage": "structural_probe",
                                "risk_evidence": {
                                    "field": "edit_scope",
                                    "quote": "Replace the whole target function.",
                                    "explanation": "Structural scope.",
                                },
                                "probe_plan": "Try the replacement.",
                                "invariant_evidence": ["tests pass"],
                                "validator": {
                                    "kind": "metric",
                                    "failure_condition": "metric does not improve",
                                },
                                "correctness_rationale": "Preserve behavior.",
                                "fallback_plan": "Revert.",
                                "rollback_or_shrink_plan": "Shrink.",
                                "acceptance": {"kind": "metric"},
                            }
                        ],
                    }
                )
                agent.models = _StaticModelManager(broad_rewrite)
                agent.state.scratch["run_spec"] = previous_spec
                agent.state.scratch["spec_rewrite_focus"] = "rewrite task-001"
                agent.state.scratch["spec_rewrite_target_task_id"] = "task-001"

                await agent._maybe_refresh_run_spec(force=True)

                persisted = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
                self.assertEqual(persisted["spec_id"], "previous")
                self.assertEqual(
                    persisted["task_graph"][0]["status"],
                    "deferred_design_invalid",
                )
                progress = (repo / ".local_micro_agent" / "spec_progress.jsonl").read_text()
                self.assertIn('"event": "design_rejected"', progress)
                self.assertIn('"action": "defer_target_preserve_siblings"', progress)
                self.assertIn("only one broad structural task", progress)
                graph_events = [
                    json.loads(line)
                    for line in (
                        repo / ".local_micro_agent" / "spec_graph_candidates.jsonl"
                    ).read_text().splitlines()
                    if line.strip()
                ]
                self.assertEqual(len(graph_events), 1)
                self.assertEqual(graph_events[0]["event"], "candidate_rejected")
                self.assertEqual(graph_events[0]["status"], "rejected_graph_contract")
                self.assertEqual(graph_events[0]["origin"], "targeted_design_rewrite")
                self.assertEqual(
                    graph_events[0]["issue_codes"],
                    ["single_broad_structural_task"],
                )
                signatures = [
                    json.loads(line)
                    for line in (
                        repo / ".local_micro_agent" / "failure_signatures.jsonl"
                    ).read_text().splitlines()
                    if line.strip()
                ]
                self.assertEqual(len(signatures), 1)
                signature = signatures[0]
                self.assertEqual(signature["phase"], "graph_rewrite")
                self.assertEqual(signature["failure_class"], "design_rewrite_invalid")
                self.assertEqual(
                    signature["issue_code"],
                    "single_broad_structural_task",
                )
                self.assertEqual(signature["issue_scope"], "target_task")
                self.assertIn(
                    "structural_probe:single_broad_structural_task",
                    signature["cooldown_key"],
                )
                self.assertNotIn("task-001", signature["cooldown_key"])

        asyncio.run(run_case())

    def test_targeted_spec_rewrite_quality_retry_merges_fixed_rewrite(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                agent = MicroAgent(
                    {
                        "models": {"reasoner": "reasoner-model"},
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_quality_gate": True,
                            "spec_quality_rewrite_attempts": 2,
                            "spec_targeted_rewrite_quality_rewrite_attempts": 1,
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                previous_spec = agent._normalize_run_spec(
                    {
                        "version": 2,
                        "spec_id": "previous",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "Rejected target",
                                "status": "needs_design",
                                "deliverables": ["target.py"],
                                "target_symbols": ["target"],
                                "target_regions": ["target.py::target"],
                                "preserved_invariants": ["existing behavior"],
                                "edit_scope": "Change one guarded branch in target.",
                                "risk_level": "local",
                                "tactic_stage": "local_edit",
                                "risk_evidence": {
                                    "field": "edit_scope",
                                    "quote": "one guarded branch",
                                    "explanation": "Local branch edit.",
                                },
                                "validator": {
                                    "kind": "command",
                                    "failure_condition": "pytest fails",
                                },
                                "correctness_rationale": "No behavior change.",
                                "fallback_plan": "Shrink to one guarded branch if tests fail.",
                                "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                            },
                            {
                                "task_id": "task-002",
                                "title": "Sibling local edit",
                                "status": "open",
                                "deliverables": ["target.py"],
                                "target_symbols": ["parse_item"],
                                "target_regions": ["target.py::parse_item"],
                                "preserved_invariants": ["existing parse outputs"],
                                "edit_scope": "Cache one local value in parse_item.",
                                "risk_level": "local",
                                "tactic_stage": "local_edit",
                                "risk_evidence": {
                                    "field": "edit_scope",
                                    "quote": "Cache one local value in parse_item.",
                                    "explanation": "Local binding only.",
                                },
                                "validator": {
                                    "kind": "command",
                                    "failure_condition": "pytest fails",
                                },
                                "correctness_rationale": "No behavior change.",
                                "fallback_plan": "Revert by removing the local binding.",
                                "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                            },
                        ],
                    }
                )
                invalid_rewrite = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "invalid-quality",
                        "task_graph": [
                            {
                                "task_id": "task-099",
                                "replaces_task_id": "task-001",
                                "title": "Missing target region",
                                "deliverables": ["target.py"],
                                "target_symbols": ["target"],
                                "target_regions": [],
                                "preserved_invariants": ["existing behavior"],
                                "edit_scope": "Change one guarded branch in target.",
                                "risk_level": "local",
                                "tactic_stage": "local_edit",
                                "risk_evidence": {
                                    "field": "edit_scope",
                                    "quote": "one guarded branch",
                                    "explanation": "Local branch edit.",
                                },
                                "validator": {
                                    "kind": "command",
                                    "failure_condition": "pytest fails",
                                },
                                "correctness_rationale": "No behavior change.",
                                "fallback_plan": "Shrink to one guarded branch if tests fail.",
                                "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                            }
                        ],
                    }
                )
                fixed_rewrite = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "fixed-quality",
                        "task_graph": [
                            {
                                "task_id": "task-099",
                                "replaces_task_id": "task-001",
                                "title": "Fixed replacement",
                                "deliverables": ["target.py"],
                                "target_symbols": ["target"],
                                "target_regions": ["target.py::target"],
                                "preserved_invariants": ["existing behavior"],
                                "edit_scope": "Change one guarded branch in target.",
                                "risk_level": "local",
                                "tactic_stage": "local_edit",
                                "risk_evidence": {
                                    "field": "edit_scope",
                                    "quote": "one guarded branch",
                                    "explanation": "Local branch edit.",
                                },
                                "validator": {
                                    "kind": "command",
                                    "failure_condition": "pytest fails",
                                },
                                "correctness_rationale": "No behavior change.",
                                "fallback_plan": "Shrink to one guarded branch if tests fail.",
                                "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                            }
                        ],
                    }
                )
                models = _RoleSequenceModelManager(
                    {"reasoner": [invalid_rewrite, fixed_rewrite]}
                )
                agent.models = models
                agent.state.scratch["run_spec"] = previous_spec
                agent.state.scratch["spec_rewrite_focus"] = "rewrite task-001"
                agent.state.scratch["spec_rewrite_target_task_id"] = "task-001"

                await agent._maybe_refresh_run_spec(force=True)

                persisted = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
                self.assertEqual(persisted["spec_id"], "fixed-quality")
                self.assertEqual(
                    [task["task_id"] for task in persisted["task_graph"]],
                    ["task-099", "task-002"],
                )
                self.assertEqual(persisted["task_graph"][0]["replaces_task_id"], "task-001")
                self.assertTrue(persisted["task_graph"][1]["portfolio_preserved_after_rewrite"])
                self.assertEqual(len(models.seen["reasoner"]), 2)
                report = json.loads(
                    (repo / ".local_micro_agent" / "spec_quality_report.json").read_text()
                )
                self.assertEqual(report["status"], "pass")
                progress = (repo / ".local_micro_agent" / "spec_progress.jsonl").read_text()
                self.assertIn('"event": "quality_rejected"', progress)
                self.assertIn('"event": "rewrite_merged"', progress)
                graph_events = [
                    json.loads(line)
                    for line in (
                        repo / ".local_micro_agent" / "spec_graph_candidates.jsonl"
                    ).read_text().splitlines()
                    if line.strip()
                ]
                self.assertEqual(
                    [(event["event"], event["status"]) for event in graph_events],
                    [
                        ("candidate_rejected", "rejected_quality"),
                        ("candidate_selected", "selected"),
                    ],
                )

        asyncio.run(run_case())

    def test_targeted_spec_rewrite_quality_failure_defers_target_and_preserves_sibling(
        self,
    ) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                agent = MicroAgent(
                    {
                        "models": {"reasoner": "reasoner-model"},
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_quality_gate": True,
                            "spec_quality_rewrite_attempts": 2,
                            "spec_targeted_rewrite_quality_rewrite_attempts": 0,
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                previous_spec = agent._normalize_run_spec(
                    {
                        "version": 2,
                        "spec_id": "previous",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "Rejected target",
                                "status": "needs_design",
                                "deliverables": ["target.py"],
                                "target_symbols": ["target"],
                                "target_regions": ["target.py::target"],
                                "preserved_invariants": ["existing behavior"],
                                "edit_scope": "Change one guarded branch in target.",
                                "risk_level": "local",
                                "tactic_stage": "local_edit",
                                "risk_evidence": {
                                    "field": "edit_scope",
                                    "quote": "one guarded branch",
                                    "explanation": "Local branch edit.",
                                },
                                "validator": {
                                    "kind": "command",
                                    "failure_condition": "pytest fails",
                                },
                                "correctness_rationale": "No behavior change.",
                                "fallback_plan": "Shrink to one guarded branch if tests fail.",
                                "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                            },
                            {
                                "task_id": "task-002",
                                "title": "Sibling local edit",
                                "status": "open",
                                "deliverables": ["target.py"],
                                "target_symbols": ["parse_item"],
                                "target_regions": ["target.py::parse_item"],
                                "preserved_invariants": ["existing parse outputs"],
                                "edit_scope": "Cache one local value in parse_item.",
                                "risk_level": "local",
                                "tactic_stage": "local_edit",
                                "risk_evidence": {
                                    "field": "edit_scope",
                                    "quote": "Cache one local value in parse_item.",
                                    "explanation": "Local binding only.",
                                },
                                "validator": {
                                    "kind": "command",
                                    "failure_condition": "pytest fails",
                                },
                                "correctness_rationale": "No behavior change.",
                                "fallback_plan": "Revert by removing the local binding.",
                                "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                            },
                        ],
                    }
                )
                invalid_rewrite = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "invalid-quality",
                        "task_graph": [
                            {
                                "task_id": "task-099",
                                "replaces_task_id": "task-001",
                                "title": "Missing target region",
                                "deliverables": ["target.py"],
                                "target_symbols": ["target"],
                                "target_regions": [],
                                "preserved_invariants": ["existing behavior"],
                                "edit_scope": "Change one guarded branch in target.",
                                "risk_level": "local",
                                "tactic_stage": "local_edit",
                                "risk_evidence": {
                                    "field": "edit_scope",
                                    "quote": "one guarded branch",
                                    "explanation": "Local branch edit.",
                                },
                                "validator": {
                                    "kind": "command",
                                    "failure_condition": "pytest fails",
                                },
                                "correctness_rationale": "No behavior change.",
                                "fallback_plan": "Shrink to one guarded branch if tests fail.",
                                "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                            }
                        ],
                    }
                )
                models = _RoleSequenceModelManager({"reasoner": [invalid_rewrite]})
                agent.models = models
                agent.state.scratch["run_spec"] = previous_spec
                agent.state.scratch["spec_rewrite_focus"] = "rewrite task-001"
                agent.state.scratch["spec_rewrite_target_task_id"] = "task-001"

                await agent._maybe_refresh_run_spec(force=True)

                persisted = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
                self.assertEqual(persisted["spec_id"], "previous")
                self.assertEqual(
                    persisted["task_graph"][0]["status"],
                    "deferred_design_invalid",
                )
                self.assertEqual(persisted["task_graph"][1]["status"], "open")
                self.assertEqual(
                    persisted["last_targeted_rewrite_quality_rejected"]["quality_issue_codes"],
                    ["target_region_count"],
                )
                self.assertEqual(len(models.seen["reasoner"]), 1)
                progress = (repo / ".local_micro_agent" / "spec_progress.jsonl").read_text()
                self.assertIn('"action": "defer_target_preserve_siblings"', progress)
                self.assertIn(
                    '"targeted_rewrite_action": "targeted_rewrite_quality_rejected"',
                    progress,
                )
                self.assertIn('"quality_issue_codes": ["target_region_count"]', progress)
                signatures = [
                    json.loads(line)
                    for line in (
                        repo / ".local_micro_agent" / "failure_signatures.jsonl"
                    ).read_text().splitlines()
                    if line.strip()
                ]
                self.assertEqual(signatures[-1]["phase"], "graph_rewrite")
                self.assertEqual(signatures[-1]["failure_class"], "design_rewrite_invalid")
                self.assertEqual(signatures[-1]["issue_code"], "target_region_count")
                self.assertEqual(signatures[-1]["issue_scope"], "target_task")

                agent._schedule_spec_task()

                self.assertEqual(agent.state.current, AgentStateName.TASK_READ)
                self.assertEqual(agent.state.scratch["active_todo"]["spec_task_id"], "task-002")

        asyncio.run(run_case())

    def test_targeted_drift_rewrite_quality_failure_defers_contract_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            previous_spec = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "previous",
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "Drift target",
                            "status": "needs_contract_rewrite",
                            "deliverables": ["target.py"],
                            "target_symbols": ["target"],
                            "target_regions": ["target.py::target"],
                            "preserved_invariants": ["existing behavior"],
                            "edit_scope": "Change one guarded branch in target.",
                            "risk_level": "local",
                            "tactic_stage": "local_edit",
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "contract_rewrite": {
                                "status": "requested",
                                "reason": "repeated_active_task_drift",
                            },
                        }
                    ],
                }
            )
            quality_report = {
                "status": "fail",
                "attempt": 1,
                "issue_codes": ["target_region_count"],
                "issues": [{"code": "target_region_count", "task_id": "task-001"}],
            }

            rejected = agent._reject_targeted_rewrite_for_quality_failure(
                previous_spec,
                "task-001",
                quality_report,
            )

            self.assertTrue(rejected)
            persisted = json.loads((artifact_dir / "run_spec.json").read_text())
            task = persisted["task_graph"][0]
            self.assertEqual(task["status"], "deferred_contract_drift")
            self.assertEqual(task["contract_rewrite"]["reason"], "quality_gate_rejected")
            progress = (artifact_dir / "spec_progress.jsonl").read_text()
            self.assertIn('"event": "drift_recovery"', progress)
            signature = json.loads(
                (artifact_dir / "failure_signatures.jsonl").read_text().splitlines()[-1]
            )
            self.assertEqual(signature["phase"], "active_task")
            self.assertEqual(signature["failure_class"], "active_task_drift")
            self.assertEqual(signature["issue_code"], "target_region_count")
            self.assertEqual(signature["issue_scope"], "target_task")

    def test_targeted_rewrite_quality_failure_missing_target_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            previous_spec = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "previous",
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "Only task",
                            "status": "open",
                            "deliverables": ["target.py"],
                            "target_symbols": ["target"],
                            "target_regions": ["target.py::target"],
                            "preserved_invariants": ["existing behavior"],
                            "edit_scope": "Change one guarded branch in target.",
                            "risk_level": "local",
                            "tactic_stage": "local_edit",
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                        }
                    ],
                }
            )
            quality_report = {
                "status": "fail",
                "attempt": 1,
                "issue_codes": ["target_region_count"],
                "issues": [{"code": "target_region_count", "task_id": "task-404"}],
            }

            rejected = agent._reject_targeted_rewrite_for_quality_failure(
                previous_spec,
                "task-404",
                quality_report,
            )

            self.assertFalse(rejected)
            self.assertFalse((artifact_dir / "run_spec.json").exists())
            self.assertFalse((artifact_dir / "spec_progress.jsonl").exists())
            self.assertFalse((artifact_dir / "failure_signatures.jsonl").exists())
            self.assertNotEqual(agent.state.current, AgentStateName.SCHEDULE)

    def test_targeted_graph_rewrite_missing_target_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            previous_spec = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "previous",
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "Only task",
                            "status": "open",
                            "target_regions": ["target.py::target"],
                        }
                    ],
                }
            )

            rejected = agent._reject_spec_graph_rewrite(
                previous_spec,
                "task-404",
                ["targeted SPEC rewrite target task was not found in previous graph"],
            )

            self.assertFalse(rejected)
            self.assertFalse((artifact_dir / "run_spec.json").exists())
            self.assertFalse((artifact_dir / "spec_progress.jsonl").exists())
            self.assertFalse((artifact_dir / "failure_signatures.jsonl").exists())
            self.assertNotEqual(agent.state.current, AgentStateName.SCHEDULE)

    def test_targeted_rewrite_transaction_preserves_sibling_and_ignores_non_target_tasks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {"spec_mode": True},
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            previous_spec = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "previous",
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "status": "needs_contract_rewrite",
                            "target_regions": ["target.py::parse_item"],
                            "tactic_stage": "local_edit",
                            "deliverables": ["target.py"],
                            "validator": {"kind": "command"},
                        },
                        {
                            "task_id": "task-002",
                            "status": "open",
                            "target_regions": ["target.py::format_item"],
                            "budget": {"attempts_used": 1, "attempts_max": 3},
                            "last_observation": {"failure_class": "no_improvement"},
                        },
                    ],
                }
            )
            rewrite_spec = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "rewrite",
                    "task_graph": [
                        {
                            "task_id": "task-099",
                            "replaces_task_id": "task-001",
                            "status": "open",
                            "target_regions": ["target.py::parse_helper"],
                            "tactic_stage": "local_edit",
                            "deliverables": ["target.py"],
                            "validator": {"kind": "command"},
                        },
                        {
                            "task_id": "task-new",
                            "status": "open",
                            "target_regions": ["target.py::unrelated"],
                        },
                    ],
                }
            )

            transaction = agent._targeted_rewrite_transaction(
                previous_spec,
                rewrite_spec,
                "task-001",
            )

            self.assertEqual(transaction.replacement_task_ids, ["task-099"])
            self.assertEqual(transaction.ignored_non_target_task_ids, ["task-new"])
            self.assertEqual(transaction.preserved_sibling_task_ids, ["task-002"])
            self.assertEqual(transaction.issues, [])
            merged_tasks = transaction.merged_spec["task_graph"]
            self.assertEqual([task["task_id"] for task in merged_tasks], ["task-099", "task-002"])
            self.assertEqual(merged_tasks[1]["budget"]["attempts_used"], 1)
            self.assertEqual(
                merged_tasks[1]["last_observation"]["failure_class"],
                "no_improvement",
            )

    def test_targeted_graph_reject_defers_target_and_reports_preserved_siblings(
        self,
    ) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                agent = MicroAgent(
                    {
                        "models": {},
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_progress_path": ".local_micro_agent/spec_progress.jsonl",
                            "spec_report_path": ".local_micro_agent/spec_report.md",
                            "spec_terminal_state_path": ".local_micro_agent/terminal_state.json",
                            "spec_tactic_portfolio": True,
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                previous_spec = agent._normalize_run_spec(
                    {
                        "version": 2,
                        "spec_id": "previous",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "Target",
                                "status": "needs_design",
                                "target_regions": ["target.py::target"],
                            },
                            {
                                "task_id": "task-002",
                                "title": "Sibling",
                                "status": "open",
                                "target_regions": ["target.py::sibling"],
                            },
                        ],
                    }
                )
                collapsed_rewrite = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "collapsed",
                        "task_graph": [
                            {
                                "task_id": "task-099",
                                "replaces_task_id": "task-001",
                                "status": "closed",
                                "target_regions": ["target.py::target"],
                            }
                        ],
                    }
                )
                agent.models = _StaticModelManager(collapsed_rewrite)
                agent.state.scratch["run_spec"] = previous_spec
                agent.state.scratch["spec_rewrite_focus"] = "rewrite task-001"
                agent.state.scratch["spec_rewrite_target_task_id"] = "task-001"

                await agent._maybe_refresh_run_spec(force=True)

                artifact_dir = repo / ".local_micro_agent"
                persisted = json.loads((artifact_dir / "run_spec.json").read_text())
                self.assertEqual(persisted["task_graph"][0]["status"], "deferred_design_invalid")
                self.assertEqual(persisted["task_graph"][1]["status"], "open")
                self.assertEqual(
                    persisted["last_targeted_rewrite_rejected"]["preserved_sibling_task_ids"],
                    ["task-002"],
                )
                progress = (artifact_dir / "spec_progress.jsonl").read_text()
                self.assertIn('"event": "design_rejected"', progress)
                self.assertIn('"action": "defer_target_preserve_siblings"', progress)
                self.assertIn("portfolio below", progress)
                self.assertNotIn('"event": "graph_reseed_requested"', progress)
                graph_events = [
                    json.loads(line)
                    for line in (artifact_dir / "spec_graph_candidates.jsonl").read_text().splitlines()
                    if line.strip()
                ]
                self.assertEqual(graph_events[-1]["issue_codes"], ["portfolio_collapsed_below_min_runnable"])

                agent._schedule_spec_task()
                self.assertEqual(agent.state.current, AgentStateName.TASK_READ)
                self.assertEqual(agent.state.scratch["active_todo"]["spec_task_id"], "task-002")
                agent._persist_spec_report()

                terminal = json.loads((artifact_dir / "terminal_state.json").read_text())
                self.assertEqual(terminal["targeted_rewrite_target_local_rejection_count"], 1)
                self.assertEqual(
                    terminal["targeted_rewrite_preserved_sibling_task_ids"],
                    ["task-002"],
                )
                report = (artifact_dir / "spec_report.md").read_text()
                self.assertIn("## Targeted Rewrite Recovery", report)
                self.assertIn("target_local_rejection_count: 1", report)

        asyncio.run(run_case())

    def test_targeted_drift_material_reject_defers_only_target(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                agent = MicroAgent(
                    {
                        "models": {},
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_progress_path": ".local_micro_agent/spec_progress.jsonl",
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                previous_spec = agent._normalize_run_spec(
                    {
                        "version": 2,
                        "spec_id": "previous",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "status": "needs_contract_rewrite",
                                "target_regions": ["target.py::target"],
                                "tactic_stage": "local_edit",
                                "deliverables": ["target.py"],
                                "validator": {"kind": "command"},
                            },
                            {
                                "task_id": "task-002",
                                "status": "open",
                                "target_regions": ["target.py::sibling"],
                            },
                        ],
                    }
                )
                repeated_rewrite = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "repeated",
                        "task_graph": [
                            {
                                "task_id": "task-099",
                                "replaces_task_id": "task-001",
                                "status": "open",
                                "target_regions": ["target.py::target"],
                                "tactic_stage": "local_edit",
                                "deliverables": ["target.py"],
                                "validator": {"kind": "command"},
                            }
                        ],
                    }
                )
                agent.models = _StaticModelManager(repeated_rewrite)
                agent.state.scratch["run_spec"] = previous_spec
                agent.state.scratch["spec_rewrite_focus"] = "rewrite task-001"
                agent.state.scratch["spec_rewrite_target_task_id"] = "task-001"

                await agent._maybe_refresh_run_spec(force=True)

                artifact_dir = repo / ".local_micro_agent"
                persisted = json.loads((artifact_dir / "run_spec.json").read_text())
                self.assertEqual(persisted["task_graph"][0]["status"], "deferred_contract_drift")
                self.assertEqual(persisted["task_graph"][1]["status"], "open")
                progress = (artifact_dir / "spec_progress.jsonl").read_text()
                self.assertIn('"event": "drift_recovery"', progress)
                self.assertIn('"action": "defer_target_preserve_siblings"', progress)
                self.assertNotIn('"event": "graph_reseed_requested"', progress)
                signature = json.loads(
                    (artifact_dir / "failure_signatures.jsonl").read_text().splitlines()[-1]
                )
                self.assertEqual(signature["failure_class"], "active_task_drift")
                self.assertEqual(signature["issue_scope"], "target_task")

                agent._schedule_spec_task()
                self.assertEqual(agent.state.current, AgentStateName.TASK_READ)
                self.assertEqual(agent.state.scratch["active_todo"]["spec_task_id"], "task-002")

        asyncio.run(run_case())

    def test_targeted_rewrite_unrelated_additional_task_is_not_promoted(
        self,
    ) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                agent = MicroAgent(
                    {
                        "models": {},
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_progress_path": ".local_micro_agent/spec_progress.jsonl",
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                previous_spec = agent._normalize_run_spec(
                    {
                        "version": 2,
                        "spec_id": "previous",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "status": "needs_design",
                                "target_regions": ["target.py::target"],
                            },
                            {
                                "task_id": "task-002",
                                "status": "open",
                                "target_regions": ["target.py::sibling"],
                            },
                        ],
                    }
                )
                unrelated_rewrite = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "unrelated",
                        "task_graph": [
                            {
                                "task_id": "task-new",
                                "status": "open",
                                "target_regions": ["target.py::unrelated"],
                            }
                        ],
                    }
                )
                agent.models = _StaticModelManager(unrelated_rewrite)
                agent.state.scratch["run_spec"] = previous_spec
                agent.state.scratch["spec_rewrite_focus"] = "rewrite task-001"
                agent.state.scratch["spec_rewrite_target_task_id"] = "task-001"

                await agent._maybe_refresh_run_spec(force=True)

                artifact_dir = repo / ".local_micro_agent"
                persisted = json.loads((artifact_dir / "run_spec.json").read_text())
                self.assertEqual(
                    [task["task_id"] for task in persisted["task_graph"]],
                    ["task-001", "task-002"],
                )
                self.assertEqual(persisted["task_graph"][0]["status"], "deferred_design_invalid")
                self.assertEqual(
                    persisted["last_targeted_rewrite_rejected"]["ignored_non_target_task_ids"],
                    ["task-new"],
                )
                progress = (artifact_dir / "spec_progress.jsonl").read_text()
                self.assertIn('"ignored_non_target_task_ids": ["task-new"]', progress)
                self.assertIn("omitted target replacement", progress)
                self.assertNotIn("task-new\", \"task-002", json.dumps(persisted))

        asyncio.run(run_case())

    def test_spec_design_failure_memory_context_summarizes_rejected_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "spec_progress.jsonl").write_text(
                json.dumps(
                    {
                        "event": "failed_design",
                        "spec_id": "portable-spec",
                        "task_id": "task-001",
                        "task_title": "Refactor pipeline",
                        "task_edit_scope": "Replace the pipeline implementation.",
                        "task_risk_level": "structural",
                        "task_tactic_stage": "structural_probe",
                        "task_target_regions": ["target.py::pipeline"],
                        "issues": ["edit_scope too broad"],
                        "rewrite_attempt": 3,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_progress_path": ".local_micro_agent/spec_progress.jsonl",
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )

            context = agent._spec_design_failure_memory_context()

            self.assertIn("negative design memory", context)
            self.assertIn("Do not regenerate the same shape", context)
            self.assertIn("Refactor pipeline", context)
            self.assertIn("edit_scope too broad", context)

    def test_spec_design_rewrite_focus_requires_narrower_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=Path(tmp), user_request="test"),
            )

            focus = agent._spec_design_rewrite_focus(
                {
                    "task_id": "task-001",
                    "title": "Refactor pipeline",
                    "edit_scope": "Replace the pipeline implementation.",
                },
                ["edit_scope too broad"],
            )

            self.assertIn("Do not regenerate the same rejected design shape", focus)
            self.assertIn("materially narrower", focus)

    def test_spec_design_rewrite_focus_marks_active_drift_unexecutable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=Path(tmp), user_request="test"),
            )

            focus = agent._spec_design_rewrite_focus(
                {
                    "task_id": "task-001",
                    "title": "Pack structural slots",
                    "target_regions": ["target.py::Builder.build"],
                    "tactic_stage": "structural_probe",
                    "validator": {"kind": "metric"},
                    "last_observation": {
                        "failure_class": "active_task_drift",
                        "failure_origin": "pre_apply_contract",
                        "candidate_status": "rejected_todo_scope_drift",
                        "candidate_id": "loop-001-single",
                        "drift_declared_regions": ["target.py::Builder.build"],
                        "drift_attempted_regions": ["target.py::Builder.build"],
                        "drift_cooldown_key": "abc:structural_probe:wide_probe",
                        "semantic_family_key": "family123",
                    },
                },
                ["repeated active_task_drift"],
            )

            self.assertIn("CODE could not execute the previous active task contract", focus)
            self.assertIn("Generate a smaller executable contract", focus)
            self.assertIn("Do not ask CODE to repair the same candidate shape", focus)
            self.assertIn("drift_declared_regions", focus)
            self.assertIn("family123", focus)

    def test_spec_design_failure_memory_includes_drift_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "spec_progress.jsonl").write_text(
                json.dumps(
                    {
                        "event": "drift_recovery",
                        "reason": "repeated_active_task_drift",
                        "action": "rewrite",
                        "task_id": "task-001",
                        "task_title": "Pack structural slots",
                        "task_tactic_stage": "structural_probe",
                        "task_target_regions": ["target.py::Builder.build"],
                        "drift_declared_regions": ["target.py::Builder.build"],
                        "drift_attempted_regions": ["target.py::Builder.build"],
                        "drift_cooldown_key": "abc:structural_probe:wide_probe",
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
                        "spec_mode": True,
                        "spec_progress_path": ".local_micro_agent/spec_progress.jsonl",
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )

            context = agent._spec_design_failure_memory_context()

            self.assertIn("active-task drift recoveries", context)
            self.assertIn("CODE could not execute an active task contract", context)
            self.assertIn("drift_recovery", context)
            self.assertIn("repeated_active_task_drift", context)
            self.assertIn("abc:structural_probe:wide_probe", context)

    def test_spec_terminal_state_records_lineage_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "spec_progress.jsonl").write_text(
                json.dumps({"event": "scheduled", "task_id": "task-001"}) + "\n"
                + json.dumps({"event": "failed", "reason": "max_code_test_loops"}) + "\n"
            )
            (artifact_dir / "candidates.jsonl").write_text(
                json.dumps(
                    {
                        "candidate_id": "loop-000-single",
                        "status": "rejected_todo_scope_drift",
                        "failure_class": "active_task_drift",
                        "budget_counted": False,
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "candidate_id": "loop-001-single",
                        "status": "improved",
                        "metric": 143638,
                        "spec_task_id": "task-002",
                        "todo_id": "task-002",
                        "last_correct_patch_path": ".local_micro_agent/last_correct.patch",
                        "changes": [
                            {
                                "path": "target.py",
                                "target_region": "target.py::parse_item",
                            }
                        ],
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
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_report_path": ".local_micro_agent/spec_report.md",
                    },
                },
                AgentState(
                    repo_root=repo,
                    user_request="test",
                    current=AgentStateName.FAILED,
                    loop_count=1,
                    max_loops=1,
                ),
            )
            spec = {
                "version": 2,
                "spec_id": "portable-spec",
                "last_stop_reason": "max_code_test_loops",
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "title": "guard parser branch",
                        "status": "needs_design",
                        "budget": {"attempts_max": 3, "attempts_used": 1},
                    },
                    {
                        "task_id": "task-002",
                        "title": "validated parser branch",
                        "status": "closed",
                        "deliverables": ["target.py"],
                        "target_regions": ["target.py::parse_item"],
                        "probe_diff_contract": {
                            "allowed_files": ["target.py"],
                            "expected_changed_regions": ["target.py::parse_item"],
                        },
                    }
                ],
            }
            agent.state.scratch["run_spec"] = spec

            agent._persist_spec_report()

            terminal = json.loads((artifact_dir / "terminal_state.json").read_text())
            self.assertEqual(terminal["stop_reason"], "max_code_test_loops")
            self.assertEqual(terminal["spec_progress_counts"]["scheduled"], 1)
            self.assertEqual(
                terminal["candidate_status_counts"]["rejected_todo_scope_drift"],
                1,
            )
            self.assertEqual(terminal["candidate_status_counts"]["improved"], 1)
            self.assertEqual(
                terminal["candidate_failure_class_counts"]["active_task_drift"],
                1,
            )
            self.assertEqual(terminal["tasks"][0]["status"], "needs_design")
            self.assertEqual(terminal["survivor"]["candidate_id"], "loop-001-single")
            self.assertEqual(terminal["survivor"]["metric"], 143638)
            self.assertEqual(
                terminal["survivor"]["patch_path"],
                ".local_micro_agent/last_correct.patch",
            )
            self.assertEqual(
                terminal["trajectory_quality"]["label"],
                "spec_aligned_success_with_drift",
            )
            self.assertEqual(
                terminal["trajectory_quality"]["spec_aligned_success_count"],
                1,
            )
            self.assertEqual(terminal["trajectory_quality"]["scope_drift_count"], 1)
            self.assertTrue(
                terminal["trajectory_quality"]["improved_candidate_matches_probe_plan"]
            )
            report = (artifact_dir / "spec_report.md").read_text()
            self.assertIn("## Survivor", report)
            self.assertIn("## Trajectory Quality", report)
            self.assertIn("loop-001-single", report)
            self.assertIn("spec_aligned_success_with_drift", report)

    def test_trajectory_quality_labels_clean_spec_aligned_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=Path(tmp), user_request="test"),
            )
            tasks = [
                {
                    "task_id": "task-001",
                    "deliverables": ["target.py"],
                    "target_regions": ["target.py::parse_item"],
                    "probe_diff_contract": {
                        "allowed_files": ["target.py"],
                        "expected_changed_regions": ["target.py::parse_item"],
                    },
                }
            ]
            records = [
                {
                    "status": "improved",
                    "candidate_id": "good",
                    "spec_task_id": "task-001",
                    "changes": [
                        {
                            "path": "target.py",
                            "target_region": "target.py::parse_item",
                        }
                    ],
                }
            ]

            quality = agent._terminal_trajectory_quality(tasks, records)

            self.assertEqual(quality["label"], "spec_aligned_success")
            self.assertEqual(quality["spec_aligned_success_count"], 1)
            self.assertTrue(quality["improved_candidate_matches_probe_plan"])

    def test_trajectory_quality_labels_no_success_and_chaotic_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=Path(tmp), user_request="test"),
            )

            no_success = agent._terminal_trajectory_quality([], [])
            chaotic = agent._terminal_trajectory_quality(
                [],
                [
                    {
                        "status": "rejected_active_task_region_drift",
                        "failure_class": "active_task_drift",
                        "budget_counted": False,
                    }
                ],
            )

            self.assertEqual(no_success["label"], "no_success")
            self.assertEqual(chaotic["label"], "chaotic_retry")
            self.assertEqual(chaotic["budget_free_contract_rejection_count"], 1)

    def test_trajectory_quality_labels_lucky_pass_risk_for_unmatched_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=Path(tmp), user_request="test"),
            )
            tasks = [
                {
                    "task_id": "task-001",
                    "deliverables": ["target.py"],
                    "target_regions": ["target.py::parse_item"],
                }
            ]
            records = [
                {
                    "status": "improved",
                    "candidate_id": "lucky",
                    "spec_task_id": "task-001",
                    "changes": [
                        {
                            "path": "other.py",
                            "target_region": "other.py::helper",
                        }
                    ],
                }
            ]

            quality = agent._terminal_trajectory_quality(tasks, records)

            self.assertEqual(quality["label"], "lucky_pass_risk")
            self.assertFalse(quality["improved_candidate_matches_probe_plan"])

    def test_spec_terminal_state_records_quality_gate_failure_without_run_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "spec_progress.jsonl").write_text(
                json.dumps(
                    {
                        "event": "quality_rejected",
                        "quality_issue_codes": ["vague_edit_scope"],
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
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_report_path": ".local_micro_agent/spec_report.md",
                    },
                },
                AgentState(
                    repo_root=repo,
                    user_request="test",
                    current=AgentStateName.FAILED,
                    loop_count=0,
                    max_loops=100,
                ),
            )
            quality_report = {
                "version": 1,
                "status": "fail",
                "attempt": 1,
                "spec_id": "bad-spec",
                "issues": [
                    {
                        "code": "vague_edit_scope",
                        "task_id": "task-001",
                        "detail": "Refactor target().",
                    }
                ],
                "issue_codes": ["vague_edit_scope"],
            }
            agent.state.scratch["spec_quality_report"] = quality_report
            agent.state.notes.append("Run spec discarded: quality gate issues remain")

            agent._persist_spec_report()

            terminal = json.loads((artifact_dir / "terminal_state.json").read_text())
            report = (artifact_dir / "spec_report.md").read_text()
            self.assertEqual(terminal["stop_reason"], "spec_quality_gate_failed")
            self.assertEqual(terminal["spec_id"], "bad-spec")
            self.assertEqual(terminal["progress"]["total"], 0)
            self.assertEqual(terminal["spec_progress_counts"]["quality_rejected"], 1)
            self.assertEqual(
                terminal["spec_quality_report"]["issue_codes"],
                ["vague_edit_scope"],
            )
            self.assertIn("stop_reason: `spec_quality_gate_failed`", report)
            self.assertIn("## Quality Gate", report)
            self.assertIn("`vague_edit_scope`", report)

    def test_spec_terminal_state_records_spec_call_budget_exhaustion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_report_path": ".local_micro_agent/spec_report.md",
                        "spec_synth_call_budget": 1,
                    },
                },
                AgentState(
                    repo_root=repo,
                    user_request="test",
                    current=AgentStateName.FAILED,
                    loop_count=0,
                    max_loops=100,
                ),
            )
            agent.state.scratch["spec_synth_call_count"] = 1
            agent.state.scratch["spec_synth_budget_exhausted"] = True
            (artifact_dir / "failure_signatures.jsonl").write_text(
                json.dumps(
                    {
                        "schema": "failure_signature.v1",
                        "failure_class": "design_rewrite_invalid",
                        "issue_code": "single_broad_structural_task",
                        "cooldown_key": "abcd1234:structural_probe:single_broad_structural_task",
                    },
                    sort_keys=True,
                )
                + "\n"
            )

            agent._persist_spec_report()

            terminal = json.loads((artifact_dir / "terminal_state.json").read_text())
            report = (artifact_dir / "spec_report.md").read_text()
            self.assertEqual(terminal["stop_reason"], "spec_budget_exhausted")
            self.assertTrue(terminal["zero_code_attempt"])
            self.assertEqual(terminal["spec_synth_call_count"], 1)
            self.assertEqual(terminal["spec_synth_calls_used"], 1)
            self.assertEqual(terminal["spec_synth_call_budget"], 1)
            self.assertTrue(terminal["spec_synth_budget_exhausted"])
            self.assertEqual(
                terminal["failure_signature_counts"],
                {"design_rewrite_invalid": 1},
            )
            self.assertEqual(
                terminal["failure_signature_issue_counts"],
                {"single_broad_structural_task": 1},
            )
            self.assertEqual(
                terminal["last_failure_signature"]["cooldown_key"],
                "abcd1234:structural_probe:single_broad_structural_task",
            )
            self.assertIn("stop_reason: `spec_budget_exhausted`", report)
            self.assertIn("spec_synth_call_count: 1", report)
            self.assertIn("spec_synth_calls_used: 1", report)
            self.assertIn("## Failure Signatures", report)

    def test_valid_spec_design_contract_becomes_active_todo_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["run_spec"] = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "perf",
                    "invariants": ["public behavior stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "guard parser branch",
                            "strategy_axis": "general_edit",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["existing parse outputs stay unchanged"],
                            "edit_scope": "Change one guarded branch in parse_item.",
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The fallback branch is unchanged.",
                            "fallback_plan": "Revert the guarded branch.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.TASK_READ)
            todo = agent.state.scratch["active_todo"]
            self.assertEqual(todo["target_symbols"], ["parse_item"])
            self.assertEqual(todo["target_regions"], ["target.py::parse_item"])
            self.assertIn("one guarded branch", todo["micro_goal"])
            self.assertEqual(todo["validator"]["failure_condition"], "pytest fails")

    def test_spec_grounding_gate_allows_writable_resolvable_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text(
                "def parse_item(value):\n    return value\n",
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_grounding_gate": True,
                        "writable_files": ["target.py"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["run_spec"] = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "grounded",
                    "invariants": ["public behavior stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "guard parser branch",
                            "strategy_axis": "general_edit",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["existing parse outputs stay unchanged"],
                            "edit_scope": "Change one guarded branch in parse_item.",
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The fallback branch is unchanged.",
                            "fallback_plan": "Revert the guarded branch.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.TASK_READ)
            facts = json.loads(
                (repo / ".local_micro_agent" / "spec_grounding_facts.json").read_text()
            )
            self.assertIn("target.py::parse_item", facts["allowed_target_regions"])

    def test_spec_grounding_gate_allows_read_only_symbol_as_context_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
            (repo / "problem.py").write_text("class Machine:\n    def step(self):\n        pass\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_grounding_gate": True,
                        "writable_files": ["target.py"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.file_context = [
                FileSnapshot(path="target.py", content=(repo / "target.py").read_text()),
                FileSnapshot(path="problem.py", content=(repo / "problem.py").read_text()),
            ]
            agent.state.scratch["run_spec"] = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "grounded-context",
                    "invariants": ["Machine.step behavior stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "guard parser branch",
                            "strategy_axis": "general_edit",
                            "deliverables": ["target.py"],
                            "read_hints": ["problem.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["Machine.step behavior stays unchanged"],
                            "edit_scope": "Change one guarded branch in parse_item.",
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "Machine.step is read-only context.",
                            "fallback_plan": "Revert the guarded branch.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.TASK_READ)

    def test_spec_grounding_gate_allows_read_only_required_unchanged_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
            (repo / "problem.py").write_text("class Machine:\n    def step(self):\n        pass\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_grounding_gate": True,
                        "writable_files": ["target.py"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.file_context = [
                FileSnapshot(path="target.py", content=(repo / "target.py").read_text()),
                FileSnapshot(path="problem.py", content=(repo / "problem.py").read_text()),
            ]
            agent.state.scratch["run_spec"] = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "grounded-unchanged",
                    "invariants": ["Machine.step behavior stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "guard parser branch",
                            "strategy_axis": "general_edit",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["Machine.step behavior stays unchanged"],
                            "edit_scope": "Change one guarded branch in parse_item.",
                            "probe_diff_contract": {
                                "allowed_regions": ["target.py::parse_item"],
                                "expected_changed_regions": ["target.py::parse_item"],
                                "required_unchanged_regions": ["problem.py::Machine.step"],
                            },
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "Machine.step is read-only context.",
                            "fallback_plan": "Revert the guarded branch.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.TASK_READ)

    def test_spec_grounding_gate_rejects_imported_symbol_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "problem.py").write_text("class Machine:\n    def step(self):\n        pass\n")
            (repo / "perf_takehome.py").write_text(
                "from problem import Machine\n\n"
                "class KernelBuilder:\n"
                "    def build(self):\n"
                "        return []\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_grounding_gate": True,
                        "writable_files": ["perf_takehome.py"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.file_context = [
                FileSnapshot(path="perf_takehome.py", content=(repo / "perf_takehome.py").read_text()),
                FileSnapshot(path="problem.py", content=(repo / "problem.py").read_text()),
            ]
            agent.state.scratch["run_spec"] = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "bad-target",
                    "invariants": ["public behavior stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "optimize machine step",
                            "strategy_axis": "general_edit",
                            "deliverables": ["perf_takehome.py"],
                            "target_symbols": ["Machine.step"],
                            "target_regions": ["perf_takehome.py::Machine.step"],
                            "preserved_invariants": ["Machine.step behavior stays unchanged"],
                            "edit_scope": "Change one guarded branch in Machine.step.",
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The fallback branch is unchanged.",
                            "fallback_plan": "Revert the guarded branch.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            agent._schedule_spec_task()

            persisted = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
            issues = persisted["task_graph"][0]["design_contract"]["issues"]
            self.assertIn("imported_symbol_targeted:Machine.step", issues)
            self.assertIn("unresolvable_target_region:perf_takehome.py::Machine.step", issues)

    def test_spec_grounding_gate_allows_writable_symbol_imported_by_read_only_tests(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "tests").mkdir()
            (repo / "perf_takehome.py").write_text(
                "class KernelBuilder:\n"
                "    def build(self):\n"
                "        return []\n"
            )
            (repo / "tests" / "submission_tests.py").write_text(
                "from perf_takehome import KernelBuilder\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_grounding_gate": True,
                        "writable_files": ["perf_takehome.py"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.file_context = [
                FileSnapshot(
                    path="perf_takehome.py",
                    content=(repo / "perf_takehome.py").read_text(),
                ),
                FileSnapshot(
                    path="tests/submission_tests.py",
                    content=(repo / "tests" / "submission_tests.py").read_text(),
                ),
            ]
            agent.state.scratch["spec_grounding_facts"] = {
                "writable_files": ["perf_takehome.py"],
                "allowed_target_regions": [
                    "perf_takehome.py::KernelBuilder",
                    "perf_takehome.py::KernelBuilder.build",
                ],
                "read_only_symbols": [],
                "imported_symbols": [
                    {
                        "path": "tests/submission_tests.py",
                        "symbol": "KernelBuilder",
                        "imported_name": "KernelBuilder",
                        "module": "perf_takehome",
                        "origin_path": "perf_takehome.py",
                        "origin_region": "perf_takehome.py::KernelBuilder",
                    }
                ],
            }
            task = {
                "task_id": "task-001",
                "title": "guard build branch",
                "strategy_axis": "general_edit",
                "deliverables": ["perf_takehome.py"],
                "target_symbols": ["KernelBuilder.build"],
                "target_regions": ["perf_takehome.py::KernelBuilder.build"],
                "preserved_invariants": ["build return behavior stays unchanged"],
                "edit_scope": "Change one guarded branch in KernelBuilder.build.",
                "validator": {
                    "kind": "command",
                    "failure_condition": "pytest fails",
                },
                "correctness_rationale": "The fallback branch is unchanged.",
                "fallback_plan": "Revert the guarded branch.",
                "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
            }

            issues = agent._spec_task_grounding_issues(task)

            self.assertNotIn("imported_symbol_targeted:KernelBuilder.build", issues)

    def test_spec_grounding_gate_allows_qualified_writable_symbol_imported_by_writable_tests(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "tests").mkdir()
            (repo / "perf_takehome.py").write_text(
                "class KernelBuilder:\n"
                "    def build_kernel(self):\n"
                "        return []\n"
            )
            (repo / "tests" / "submission_tests.py").write_text(
                "from perf_takehome import KernelBuilder\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_grounding_gate": True,
                        "writable_files": [
                            "perf_takehome.py",
                            "tests/submission_tests.py",
                        ],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["spec_grounding_facts"] = {
                "writable_files": ["perf_takehome.py", "tests/submission_tests.py"],
                "allowed_target_regions": [
                    "perf_takehome.py::KernelBuilder",
                    "perf_takehome.py::KernelBuilder.build_kernel",
                    "tests/submission_tests.py::kernel_builder",
                ],
                "read_only_symbols": [],
                "imported_symbols": [
                    {
                        "path": "tests/submission_tests.py",
                        "symbol": "KernelBuilder",
                        "imported_name": "KernelBuilder",
                        "module": "perf_takehome",
                        "origin_path": "perf_takehome.py",
                        "origin_region": "perf_takehome.py::KernelBuilder",
                    }
                ],
            }
            task = {
                "task_id": "task-001",
                "title": "guard build kernel branch",
                "strategy_axis": "general_edit",
                "deliverables": ["perf_takehome.py"],
                "target_symbols": ["perf_takehome.py::KernelBuilder.build_kernel"],
                "target_regions": ["perf_takehome.py::KernelBuilder.build_kernel"],
                "preserved_invariants": ["build_kernel fallback stays unchanged"],
                "edit_scope": "Change one guarded branch in KernelBuilder.build_kernel.",
                "validator": {
                    "kind": "command",
                    "failure_condition": "pytest fails",
                },
                "correctness_rationale": "The fallback branch is unchanged.",
                "fallback_plan": "Revert the guarded branch.",
                "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
            }

            issues = agent._spec_task_grounding_issues(task)

            self.assertNotIn(
                "imported_symbol_targeted:perf_takehome.py::KernelBuilder.build_kernel",
                issues,
            )

    def test_spec_grounding_gate_allows_region_grounded_unqualified_symbol_imported_by_writable_tests(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "tests").mkdir()
            (repo / "perf_takehome.py").write_text(
                "class KernelBuilder:\n"
                "    def build_kernel(self):\n"
                "        return []\n"
            )
            (repo / "tests" / "submission_tests.py").write_text(
                "from perf_takehome import KernelBuilder\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_grounding_gate": True,
                        "writable_files": [
                            "perf_takehome.py",
                            "tests/submission_tests.py",
                        ],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["spec_grounding_facts"] = {
                "writable_files": ["perf_takehome.py", "tests/submission_tests.py"],
                "allowed_target_regions": [
                    "perf_takehome.py::KernelBuilder",
                    "perf_takehome.py::KernelBuilder.build_kernel",
                    "tests/submission_tests.py::kernel_builder",
                ],
                "read_only_symbols": [],
                "imported_symbols": [
                    {
                        "path": "tests/submission_tests.py",
                        "symbol": "KernelBuilder",
                        "imported_name": "KernelBuilder",
                        "module": "perf_takehome",
                        "origin_path": "perf_takehome.py",
                        "origin_region": "perf_takehome.py::KernelBuilder",
                    }
                ],
            }
            task = {
                "task_id": "task-001",
                "title": "guard build kernel branch",
                "strategy_axis": "general_edit",
                "deliverables": ["perf_takehome.py"],
                "target_symbols": ["KernelBuilder.build_kernel"],
                "target_regions": ["perf_takehome.py::KernelBuilder.build_kernel"],
                "preserved_invariants": ["build_kernel fallback stays unchanged"],
                "edit_scope": "Change one guarded branch in KernelBuilder.build_kernel.",
                "validator": {
                    "kind": "command",
                    "failure_condition": "pytest fails",
                },
                "correctness_rationale": "The fallback branch is unchanged.",
                "fallback_plan": "Revert the guarded branch.",
                "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
            }

            issues = agent._spec_task_grounding_issues(task)

            self.assertNotIn(
                "imported_symbol_targeted:KernelBuilder.build_kernel",
                issues,
            )

    def test_spec_grounding_gate_rejects_unqualified_sibling_imported_by_writable_tests(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "tests").mkdir()
            (repo / "perf_takehome.py").write_text(
                "class KernelBuilder:\n"
                "    def build(self):\n"
                "        return []\n"
                "    def build_kernel(self):\n"
                "        return []\n"
            )
            (repo / "tests" / "submission_tests.py").write_text(
                "from perf_takehome import KernelBuilder\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_grounding_gate": True,
                        "writable_files": [
                            "perf_takehome.py",
                            "tests/submission_tests.py",
                        ],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["spec_grounding_facts"] = {
                "writable_files": ["perf_takehome.py", "tests/submission_tests.py"],
                "allowed_target_regions": [
                    "perf_takehome.py::KernelBuilder",
                    "perf_takehome.py::KernelBuilder.build",
                    "perf_takehome.py::KernelBuilder.build_kernel",
                    "tests/submission_tests.py::kernel_builder",
                ],
                "read_only_symbols": [],
                "imported_symbols": [
                    {
                        "path": "tests/submission_tests.py",
                        "symbol": "KernelBuilder",
                        "imported_name": "KernelBuilder",
                        "module": "perf_takehome",
                        "origin_path": "perf_takehome.py",
                        "origin_region": "perf_takehome.py::KernelBuilder",
                    }
                ],
            }
            task = {
                "task_id": "task-001",
                "title": "guard build kernel branch",
                "strategy_axis": "general_edit",
                "deliverables": ["perf_takehome.py"],
                "target_symbols": ["KernelBuilder.build"],
                "target_regions": ["perf_takehome.py::KernelBuilder.build_kernel"],
                "preserved_invariants": ["build_kernel fallback stays unchanged"],
                "edit_scope": "Change one guarded branch in KernelBuilder.build_kernel.",
                "validator": {
                    "kind": "command",
                    "failure_condition": "pytest fails",
                },
                "correctness_rationale": "The fallback branch is unchanged.",
                "fallback_plan": "Revert the guarded branch.",
                "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
            }

            issues = agent._spec_task_grounding_issues(task)

            self.assertIn("imported_symbol_targeted:KernelBuilder.build", issues)

    def test_spec_grounding_gate_allows_class_root_for_grounded_method_region(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "tests").mkdir()
            (repo / "perf_takehome.py").write_text(
                "class KernelBuilder:\n"
                "    def build_kernel(self):\n"
                "        return []\n"
            )
            (repo / "tests" / "submission_tests.py").write_text(
                "from perf_takehome import KernelBuilder\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_grounding_gate": True,
                        "writable_files": [
                            "perf_takehome.py",
                            "tests/submission_tests.py",
                        ],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["spec_grounding_facts"] = {
                "writable_files": ["perf_takehome.py", "tests/submission_tests.py"],
                "allowed_target_regions": [
                    "perf_takehome.py::KernelBuilder",
                    "perf_takehome.py::KernelBuilder.build_kernel",
                    "tests/submission_tests.py::kernel_builder",
                ],
                "read_only_symbols": [],
                "imported_symbols": [
                    {
                        "path": "tests/submission_tests.py",
                        "symbol": "KernelBuilder",
                        "imported_name": "KernelBuilder",
                        "module": "perf_takehome",
                        "origin_path": "perf_takehome.py",
                        "origin_region": "perf_takehome.py::KernelBuilder",
                    }
                ],
            }
            task = {
                "task_id": "task-001",
                "title": "guard build kernel branch",
                "strategy_axis": "general_edit",
                "deliverables": ["perf_takehome.py"],
                "target_symbols": ["KernelBuilder"],
                "target_regions": ["perf_takehome.py::KernelBuilder.build_kernel"],
                "preserved_invariants": ["build_kernel fallback stays unchanged"],
                "edit_scope": "Change one guarded branch in KernelBuilder.build_kernel.",
                "validator": {
                    "kind": "command",
                    "failure_condition": "pytest fails",
                },
                "correctness_rationale": "The fallback branch is unchanged.",
                "fallback_plan": "Revert the guarded branch.",
                "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
            }

            issues = agent._spec_task_grounding_issues(task)

            self.assertNotIn("imported_symbol_targeted:KernelBuilder", issues)

    def test_spec_grounding_gate_rejects_read_only_deliverable_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "problem.py").write_text("class Machine:\n    def step(self):\n        pass\n")
            (repo / "perf_takehome.py").write_text("def build_kernel():\n    return []\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_grounding_gate": True,
                        "writable_files": ["perf_takehome.py"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.file_context = [
                FileSnapshot(path="perf_takehome.py", content=(repo / "perf_takehome.py").read_text()),
                FileSnapshot(path="problem.py", content=(repo / "problem.py").read_text()),
            ]
            agent.state.scratch["run_spec"] = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "read-only-target",
                    "invariants": ["public behavior stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "optimize machine step",
                            "strategy_axis": "general_edit",
                            "deliverables": ["problem.py"],
                            "target_symbols": ["problem.py::Machine.step"],
                            "target_regions": ["problem.py::Machine.step"],
                            "preserved_invariants": ["Machine.step behavior stays unchanged"],
                            "edit_scope": "Change one guarded branch in Machine.step.",
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The fallback branch is unchanged.",
                            "fallback_plan": "Revert the guarded branch.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            agent._schedule_spec_task()

            persisted = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
            issues = persisted["task_graph"][0]["design_contract"]["issues"]
            self.assertIn("read_only_deliverable:problem.py", issues)
            self.assertIn("non_writable_target_region:problem.py::Machine.step", issues)

    def test_spec_grounding_gate_rejects_probe_contract_region_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text(
                "def parse_item(value):\n    return value\n\n"
                "def format_item(value):\n    return str(value)\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_grounding_gate": True,
                        "writable_files": ["target.py"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["run_spec"] = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "mismatch",
                    "invariants": ["public behavior stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "guard parser branch",
                            "strategy_axis": "general_edit",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["existing parse outputs stay unchanged"],
                            "edit_scope": "Change one guarded branch in parse_item.",
                            "probe_diff_contract": {
                                "allowed_regions": ["target.py::format_item"],
                                "expected_changed_regions": ["target.py::format_item"],
                            },
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The fallback branch is unchanged.",
                            "fallback_plan": "Revert the guarded branch.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            agent._schedule_spec_task()

            persisted = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
            issues = persisted["task_graph"][0]["design_contract"]["issues"]
            self.assertIn("probe_contract_region_mismatch:target.py::format_item", issues)

    def test_structural_risk_gate_rejects_broad_rewrite_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_structural_risk_gate": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["run_spec"] = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "perf",
                    "invariants": ["public behavior stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "Replace parser loop with batched state flow",
                            "strategy_axis": "general_edit",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["existing parse outputs stay unchanged"],
                            "edit_scope": "Replace the parser loop with a batched state machine.",
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "Outputs should stay equivalent.",
                            "fallback_plan": "Revert the parser rewrite.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.SPEC_SYNTH)
            persisted = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
            task = persisted["task_graph"][0]
            self.assertEqual(task["status"], "needs_design")
            issues = task["design_contract"]["issues"]
            self.assertIn("structural task must declare risk_level=structural", issues)
            self.assertIn("missing risk_evidence", issues)
            self.assertIn("missing probe_plan for structural task", issues)
            self.assertIn(
                "structural edit_scope too broad; start with one reversible probe",
                issues,
            )

    def test_structural_risk_gate_allows_local_micro_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_structural_risk_gate": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["run_spec"] = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "perf",
                    "invariants": ["public behavior stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "guard parser branch",
                            "strategy_axis": "general_edit",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["existing parse outputs stay unchanged"],
                            "edit_scope": "Change one guarded branch in parse_item.",
                            "risk_level": "local",
                            "tactic_stage": "local_edit",
                            "risk_evidence": {
                                "field": "edit_scope",
                                "quote": "Change one guarded branch",
                                "explanation": "This is a single local branch edit.",
                            },
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The fallback branch is unchanged.",
                            "fallback_plan": "Revert the guarded branch.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.TASK_READ)
            todo = agent.state.scratch["active_todo"]
            self.assertEqual(todo["risk_level"], "local")
            self.assertEqual(todo["tactic_stage"], "local_edit")

    def test_structural_risk_gate_allows_local_fix_with_negated_rationale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_structural_risk_gate": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["run_spec"] = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "perf",
                    "invariants": ["public behavior stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "Fix indentation in parser helper",
                            "strategy_axis": "syntax_fix",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["existing parse outputs stay unchanged"],
                            "edit_scope": "Correct indentation of one method body.",
                            "risk_level": "local",
                            "tactic_stage": "local_edit",
                            "risk_evidence": {
                                "field": "edit_scope",
                                "quote": "Correct indentation",
                                "explanation": "Whitespace-only parseability fix.",
                            },
                            "validator": {
                                "kind": "command",
                                "failure_condition": "SyntaxError or pytest fails",
                            },
                            "correctness_rationale": (
                                "This does not change any logic, variables, or control flow."
                            ),
                            "fallback_plan": "Revert the indentation-only change.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.TASK_READ)
            self.assertEqual(agent.state.scratch["active_todo"]["risk_level"], "local")

    def test_structural_risk_gate_rejects_local_label_with_structural_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_structural_risk_gate": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["run_spec"] = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "perf",
                    "invariants": ["public behavior stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "guard parser branch",
                            "strategy_axis": "general_edit",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["existing parse outputs stay unchanged"],
                            "edit_scope": "Change control flow in the parser loop.",
                            "risk_level": "local",
                            "tactic_stage": "local_edit",
                            "risk_evidence": {
                                "field": "edit_scope",
                                "quote": "Change control flow",
                                "explanation": "Incorrectly labeled as local.",
                            },
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The fallback branch is unchanged.",
                            "fallback_plan": "Revert the branch.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.SPEC_SYNTH)
            persisted = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
            issues = persisted["task_graph"][0]["design_contract"]["issues"]
            self.assertIn(
                "local risk_level contradicts structural action in task scope",
                issues,
            )

    def test_structural_risk_gate_allows_small_probe_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_structural_risk_gate": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["run_spec"] = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "perf",
                    "invariants": ["public behavior stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "Probe parser branch scheduling",
                            "strategy_axis": "general_edit",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["existing parse outputs stay unchanged"],
                            "edit_scope": (
                                "Add one guarded branch inside parse_item before the existing "
                                "parser loop."
                            ),
                            "risk_level": "structural",
                            "tactic_stage": "structural_probe",
                            "risk_evidence": {
                                "field": "title",
                                "quote": "scheduling",
                                "explanation": "Scheduling is a structural behavior risk.",
                            },
                            "probe_plan": "Add a single guarded branch and keep the old path.",
                            "invariant_evidence": [
                                "The existing branch remains the fallback path.",
                                "pytest covers equivalent parse outputs.",
                            ],
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The fallback parser path is unchanged.",
                            "fallback_plan": "Disable the guarded branch.",
                            "rollback_or_shrink_plan": (
                                "Shrink to a smaller guarded probe or revert the branch."
                            ),
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.TASK_READ)
            todo = agent.state.scratch["active_todo"]
            self.assertEqual(todo["risk_level"], "structural")
            self.assertEqual(todo["tactic_stage"], "structural_probe")
            self.assertIn("single guarded branch", todo["probe_plan"])
            self.assertIn("fallback path", todo["invariant_evidence"][0])

    def test_structural_risk_gate_rejects_missing_risk_evidence_quote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_structural_risk_gate": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["run_spec"] = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "perf",
                    "invariants": ["public behavior stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "Rewrite parser loop",
                            "strategy_axis": "general_edit",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["existing parse outputs stay unchanged"],
                            "edit_scope": "Add one guarded branch before the existing parser loop.",
                            "risk_level": "structural",
                            "tactic_stage": "structural_probe",
                            "risk_evidence": {
                                "field": "title",
                                "explanation": "The title describes a rewrite.",
                            },
                            "probe_plan": "Add a single guarded branch and keep the old path.",
                            "invariant_evidence": ["The old path remains the fallback path."],
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The fallback parser path is unchanged.",
                            "fallback_plan": "Disable the guarded branch.",
                            "rollback_or_shrink_plan": "Shrink to a smaller guarded probe.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.SPEC_SYNTH)
            persisted = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
            issues = persisted["task_graph"][0]["design_contract"]["issues"]
            self.assertIn("missing risk_evidence.quote", issues)

    def test_structural_risk_gate_rejects_risk_evidence_quote_outside_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_design_contract_gate": True,
                        "spec_structural_risk_gate": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["run_spec"] = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "perf",
                    "invariants": ["public behavior stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "Probe parser branch scheduling",
                            "strategy_axis": "general_edit",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["existing parse outputs stay unchanged"],
                            "edit_scope": "Add one guarded branch before the existing parser loop.",
                            "risk_level": "structural",
                            "tactic_stage": "structural_probe",
                            "risk_evidence": {
                                "field": "edit_scope",
                                "quote": "parallelize the scheduler",
                                "explanation": "Quote is not from edit_scope.",
                            },
                            "probe_plan": "Add a single guarded branch and keep the old path.",
                            "invariant_evidence": ["The old path remains the fallback path."],
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The fallback parser path is unchanged.",
                            "fallback_plan": "Disable the guarded branch.",
                            "rollback_or_shrink_plan": "Shrink to a smaller guarded probe.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            agent._schedule_spec_task()

            self.assertEqual(agent.state.current, AgentStateName.SPEC_SYNTH)
            persisted = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
            issues = persisted["task_graph"][0]["design_contract"]["issues"]
            self.assertIn("risk_evidence.quote must appear in the named field", issues)

    def test_spec_active_todo_contract_stays_hard_before_first_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_design_contract_gate": True,
                        "todo_soft_until_first_improvement": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "spec_task_id": "task-001",
                "status": "active",
                "strategy_axis": "general_edit",
                "title": "guard parser branch",
                "target_symbols": ["parse_item"],
                "target_regions": ["target.py::parse_item"],
                "source": "spec_scheduler",
            }

            self.assertFalse(agent._todo_contract_soft_now())
            self.assertEqual(agent._active_todo_id(), "task-001")
            self.assertIn("guard parser branch", agent._format_active_todo())

    def test_active_todo_change_scope_rejects_wrong_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text(
                "class Parser:\n"
                "    def parse_item(self, value):\n"
                "        return value.strip()\n\n"
                "    def build(self, value):\n"
                "        return value\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_design_contract_gate": True,
                        "todo_soft_until_first_improvement": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "spec_task_id": "task-001",
                "status": "active",
                "strategy_axis": "general_edit",
                "target_symbols": ["Parser.parse_item"],
                "target_regions": ["target.py::Parser.parse_item"],
                "allowed_files": ["target.py"],
                "source": "spec_scheduler",
            }
            change = CodeChange(
                "target.py",
                "speed up unrelated build helper",
                target="    def build(self, value):\n        return value\n",
                replacement="    def build(self, value):\n        return value + 1\n",
            )

            rejection = agent._active_todo_change_scope_rejection([change])

            self.assertIsNotNone(rejection)
            self.assertEqual(rejection[0], "rejected_todo_scope_drift")
            self.assertIn("Parser.parse_item", rejection[1])

    def test_active_todo_scope_drift_is_non_budget_candidate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text(
                "class Parser:\n"
                "    def parse_item(self, value):\n"
                "        return value.strip()\n\n"
                "    def build(self, value):\n"
                "        return value\n"
            )
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            todo = {
                "todo_id": "task-001",
                "spec_task_id": "task-001",
                "status": "active",
                "strategy_axis": "general_edit",
                "target_symbols": ["Parser.parse_item"],
                "target_regions": ["target.py::Parser.parse_item"],
                "allowed_files": ["target.py"],
                "source": "spec_scheduler",
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
                {
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
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            candidate = CodeCandidate(
                "scope-drift",
                [
                    CodeChange(
                        "target.py",
                        "local parser edit in Parser.parse_item",
                        target="    def build(self, value):\n        return value\n",
                        replacement="    def build(self, value):\n        return value + 1\n",
                    )
                ],
                "local parser edit in Parser.parse_item",
                strategy_axis="general_edit",
            )

            async def evaluate_once() -> None:
                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([candidate], {"target.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(evaluate_once())

            self.assertIn("return value\n", target.read_text())
            history = [
                json.loads(line)
                for line in (artifact_dir / "candidates.jsonl").read_text().splitlines()
            ]
            attempts = [
                json.loads(line)
                for line in (artifact_dir / "todo_attempts.jsonl").read_text().splitlines()
            ]
            updated = json.loads((artifact_dir / "todo_plan.json").read_text())
            self.assertEqual(history[0]["status"], "rejected_todo_scope_drift")
            self.assertFalse(history[0]["budget_counted"])
            self.assertEqual(history[0]["failure_class"], "active_task_drift")
            self.assertEqual(history[0]["issue_scope"], "candidate_delta")
            self.assertFalse(attempts[0]["budget_counted"])
            self.assertEqual(updated["todos"][0]["attempts"], 1)
            self.assertEqual(updated["todos"][0]["non_budget_attempts"], 1)

    def test_active_todo_drift_updates_last_candidate_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            todo = {
                "todo_id": "task-001",
                "spec_task_id": "task-001",
                "status": "active",
                "strategy_axis": "general_edit",
                "target_symbols": ["parse_item"],
                "target_regions": ["target.py::parse_item"],
                "source": "spec_scheduler",
            }
            (artifact_dir / "active_todo.json").write_text(json.dumps(todo) + "\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": takehome_workflow(
                        candidate_history_path=".local_micro_agent/candidates.jsonl",
                        todo_soft_until_first_improvement=False,
                    ),
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = todo
            candidate = CodeCandidate(
                "drift",
                [
                    CodeChange(
                        "target.py",
                        "edit helper",
                        target="old",
                        replacement="new",
                        target_region="target.py::helper",
                    )
                ],
                "edit helper",
                strategy_axis="general_edit",
            )
            extra = agent._candidate_rejection_extra(
                candidate,
                "rejected_todo_scope_drift",
                "outside active task",
            )

            agent._append_candidate_history(
                candidate,
                status="rejected_todo_scope_drift",
                metric=None,
                applied=0,
                failed=True,
                extra=extra,
            )

            observation = agent.state.scratch["last_candidate_observation"]
            self.assertEqual(observation["failure_class"], "active_task_drift")
            self.assertEqual(observation["failure_origin"], "pre_apply_contract")
            self.assertFalse(observation["budget_counted"])
            self.assertEqual(observation["drift_declared_regions"], ["target.py::parse_item"])
            self.assertEqual(observation["drift_attempted_regions"], ["target.py::helper"])
            self.assertEqual(
                observation["drift_region_pairs"],
                [
                    {
                        "declared": "target.py::parse_item",
                        "attempted": "target.py::helper",
                    }
                ],
            )
            attempts = [
                json.loads(line)
                for line in (artifact_dir / "todo_attempts.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                attempts[0]["drift_declared_regions"],
                ["target.py::parse_item"],
            )
            self.assertEqual(attempts[0]["drift_attempted_regions"], ["target.py::helper"])
            self.assertEqual(attempts[0]["drift_cooldown_key"], observation["drift_cooldown_key"])
            latest = agent._latest_active_task_drift_attempt("task-001")
            self.assertIsNotNone(latest)
            self.assertEqual(
                latest["drift_attempted_regions"],
                ["target.py::helper"],
            )

    def test_active_task_probe_contract_allows_matching_region_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text(
                "class Parser:\n"
                "    def parse_item(self, value):\n"
                "        return value.strip()\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_design_contract_gate": True,
                        "todo_enforce_active_change_scope": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "spec_task_id": "task-001",
                "status": "active",
                "strategy_axis": "general_edit",
                "target_symbols": ["Parser.parse_item"],
                "target_regions": ["target.py::Parser.parse_item"],
                "source": "spec_scheduler",
                "probe_diff_contract": {
                    "allowed_files": ["target.py"],
                    "allowed_regions": ["target.py::Parser.parse_item"],
                    "expected_changed_regions": ["target.py::Parser.parse_item"],
                    "max_files_changed": 1,
                    "max_hunks": 1,
                    "max_changed_lines": 3,
                },
            }
            change = CodeChange(
                "target.py",
                "narrow parse_item guard",
                target="        return value.strip()\n",
                replacement="        return value.strip() if value else ''\n",
                target_region="target.py::Parser.parse_item",
            )

            self.assertIsNone(agent._active_todo_change_scope_rejection([change]))

    def test_active_task_probe_contract_rejects_file_and_region_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
            (repo / "other.py").write_text("def helper(value):\n    return value\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_design_contract_gate": True,
                        "todo_enforce_active_change_scope": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "spec_task_id": "task-001",
                "status": "active",
                "strategy_axis": "general_edit",
                "target_symbols": ["parse_item"],
                "target_regions": ["target.py::parse_item"],
                "source": "spec_scheduler",
                "probe_diff_contract": {
                    "allowed_files": ["target.py"],
                    "allowed_regions": ["target.py::parse_item"],
                    "expected_changed_regions": ["target.py::parse_item"],
                    "max_files_changed": 1,
                    "max_hunks": 1,
                    "max_changed_lines": 3,
                },
            }
            file_drift = CodeChange(
                "other.py",
                "edit helper instead",
                target="    return value\n",
                replacement="    return value + 1\n",
                target_region="other.py::helper",
            )
            region_drift = CodeChange(
                "target.py",
                "edit helper instead",
                target="def parse_item(value):\n    return value\n",
                replacement="def parse_item(value):\n    return value + 1\n",
                target_region="target.py::helper",
            )

            file_rejection = agent._active_todo_change_scope_rejection([file_drift])
            region_rejection = agent._active_todo_change_scope_rejection([region_drift])

            self.assertIsNotNone(file_rejection)
            self.assertEqual(file_rejection[0], "rejected_active_task_file_drift")
            self.assertIsNotNone(region_rejection)
            self.assertEqual(region_rejection[0], "rejected_active_task_region_drift")

    def test_active_task_probe_contract_rejects_shape_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text(
                "def parse_item(value):\n"
                "    left = value.strip()\n"
                "    right = value.lower()\n"
                "    return left or right\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_design_contract_gate": True,
                        "todo_enforce_active_change_scope": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "spec_task_id": "task-001",
                "status": "active",
                "strategy_axis": "general_edit",
                "target_symbols": ["parse_item"],
                "target_regions": ["target.py::parse_item"],
                "source": "spec_scheduler",
                "probe_diff_contract": {
                    "allowed_files": ["target.py"],
                    "allowed_regions": ["target.py::parse_item"],
                    "expected_changed_regions": ["target.py::parse_item"],
                    "max_files_changed": 1,
                    "max_hunks": 1,
                    "max_changed_lines": 2,
                },
            }
            first = CodeChange(
                "target.py",
                "first hunk",
                target="    left = value.strip()\n",
                replacement="    left = value.strip() if value else ''\n",
                target_region="target.py::parse_item",
            )
            second = CodeChange(
                "target.py",
                "second hunk",
                target="    right = value.lower()\n",
                replacement="    right = value.lower() if value else ''\n",
                target_region="target.py::parse_item",
            )
            broad = CodeChange(
                "target.py",
                "too many declared lines",
                target=(
                    "    left = value.strip()\n"
                    "    right = value.lower()\n"
                    "    return left or right\n"
                ),
                replacement=(
                    "    left = value.strip() if value else ''\n"
                    "    right = value.lower() if value else ''\n"
                    "    return left or right\n"
                ),
                target_region="target.py::parse_item",
            )

            hunk_rejection = agent._active_todo_change_scope_rejection([first, second])
            line_rejection = agent._active_todo_change_scope_rejection([broad])

            self.assertIsNotNone(hunk_rejection)
            self.assertEqual(hunk_rejection[0], "rejected_active_task_shape_drift")
            self.assertIn("max_hunks", hunk_rejection[1])
            self.assertIsNotNone(line_rejection)
            self.assertEqual(line_rejection[0], "rejected_active_task_shape_drift")
            self.assertIn("max_changed_lines", line_rejection[1])

    def test_active_todo_contract_does_not_duplicate_scope_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text(
                "class Parser:\n"
                "    def parse_item(self, value):\n"
                "        return value.strip()\n\n"
                "    def build(self, value):\n"
                "        return value\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_design_contract_gate": True,
                        "todo_enforce_active_contract": True,
                        "todo_enforce_active_change_scope": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "spec_task_id": "task-001",
                "status": "active",
                "strategy_axis": "general_edit",
                "target_symbols": ["Parser.parse_item"],
                "target_regions": ["target.py::Parser.parse_item"],
                "allowed_files": ["target.py"],
                "source": "spec_scheduler",
            }
            change = CodeChange(
                "target.py",
                "speed up unrelated build helper",
                target="    def build(self, value):\n        return value\n",
                replacement="    def build(self, value):\n        return value + 1\n",
            )
            candidate = CodeCandidate(
                "queue-bad-scope",
                [change],
                "Try a local parser edit in Parser.parse_item",
                strategy_axis="general_edit",
            )

            self.assertIsNone(agent._active_todo_contract_rejection(candidate))
            scope_rejection = agent._active_todo_change_scope_rejection(candidate.changes)

            self.assertIsNotNone(scope_rejection)
            self.assertEqual(scope_rejection[0], "rejected_todo_scope_drift")

    def test_active_todo_change_scope_allows_target_symbol_span(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text(
                "class Parser:\n"
                "    def parse_item(self, value):\n"
                "        return value.strip()\n\n"
                "    def build(self, value):\n"
                "        return value\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_design_contract_gate": True,
                        "todo_soft_until_first_improvement": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "spec_task_id": "task-001",
                "status": "active",
                "strategy_axis": "general_edit",
                "target_symbols": ["Parser.parse_item"],
                "target_regions": ["target.py::Parser.parse_item"],
                "allowed_files": ["target.py"],
                "source": "spec_scheduler",
            }
            change = CodeChange(
                "target.py",
                "guard active parser todo task-001",
                target="    def parse_item(self, value):\n        return value.strip()\n",
                replacement=(
                    "    def parse_item(self, value):\n"
                    "        return value.strip() if value is not None else ''\n"
                ),
            )

            self.assertIsNone(agent._active_todo_change_scope_rejection([change]))

    def test_local_task_one_change_allows_single_target_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_design_contract_gate": True,
                        "spec_local_task_one_change": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "spec_task_id": "task-001",
                "status": "active",
                "tactic_stage": "local_edit",
                "target_symbols": ["parse_item"],
                "target_regions": ["target.py::parse_item"],
                "allowed_files": ["target.py"],
                "source": "spec_scheduler",
            }
            change = CodeChange(
                "target.py",
                "guard active parser target",
                target="def parse_item(value):\n    return value\n",
                replacement="def parse_item(value):\n    return value\n",
            )

            self.assertIsNone(agent._active_todo_change_scope_rejection([change]))

    def test_local_task_one_change_can_be_disabled_for_legacy_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_design_contract_gate": True,
                        "spec_local_task_one_change": False,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "spec_task_id": "task-001",
                "status": "active",
                "tactic_stage": "local_edit",
                "target_symbols": ["parse_item"],
                "target_regions": ["target.py::parse_item"],
                "allowed_files": ["target.py"],
                "source": "spec_scheduler",
            }
            changes = [
                CodeChange(
                    "target.py",
                    "first parser change",
                    target="def parse_item(value):\n    return value\n",
                    replacement="def parse_item(value):\n    return value\n",
                ),
                CodeChange(
                    "target.py",
                    "second parser change",
                    target="def parse_item(value):\n    return value\n",
                    replacement="def parse_item(value):\n    return value\n",
                ),
            ]

            self.assertIsNone(agent._active_todo_change_scope_rejection(changes))

    def test_local_task_one_change_rejects_multi_change_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_design_contract_gate": True,
                        "spec_local_task_one_change": True,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "spec_task_id": "task-001",
                "status": "active",
                "tactic_stage": "local_edit",
                "target_symbols": ["parse_item"],
                "target_regions": ["target.py::parse_item"],
                "allowed_files": ["target.py"],
                "source": "spec_scheduler",
            }
            changes = [
                CodeChange(
                    "target.py",
                    "first parser change",
                    target="def parse_item(value):\n    return value\n",
                    replacement="def parse_item(value):\n    return value\n",
                ),
                CodeChange(
                    "target.py",
                    "second parser change",
                    target="def parse_item(value):\n    return value\n",
                    replacement="def parse_item(value):\n    return value\n",
                ),
            ]

            rejection = agent._active_todo_change_scope_rejection(changes)

            self.assertIsNotNone(rejection)
            self.assertEqual(rejection[0], "rejected_todo_scope_drift")
            self.assertIn("one change", rejection[1])

    def test_structural_probe_change_scope_rejects_broad_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text(
                "def parse_item(value):\n"
                "    for part in value:\n"
                "        yield part\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_design_contract_gate": True,
                        "todo_enforce_active_change_scope": True,
                        "structural_probe_max_changes": 1,
                        "structural_probe_max_changed_lines": 40,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "spec_task_id": "task-001",
                "status": "active",
                "strategy_axis": "general_edit",
                "risk_level": "structural",
                "tactic_stage": "structural_probe",
                "target_symbols": ["parse_item"],
                "target_regions": ["target.py::parse_item"],
                "allowed_files": ["target.py"],
                "source": "spec_scheduler",
            }
            change = CodeChange(
                "target.py",
                "rewrite the whole parser function",
                target=(
                    "def parse_item(value):\n"
                    "    for part in value:\n"
                    "        yield part\n"
                ),
                replacement=(
                    "def parse_item(value):\n"
                    "    items = list(value)\n"
                    "    for index in range(len(items)):\n"
                    "        yield items[index]\n"
                ),
            )

            rejection = agent._active_todo_change_scope_rejection([change])

            self.assertIsNotNone(rejection)
            self.assertEqual(rejection[0], "rejected_todo_scope_drift")
            self.assertIn("structural_probe change is too broad", rejection[1])

    def test_probe_diff_contract_allows_expected_symbol_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            before = (
                "def parse_item(value):\n"
                "    return value.strip()\n\n"
                "def other(value):\n"
                "    return value\n"
            )
            target.write_text(before)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_hard_active_todo_contract": True,
                        "probe_diff_contract_gate": True,
                        "todo_soft_until_first_improvement": False,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "status": "active",
                "risk_level": "structural",
                "tactic_stage": "structural_probe",
                "target_symbols": ["parse_item"],
                "target_regions": ["target.py::parse_item"],
                "allowed_files": ["target.py"],
                "probe_diff_contract": {
                    "allowed_files": ["target.py"],
                    "allowed_regions": ["target.py::parse_item"],
                    "expected_changed_regions": ["target.py::parse_item"],
                    "max_files_changed": 1,
                    "max_hunks": 1,
                    "max_changed_lines": 2,
                    "max_changed_functions": 1,
                },
            }
            snapshot = {"target.py": before}
            target.write_text(
                before.replace(
                    "return value.strip()",
                    "return value.strip() if value else ''",
                )
            )

            rejection = agent._active_probe_diff_contract_rejection(
                snapshot,
                {"target.py"},
            )

            self.assertIsNone(rejection)

    def test_probe_diff_contract_rejects_symbol_outside_allowed_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            before = (
                "def parse_item(value):\n"
                "    return value.strip()\n\n"
                "def other(value):\n"
                "    return value\n"
            )
            target.write_text(before)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_hard_active_todo_contract": True,
                        "probe_diff_contract_gate": True,
                        "todo_soft_until_first_improvement": False,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "status": "active",
                "risk_level": "structural",
                "tactic_stage": "structural_probe",
                "target_symbols": ["parse_item"],
                "target_regions": ["target.py::parse_item"],
                "allowed_files": ["target.py"],
                "probe_diff_contract": {
                    "allowed_files": ["target.py"],
                    "allowed_regions": ["target.py::parse_item"],
                    "expected_changed_regions": ["target.py::parse_item"],
                    "max_files_changed": 1,
                    "max_hunks": 2,
                    "max_changed_lines": 5,
                    "max_changed_functions": 1,
                },
            }
            snapshot = {"target.py": before}
            target.write_text(before.replace("return value\n", "return value + 1\n"))

            rejection = agent._active_probe_diff_contract_rejection(
                snapshot,
                {"target.py"},
            )

            self.assertIsNotNone(rejection)
            self.assertEqual(rejection[0], "rejected_probe_contract_mismatch")
            self.assertIn("outside allowed_regions", rejection[1])
            self.assertEqual(rejection[2]["issue_scope"], "candidate_delta")
            self.assertFalse(rejection[2]["repair_task_eligible"])

    def test_probe_diff_contract_rejects_forbidden_region_and_syntax_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            before = (
                "def parse_item(value):\n"
                "    return value.strip()\n\n"
                "def other(value):\n"
                "    return value\n"
            )
            target.write_text(before)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_hard_active_todo_contract": True,
                        "probe_diff_contract_gate": True,
                        "todo_soft_until_first_improvement": False,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "status": "active",
                "risk_level": "structural",
                "tactic_stage": "structural_probe",
                "target_symbols": ["parse_item"],
                "target_regions": ["target.py::parse_item"],
                "allowed_files": ["target.py"],
                "probe_diff_contract": {
                    "allowed_files": ["target.py"],
                    "allowed_regions": ["target.py::parse_item"],
                    "expected_changed_regions": ["target.py::parse_item"],
                    "forbidden_symbols": ["other"],
                    "max_files_changed": 1,
                    "max_hunks": 2,
                    "max_changed_lines": 5,
                    "max_changed_functions": 1,
                },
            }
            snapshot = {"target.py": before}
            target.write_text(before.replace("return value.strip()", "return ("))

            rejection = agent._active_probe_diff_contract_rejection(
                snapshot,
                {"target.py"},
            )

            self.assertIsNotNone(rejection)
            self.assertIn("region mapping failed", rejection[1])

    def test_probe_diff_contract_fallback_rejects_outside_target_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            before = (
                "def parse_item(value):\n"
                "    return value.strip()\n\n"
                "def other(value):\n"
                "    return value\n"
            )
            target.write_text(before)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_hard_active_todo_contract": True,
                        "probe_diff_contract_gate": True,
                        "todo_soft_until_first_improvement": False,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "status": "active",
                "risk_level": "structural",
                "tactic_stage": "structural_probe",
                "target_symbols": ["parse_item"],
                "target_regions": ["target.py::parse_item"],
                "allowed_files": ["target.py"],
            }
            snapshot = {"target.py": before}
            target.write_text(before.replace("return value\n", "return value + 1\n"))

            rejection = agent._active_probe_diff_contract_rejection(
                snapshot,
                {"target.py"},
            )

            self.assertIsNotNone(rejection)
            self.assertIn("outside allowed_regions", rejection[1])
            self.assertIn("probe_diff_summary", rejection[2])

    def test_probe_diff_contract_rejects_hunk_line_and_unchanged_region_violations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            before = (
                "def parse_item(value):\n"
                "    left = value.strip()\n"
                "    return left\n\n"
                "def other(value):\n"
                "    cached = value\n"
                "    return cached\n"
            )
            target.write_text(before)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_hard_active_todo_contract": True,
                        "probe_diff_contract_gate": True,
                        "todo_soft_until_first_improvement": False,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "status": "active",
                "risk_level": "structural",
                "tactic_stage": "structural_probe",
                "target_symbols": ["parse_item"],
                "target_regions": ["target.py::parse_item"],
                "allowed_files": ["target.py"],
                "probe_diff_contract": {
                    "allowed_files": ["target.py"],
                    "allowed_regions": ["target.py::parse_item"],
                    "expected_changed_regions": ["target.py::parse_item"],
                    "required_unchanged_regions": ["target.py::other"],
                    "max_files_changed": 1,
                    "max_hunks": 1,
                    "max_changed_lines": 1,
                    "max_changed_functions": 1,
                },
            }
            snapshot = {"target.py": before}
            target.write_text(
                before.replace("left = value.strip()", "left = value.strip() or ''")
                .replace("return left", "return left.upper()")
                .replace("return cached", "return cached or ''")
            )

            rejection = agent._active_probe_diff_contract_rejection(
                snapshot,
                {"target.py"},
            )

            self.assertIsNotNone(rejection)
            note = rejection[1]
            self.assertIn("max_hunks", note)
            self.assertIn("max_changed_lines", note)
            self.assertIn("required-unchanged", note)

    def test_probe_diff_contract_ignores_missing_active_todo_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            before = "def parse_item(value):\n    return value.strip()\n"
            target.write_text("def parse_item(value):\n    return value\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "probe_diff_contract_gate": True,
                        "todo_soft_until_first_improvement": False,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )

            rejection = agent._active_probe_diff_contract_rejection(
                {"target.py": before},
                {"target.py"},
            )

            self.assertIsNone(rejection)

    def test_local_edit_is_not_probe_diff_gated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            before = "def parse_item(value):\n    return value.strip()\n"
            target.write_text(before)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_hard_active_todo_contract": True,
                        "probe_diff_contract_gate": True,
                        "todo_soft_until_first_improvement": False,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "status": "active",
                "risk_level": "local",
                "tactic_stage": "local_edit",
                "target_symbols": ["parse_item"],
                "target_regions": ["target.py::parse_item"],
                "allowed_files": ["target.py"],
            }
            snapshot = {"target.py": before}
            target.write_text("def parse_item(value):\n    return value\n")

            rejection = agent._active_probe_diff_contract_rejection(
                snapshot,
                {"target.py"},
            )

            self.assertIsNone(rejection)

    def test_probe_diff_contract_non_python_uses_file_hunk_line_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "app.js"
            before = "function handler(value) {\n  return value;\n}\n"
            target.write_text(before)
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_hard_active_todo_contract": True,
                        "probe_diff_contract_gate": True,
                        "todo_soft_until_first_improvement": False,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "status": "active",
                "risk_level": "structural",
                "tactic_stage": "structural_probe",
                "target_symbols": ["handler"],
                "target_regions": ["app.js::handler"],
                "allowed_files": ["app.js"],
                "probe_diff_contract": {
                    "allowed_files": ["app.js"],
                    "allowed_regions": ["app.js::handler"],
                    "expected_changed_regions": ["app.js::handler"],
                    "max_files_changed": 1,
                    "max_hunks": 1,
                    "max_changed_lines": 1,
                },
            }
            snapshot = {"app.js": before}
            target.write_text(before.replace("return value;", "return value ?? null;"))

            rejection = agent._active_probe_diff_contract_rejection(
                snapshot,
                {"app.js"},
            )

            self.assertIsNone(rejection)

    def test_probe_contract_mismatch_records_candidate_delta_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["active_todo"] = {
                "todo_id": "task-001",
                "status": "active",
                "tactic_stage": "structural_probe",
            }
            candidate = CodeCandidate(
                "probe-bad",
                [CodeChange("target.py", "probe", content="x = 1\n")],
                "probe candidate",
            )

            extra = agent._candidate_history_extra(
                candidate,
                status="rejected_probe_contract_mismatch",
                metric=None,
                applied=0,
                failed=True,
                patch_text="",
                results=[],
                failure_detail="probe diff contract mismatch: changed symbols outside allowed_regions",
            )

            self.assertEqual(extra["failure_class"], "probe_contract_mismatch")
            self.assertEqual(extra["failure_origin"], "post_apply_contract")
            self.assertEqual(extra["issue_scope"], "candidate_delta")
            self.assertFalse(extra["repair_task_eligible"])

    def test_single_code_candidate_probe_diff_mismatch_restores_before_test(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                target = repo / "target.py"
                before = (
                    "def parse_item(value):\n"
                    "    return value.strip()\n\n"
                    "def other(value):\n"
                    "    return value\n"
                )
                target.write_text(before)
                agent = MicroAgent(
                    {
                        "models": {},
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "writable_files": ["target.py"],
                            "spec_mode": True,
                            "spec_hard_active_todo_contract": True,
                            "probe_diff_contract_gate": True,
                            "todo_soft_until_first_improvement": False,
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = _StaticModelManager(
                    json.dumps(
                        {
                            "changes": [
                                {
                                    "path": "target.py",
                                    "reason": "probe task-001 parse_item boundary",
                                    "content": before.replace(
                                        "return value\n",
                                        "return value + 1\n",
                                    ),
                                }
                            ]
                        }
                    )
                )
                agent.state.scratch["active_todo"] = {
                    "todo_id": "task-001",
                    "spec_task_id": "task-001",
                    "status": "active",
                    "source": "spec_scheduler",
                    "risk_level": "structural",
                    "tactic_stage": "structural_probe",
                    "target_symbols": ["parse_item"],
                    "target_regions": ["target.py::parse_item"],
                    "allowed_files": ["target.py"],
                    "probe_diff_contract": {
                        "allowed_files": ["target.py"],
                        "allowed_regions": ["target.py::parse_item"],
                        "expected_changed_regions": ["target.py::parse_item"],
                        "max_files_changed": 1,
                        "max_hunks": 2,
                        "max_changed_lines": 5,
                        "max_changed_functions": 1,
                    },
                }

                await agent.mcp.start()
                try:
                    await agent.code()
                finally:
                    await agent.mcp.close()

                self.assertEqual(target.read_text(), before)
                self.assertEqual(agent.state.current, AgentStateName.TEST)
                rejection = agent.state.scratch["pre_apply_candidate_rejection"]
                self.assertEqual(
                    rejection["status"],
                    "rejected_probe_contract_mismatch",
                )
                self.assertIn("outside allowed_regions", rejection["note"])

        asyncio.run(run_case())

    def test_candidate_queue_probe_diff_mismatch_records_and_continues(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                target = repo / "target.py"
                before = (
                    "def parse_item(value):\n"
                    "    return value.strip()\n\n"
                    "def other(value):\n"
                    "    return value\n"
                )
                target.write_text(before)
                config = {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "writable_files": ["target.py"],
                        "test_commands": ["python3 -c \"print('cycles: 80')\""],
                        "metric_regex": r"cycles: (\d+)",
                        "baseline_metric": 100,
                        "candidate_queue": True,
                        "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                        "spec_mode": True,
                        "spec_hard_active_todo_contract": True,
                        "probe_diff_contract_gate": True,
                        "todo_soft_until_first_improvement": False,
                        "todo_reject_duplicate_variants": False,
                    },
                }
                agent = MicroAgent(config, AgentState(repo_root=repo, user_request="test"))
                agent.state.scratch["pre_code_snapshot"] = {"target.py": before}
                agent.state.scratch["active_todo"] = {
                    "todo_id": "task-001",
                    "spec_task_id": "task-001",
                    "status": "active",
                    "source": "spec_scheduler",
                    "risk_level": "structural",
                    "tactic_stage": "structural_probe",
                    "target_symbols": ["parse_item"],
                    "target_regions": ["target.py::parse_item"],
                    "allowed_files": ["target.py"],
                    "probe_diff_contract": {
                        "allowed_files": ["target.py"],
                        "allowed_regions": ["target.py::parse_item"],
                        "expected_changed_regions": ["target.py::parse_item"],
                        "max_files_changed": 1,
                        "max_hunks": 2,
                        "max_changed_lines": 5,
                        "max_changed_functions": 1,
                    },
                }
                bad = CodeCandidate(
                    "bad",
                    [
                        CodeChange(
                            "target.py",
                            "task-001 parse_item probe but edits other",
                            content=before.replace("return value\n", "return value + 1\n"),
                        )
                    ],
                    "task-001 parse_item probe",
                    strategy_axis="general_edit",
                )
                good = CodeCandidate(
                    "good",
                    [
                        CodeChange(
                            "target.py",
                            "task-001 parse_item probe",
                            content=before.replace(
                                "return value.strip()",
                                "return value.strip() if value else ''",
                            ),
                        )
                    ],
                    "task-001 parse_item probe",
                    strategy_axis="general_edit",
                )

                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([bad, good], {"target.py"})
                finally:
                    await agent.mcp.close()

                self.assertIn("if value else", target.read_text())
                rows = [
                    json.loads(line)
                    for line in (repo / ".local_micro_agent" / "candidates.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(rows[0]["candidate_id"], "bad")
                self.assertEqual(rows[0]["status"], "rejected_probe_contract_mismatch")
                self.assertEqual(rows[0]["failure_class"], "probe_contract_mismatch")
                self.assertEqual(rows[0]["issue_scope"], "candidate_delta")
                self.assertFalse(rows[0]["repair_task_eligible"])
                self.assertIn("diff_contract_violations", rows[0])
                self.assertEqual(rows[1]["candidate_id"], "good")
                self.assertEqual(rows[1]["status"], "improved")
                self.assertIn("Candidate bad rejected after diff check", "\n".join(agent.state.notes))

        asyncio.run(run_case())

    def test_repeated_correctness_failure_routes_task_to_design_rewrite(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                task = {
                    "task_id": "task-001",
                    "title": "guard parser branch",
                    "strategy_axis": "general_edit",
                    "status": "in_progress",
                    "deliverables": ["target.py"],
                    "target_symbols": ["parse_item"],
                    "target_regions": ["target.py::parse_item"],
                    "preserved_invariants": ["existing parse outputs stay unchanged"],
                    "edit_scope": "Change one guarded branch in parse_item.",
                    "validator": {
                        "kind": "command",
                        "failure_condition": "pytest fails",
                    },
                    "correctness_rationale": "The fallback branch is unchanged.",
                    "fallback_plan": "Revert the guarded branch.",
                    "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                    "budget": {"attempts_max": 4, "attempts_used": 0},
                }
                spec = {"version": 2, "spec_id": "perf", "task_graph": [task]}
                agent = MicroAgent(
                    {
                        "models": {},
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_design_contract_gate": True,
                            "spec_redesign_after_correctness_failures": 2,
                        },
                    },
                    AgentState(repo_root=repo, user_request="test", max_loops=10),
                )
                agent.state.scratch["run_spec"] = spec
                agent.state.scratch["current_spec_task"] = task
                agent.state.scratch["last_candidate_observation"] = {
                    "failure_class": "correctness_failure",
                    "summary": "assertion failed",
                }
                agent.state.test_results = [
                    TestResult(command="python -m pytest", exit_code=1, stderr="assertion failed")
                ]

                await agent._handle_spec_task_test_result(True)
                self.assertNotEqual(task["status"], "needs_design")

                agent.state.scratch["last_candidate_observation"] = {
                    "failure_class": "correctness_failure",
                    "summary": "assertion failed again",
                }
                await agent._handle_spec_task_test_result(True)

                self.assertEqual(task["status"], "needs_design")
                self.assertEqual(agent.state.current, AgentStateName.SPEC_SYNTH)
                self.assertIn("repeated correctness_failure", agent._spec_rewrite_focus_context())

        asyncio.run(run_case())

    def test_spec_acceptance_policy_context_guides_metric_specs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_default_acceptance_kind": "metric",
                        "spec_force_default_acceptance_kind": True,
                    },
                },
                AgentState(repo_root=Path(tmp), user_request="test"),
            )

            context = agent._spec_acceptance_policy_context()

            self.assertIn("Default acceptance kind: metric", context)
            self.assertIn("force every task", context)
            self.assertIn("do not synthesize unit tests", context)

    def test_spec_synth_falls_back_after_primary_model_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fallback_spec = json.dumps(
                {
                    "version": 2,
                    "spec_id": "fallback",
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "write target",
                            "deliverables": ["target.py"],
                            "acceptance": {"kind": "metric"},
                        }
                    ],
                }
            )
            agent = MicroAgent(
                {
                    "models": {"reasoner": "bad", "coder": "good", "default": "good"},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_enabled": True,
                        "run_spec_after_read": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_synth_fallback_model_role": "coder",
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.models = _SpecFallbackModelManager(fallback_spec)

            asyncio.run(agent._maybe_refresh_run_spec(force=True))

            spec = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
            self.assertEqual(spec["spec_id"], "fallback")
            self.assertEqual(spec["search"]["graph_id"], "graph-0001")
            graph_events = [
                json.loads(line)
                for line in (
                    repo / ".local_micro_agent" / "spec_graph_candidates.jsonl"
                ).read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(graph_events), 1)
            self.assertEqual(graph_events[0]["event"], "candidate_selected")
            self.assertEqual(graph_events[0]["status"], "selected")
            self.assertEqual(graph_events[0]["graph_id"], "graph-0001")
            self.assertTrue(
                (repo / ".local_micro_agent" / "spec_graph_candidates" / "graph-0001.json").exists()
            )
            self.assertIn("Run spec model fallback succeeded: coder", "\n".join(agent.state.notes))

    def test_spec_synth_falls_back_after_primary_parse_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fallback_spec = json.dumps(
                {
                    "version": 2,
                    "spec_id": "parse-fallback",
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "write target",
                            "deliverables": ["target.py"],
                            "acceptance": {"kind": "metric"},
                        }
                    ],
                }
            )
            agent = MicroAgent(
                {
                    "models": {"reasoner": "bad", "coder": "good", "default": "good"},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "run_spec_enabled": True,
                        "run_spec_after_read": True,
                        "run_spec_path": ".local_micro_agent/run_spec.json",
                        "spec_synth_fallback_model_role": "coder",
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.models = _SpecParseFallbackModelManager(fallback_spec)

            asyncio.run(agent._maybe_refresh_run_spec(force=True))

            spec = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
            self.assertEqual(spec["spec_id"], "parse-fallback")
            notes = "\n".join(agent.state.notes)
            self.assertIn("Run spec JSON parse failed", notes)
            self.assertIn("Run spec model fallback succeeded: coder", notes)

    def test_spec_idea_prompt_is_markdown_advisory_not_json_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("def parse_item(value):\n    return value\n")
            state = AgentState(repo_root=repo, user_request="optimize target")
            state.plan_markdown = "Read target.py and create a grounded spec."
            state.file_context = [FileSnapshot(path="target.py", content=target.read_text())]

            messages = spec_idea_prompt(
                state,
                focus="allowed_target_regions: target.py::parse_item",
            )

            self.assertIn("Do not write code and do not emit run_spec JSON", messages[0]["content"])
            self.assertIn("allowed_target_regions", messages[1]["content"])
            self.assertIn("target.py::parse_item", messages[1]["content"])

    def test_two_call_spec_synthesis_passes_idea_brief_to_finalizer(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
                final_spec = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "two-call",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "guard parser branch",
                                "deliverables": ["target.py"],
                                "target_symbols": ["parse_item"],
                                "target_regions": ["target.py::parse_item"],
                                "acceptance": {"kind": "metric"},
                            }
                        ],
                    }
                )
                manager = _RoleModelManager(
                    {
                        "reasoner": "Idea brief: use target.py::parse_item only.",
                        "spec_synth": final_spec,
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {
                            "reasoner": "reasoner-model",
                            "spec_synth": "spec-model",
                            "default": "spec-model",
                        },
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_two_call_synthesis": True,
                            "spec_idea_model_role": "reasoner",
                            "spec_finalize_model_role": "spec_synth",
                            "spec_grounding_gate": True,
                            "writable_files": ["target.py"],
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = manager
                agent.state.plan_markdown = "Plan"
                agent.state.file_context = [
                    FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
                ]

                await agent._maybe_refresh_run_spec(force=True)

                spec = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
                self.assertEqual(spec["spec_id"], "two-call")
                self.assertTrue((repo / ".local_micro_agent" / "spec_idea.md").exists())
                finalizer_prompt = manager.seen["spec_synth"][0][-1]["content"]
                self.assertIn("Spec idea brief", finalizer_prompt)
                self.assertIn("target.py::parse_item", finalizer_prompt)

        asyncio.run(run_case())

    def test_two_call_spec_synthesis_continues_after_reasoning_only_idea(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
                final_spec = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "facts-only",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "guard parser branch",
                                "deliverables": ["target.py"],
                                "target_symbols": ["parse_item"],
                                "target_regions": ["target.py::parse_item"],
                                "acceptance": {"kind": "metric"},
                            }
                        ],
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {
                            "reasoner": "reasoner-model",
                            "spec_synth": "spec-model",
                            "default": "spec-model",
                        },
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_two_call_synthesis": True,
                            "spec_idea_model_role": "reasoner",
                            "spec_finalize_model_role": "spec_synth",
                            "spec_grounding_gate": True,
                            "writable_files": ["target.py"],
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = _SpecIdeaReasoningOnlyManager(final_spec)
                agent.state.plan_markdown = "Plan"
                agent.state.file_context = [
                    FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
                ]

                await agent._maybe_refresh_run_spec(force=True)

                spec = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
                self.assertEqual(spec["spec_id"], "facts-only")
                self.assertFalse((repo / ".local_micro_agent" / "spec_idea.md").exists())
                self.assertIn("Spec idea model call failed", "\n".join(agent.state.notes))

        asyncio.run(run_case())

    @staticmethod
    def _valid_hypothesis_brief(
        *,
        hypothesis_id: str = "hyp-parse-guard",
        region: str = "target.py::parse_item",
        kind: str = "local_edit",
        why_not_smaller: str = (
            "parse_item is the smallest writable symbol exposed by grounding facts."
        ),
        domain_terms: str = "parser guard branch",
    ) -> str:
        return f"""BEGIN_HYPOTHESIS_OPTION {hypothesis_id}
hypothesis: {domain_terms} should improve the configured objective without changing the public return contract.
change_boundary.regions: {region}
change_boundary.kind: {kind}
change_boundary.minimality_claim: {region} is the smallest resolvable writable boundary for this claim.
causal_evidence: Source shows parse_item returns the input directly, so the option is grounded in current source.
expected_signal.validator_kind: metric
expected_signal.command_or_metric: configured workflow metric
expected_signal.success_condition: metric improves or stays safe while deterministic tests pass.
invariants: parse_item keeps returning valid input values unchanged.
fallback.on_failure: revert this hypothesis and retarget a different boundary.
fallback.preserve: sibling hypotheses and parse_item public return behavior.
why_not_smaller: {why_not_smaller}
END_HYPOTHESIS_OPTION"""

    @staticmethod
    def _valid_hypothesis_spec(
        *,
        spec_id: str = "hypothesis-normal",
        hypothesis_id: str = "hyp-parse-guard",
        region: str = "target.py::parse_item",
    ) -> str:
        edit_scope = "Add one guarded parser branch inside parse_item before returning value."
        return json.dumps(
            {
                "version": 2,
                "spec_id": spec_id,
                "objective": "Validate a bounded parser hypothesis.",
                "invariants": ["parse_item public return behavior remains stable"],
                "known_facts": ["target.py::parse_item is writable"],
                "task_graph": [
                    {
                        "task_id": "task-001",
                        "hypothesis_id": hypothesis_id,
                        "title": "guard parser branch",
                        "strategy_axis": "parser_guard",
                        "family_key": "input_validation",
                        "expected_signal": "configured metric improves while tests pass",
                        "target_symbols": ["parse_item"],
                        "target_regions": [region],
                        "preserved_invariants": [
                            "parse_item returns valid input values unchanged"
                        ],
                        "edit_scope": edit_scope,
                        "risk_level": "local",
                        "tactic_stage": "local_edit",
                        "risk_evidence": {
                            "field": "edit_scope",
                            "quote": edit_scope,
                            "explanation": "the edit stays inside one local branch",
                        },
                        "probe_plan": "Add one guarded parser branch and measure.",
                        "invariant_evidence": ["existing return behavior is preserved"],
                        "validator": {
                            "kind": "metric",
                            "failure_condition": "metric does not improve or tests fail",
                        },
                        "correctness_rationale": (
                            "The fallback path keeps the existing return behavior."
                        ),
                        "fallback_plan": "Revert the guarded branch and preserve parse_item.",
                        "rollback_or_shrink_plan": (
                            "Revert the guarded branch to restore parse_item."
                        ),
                        "status": "open",
                        "depends_on": [],
                        "deliverables": ["target.py"],
                        "read_hints": ["target.py"],
                        "acceptance": {"kind": "metric"},
                        "budget": {"attempts_max": 1, "attempts_used": 0},
                    }
                ],
            }
        )

    def test_spec_thinking_brief_accepts_reasoning_only_output(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
                brief = "Use target.py::parse_item and avoid broad structural probes."
                final_spec = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "think-brief",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "guard parser branch",
                                "deliverables": ["target.py"],
                                "target_symbols": ["parse_item"],
                                "target_regions": ["target.py::parse_item"],
                                "acceptance": {"kind": "metric"},
                            }
                        ],
                    }
                )
                manager = _RoleModelManager(
                    {
                        "reasoner": ModelResponse(
                            "",
                            usage={
                                "reasoning_only_response": True,
                                "reasoning_content_chars": len(brief),
                            },
                            reasoning=brief,
                        ),
                        "spec_synth": final_spec,
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {
                            "reasoner": "reasoner-model",
                            "spec_synth": "spec-model",
                            "default": "spec-model",
                        },
                        "providers": {
                            "reasoner-model": {"kind": "ollama_native", "think": True},
                            "spec-model": {"kind": "ollama_native", "think": False},
                        },
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_two_call_synthesis": True,
                            "spec_thinking_brief_enabled": True,
                            "spec_thinking_brief_model_role": "reasoner",
                            "spec_finalize_model_role": "spec_synth",
                            "spec_grounding_gate": True,
                            "writable_files": ["target.py"],
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = manager
                agent.state.plan_markdown = "Plan"
                agent.state.file_context = [
                    FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
                ]

                await agent._maybe_refresh_run_spec(force=True)

                spec = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
                self.assertEqual(spec["spec_id"], "think-brief")
                self.assertFalse((repo / ".local_micro_agent" / "spec_idea.md").exists())
                self.assertEqual(
                    (repo / ".local_micro_agent" / "spec_think_brief.md").read_text().strip(),
                    brief,
                )
                meta = json.loads(
                    (repo / ".local_micro_agent" / "spec_think_brief_meta.json").read_text()
                )
                self.assertEqual(meta["selected_source"], "reasoning")
                self.assertTrue(meta["reasoning_only_accepted"])
                self.assertTrue(meta["thinking_enabled"])
                constraints = json.loads(
                    (repo / ".local_micro_agent" / "spec_synthesis_constraints.json").read_text()
                )
                self.assertIn("target.py::parse_item", constraints["allowed_target_regions"])
                finalizer_prompt = manager.seen["spec_synth"][0][-1]["content"]
                self.assertIn("Spec thinking brief", finalizer_prompt)
                self.assertIn("Controller-owned SPEC synthesis constraints", finalizer_prompt)
                self.assertIn("target.py::parse_item", finalizer_prompt)

        asyncio.run(run_case())

    def test_model_thinking_brief_falls_back_to_content_only_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {},
                },
                state,
            )
            agent.models = _StaticModelManager("Content-only brief")

            parts = asyncio.run(
                agent._model_thinking_brief(
                    "reasoner",
                    [{"role": "user", "content": "brief"}],
                    call_site="spec_think_brief",
                )
            )

            self.assertEqual(parts.content, "Content-only brief")
            self.assertEqual(parts.reasoning, "")
            self.assertEqual(parts.source, "content")

    def test_spec_thinking_brief_empty_output_continues_with_constraints(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
                final_spec = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "empty-brief",
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "guard parser branch",
                                "deliverables": ["target.py"],
                                "target_symbols": ["parse_item"],
                                "target_regions": ["target.py::parse_item"],
                                "acceptance": {"kind": "metric"},
                            }
                        ],
                    }
                )
                manager = _RoleModelManager(
                    {
                        "reasoner": ModelResponse("", usage={}, reasoning=""),
                        "spec_synth": final_spec,
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {
                            "reasoner": "reasoner-model",
                            "spec_synth": "spec-model",
                            "default": "spec-model",
                        },
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_two_call_synthesis": True,
                            "spec_thinking_brief_enabled": True,
                            "spec_thinking_brief_model_role": "reasoner",
                            "spec_finalize_model_role": "spec_synth",
                            "spec_grounding_gate": True,
                            "writable_files": ["target.py"],
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = manager
                agent.state.plan_markdown = "Plan"
                agent.state.file_context = [
                    FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
                ]

                await agent._maybe_refresh_run_spec(force=True)

                spec = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
                self.assertEqual(spec["spec_id"], "empty-brief")
                self.assertFalse((repo / ".local_micro_agent" / "spec_think_brief.md").exists())
                meta = json.loads(
                    (repo / ".local_micro_agent" / "spec_think_brief_meta.json").read_text()
                )
                self.assertEqual(meta["selected_source"], "empty")
                self.assertIn(
                    "Spec think brief returned empty output",
                    "\n".join(agent.state.notes),
                )
                finalizer_prompt = manager.seen["spec_synth"][0][-1]["content"]
                self.assertIn("Controller-owned SPEC synthesis constraints", finalizer_prompt)

        asyncio.run(run_case())

    def test_spec_hypothesis_brief_accepts_structured_option(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
                manager = _RoleModelManager(
                    {
                        "reasoner": ModelResponse(
                            "",
                            usage={"reasoning_only_response": True},
                            reasoning=self._valid_hypothesis_brief(),
                        ),
                        "spec_synth": self._valid_hypothesis_spec(),
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {
                            "reasoner": "reasoner-model",
                            "spec_synth": "spec-model",
                            "default": "spec-model",
                        },
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_two_call_synthesis": True,
                            "spec_thinking_brief_enabled": True,
                            "spec_hypothesis_brief_enabled": True,
                            "spec_finalize_model_role": "spec_synth",
                            "spec_quality_gate": True,
                            "spec_quality_rewrite_attempts": 0,
                            "spec_grounding_gate": True,
                            "writable_files": ["target.py"],
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = manager
                agent.state.plan_markdown = "Plan"
                agent.state.file_context = [
                    FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
                ]

                await agent._maybe_refresh_run_spec(force=True)

                spec = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
                self.assertEqual(spec["task_graph"][0]["hypothesis_id"], "hyp-parse-guard")
                options = json.loads(
                    (repo / ".local_micro_agent" / "spec_hypothesis_options.json").read_text()
                )
                self.assertEqual(options["accepted_count"], 1)
                self.assertEqual(options["rejected_count"], 0)
                finalizer_prompt = manager.seen["spec_synth"][0][-1]["content"]
                self.assertIn("Accepted SPEC hypothesis options", finalizer_prompt)
                self.assertIn("hyp-parse-guard", finalizer_prompt)

        asyncio.run(run_case())

    def test_spec_hypothesis_brief_repairs_freeform_brief_once(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
                manager = _RoleSequenceModelManager(
                    {
                        "reasoner": [
                            "We should inspect parse_item and make a narrow guard.",
                            self._valid_hypothesis_brief(),
                        ],
                        "spec_synth": [self._valid_hypothesis_spec()],
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {
                            "reasoner": "reasoner-model",
                            "spec_synth": "spec-model",
                            "default": "spec-model",
                        },
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_two_call_synthesis": True,
                            "spec_thinking_brief_enabled": True,
                            "spec_hypothesis_brief_enabled": True,
                            "spec_finalize_model_role": "spec_synth",
                            "spec_quality_gate": True,
                            "spec_quality_rewrite_attempts": 0,
                            "spec_grounding_gate": True,
                            "writable_files": ["target.py"],
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = manager
                agent.state.plan_markdown = "Plan"
                agent.state.file_context = [
                    FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
                ]

                await agent._maybe_refresh_run_spec(force=True)

                spec = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
                self.assertEqual(spec["task_graph"][0]["hypothesis_id"], "hyp-parse-guard")
                self.assertEqual(len(manager.seen["reasoner"]), 2)
                repair_prompt = manager.seen["reasoner"][1][-1]["content"]
                self.assertIn("Repair the typed hypothesis blocks", repair_prompt)
                self.assertIn("Fix every validation issue", repair_prompt)
                self.assertIn("BEGIN_HYPOTHESIS_OPTION", repair_prompt)
                options = json.loads(
                    (repo / ".local_micro_agent" / "spec_hypothesis_options.json").read_text()
                )
                self.assertEqual(options["accepted_count"], 1)
                meta = json.loads(
                    (repo / ".local_micro_agent" / "spec_think_brief_meta.json").read_text()
                )
                self.assertTrue(meta["hypothesis_repair"]["attempted"])

        asyncio.run(run_case())

    def test_spec_hypothesis_brief_skips_repair_when_options_valid(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
                manager = _RoleSequenceModelManager(
                    {
                        "reasoner": [self._valid_hypothesis_brief()],
                        "spec_synth": [self._valid_hypothesis_spec()],
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {
                            "reasoner": "reasoner-model",
                            "spec_synth": "spec-model",
                            "default": "spec-model",
                        },
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_two_call_synthesis": True,
                            "spec_thinking_brief_enabled": True,
                            "spec_hypothesis_brief_enabled": True,
                            "spec_finalize_model_role": "spec_synth",
                            "spec_quality_gate": True,
                            "spec_quality_rewrite_attempts": 0,
                            "spec_grounding_gate": True,
                            "writable_files": ["target.py"],
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = manager
                agent.state.plan_markdown = "Plan"
                agent.state.file_context = [
                    FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
                ]

                await agent._maybe_refresh_run_spec(force=True)

                self.assertTrue((repo / ".local_micro_agent" / "run_spec.json").exists())
                self.assertEqual(len(manager.seen["reasoner"]), 1)

        asyncio.run(run_case())

    def test_spec_hypothesis_brief_repair_failure_stays_hard_blocked(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
                manager = _RoleSequenceModelManager(
                    {
                        "reasoner": [
                            "Free-form analysis only.",
                            "Still free-form with no typed blocks.",
                        ],
                        "spec_synth": [self._valid_hypothesis_spec()],
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {
                            "reasoner": "reasoner-model",
                            "spec_synth": "spec-model",
                            "default": "spec-model",
                        },
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_two_call_synthesis": True,
                            "spec_thinking_brief_enabled": True,
                            "spec_hypothesis_brief_enabled": True,
                            "spec_finalize_model_role": "spec_synth",
                            "spec_quality_gate": True,
                            "spec_quality_rewrite_attempts": 0,
                            "spec_gate_soft_fallback": True,
                            "spec_grounding_gate": True,
                            "writable_files": ["target.py"],
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = manager
                agent.state.plan_markdown = "Plan"
                agent.state.file_context = [
                    FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
                ]

                await agent._maybe_refresh_run_spec(force=True)

                self.assertFalse((repo / ".local_micro_agent" / "run_spec.json").exists())
                self.assertEqual(len(manager.seen["reasoner"]), 2)
                report = json.loads(
                    (repo / ".local_micro_agent" / "spec_quality_report.json").read_text()
                )
                self.assertIn("hypothesis_option_missing", report["issue_codes"])
                progress_path = repo / ".local_micro_agent" / "spec_progress.jsonl"
                progress = progress_path.read_text() if progress_path.exists() else ""
                self.assertNotIn("quality_soft_fallback", progress)

        asyncio.run(run_case())

    def test_spec_hypothesis_task_repair_persists_guarded_shrink(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "target.py").write_text(
                    "def parse_item(value):\n    return value\n\n"
                    "def helper(value):\n    return value\n"
                )
                brief = self._valid_hypothesis_brief(
                    hypothesis_id="hyp-wide",
                    region="target.py::parse_item, target.py::helper",
                    kind="local_edit",
                    why_not_smaller=(
                        "The claim spans parse_item and helper, but finalizer may "
                        "start with one single guarded branch."
                    ),
                )
                bad_spec = self._valid_hypothesis_spec(
                    spec_id="bad-shrink",
                    hypothesis_id="hyp-wide",
                    region="target.py::parse_item",
                )
                bad_payload = json.loads(bad_spec)
                bad_task = bad_payload["task_graph"][0]
                bad_task["edit_scope"] = "Change parse_item implementation."
                bad_task["probe_plan"] = "Change parse_item implementation."
                bad_task["fallback_plan"] = "Revert the patch."
                bad_task["rollback_or_shrink_plan"] = "Revert the patch."
                bad_task["risk_evidence"] = {
                    "field": "edit_scope",
                    "quote": "Change parse_item implementation",
                    "explanation": "implementation change",
                }
                repair_spec = self._valid_hypothesis_spec(
                    spec_id="repaired-shrink",
                    hypothesis_id="hyp-wide",
                    region="target.py::parse_item",
                )
                repair_payload = json.loads(repair_spec)
                repair_task = repair_payload["task_graph"][0]
                repair_task["rollback_or_shrink_plan"] = (
                    "Shrink to one single guarded branch in parse_item; keep the "
                    "old return path as fallback and revert the branch if it fails."
                )
                repair_spec = json.dumps(repair_payload)
                manager = _RoleSequenceModelManager(
                    {
                        "reasoner": [brief],
                        "spec_synth": [json.dumps(bad_payload), repair_spec],
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {
                            "reasoner": "reasoner-model",
                            "spec_synth": "spec-model",
                            "default": "spec-model",
                        },
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_two_call_synthesis": True,
                            "spec_thinking_brief_enabled": True,
                            "spec_hypothesis_brief_enabled": True,
                            "spec_finalize_model_role": "spec_synth",
                            "spec_quality_gate": True,
                            "spec_quality_rewrite_attempts": 0,
                            "spec_grounding_gate": True,
                            "writable_files": ["target.py"],
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = manager
                agent.state.plan_markdown = "Plan"
                agent.state.file_context = [
                    FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
                ]

                await agent._maybe_refresh_run_spec(force=True)

                spec = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
                self.assertEqual(spec["spec_id"], "repaired-shrink")
                self.assertEqual(len(manager.seen["spec_synth"]), 2)
                repair_prompt = manager.seen["spec_synth"][1][-1]["content"]
                self.assertIn("Repair context", repair_prompt)
                self.assertIn("hypothesis_boundary_shrink_plan_missing", repair_prompt)
                progress = (repo / ".local_micro_agent" / "spec_progress.jsonl").read_text()
                self.assertIn("quality_repaired", progress)

        asyncio.run(run_case())

    def test_spec_hypothesis_task_repair_skips_non_hypothesis_quality_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_hypothesis_brief_enabled": True,
                        "spec_hypothesis_task_repair_enabled": True,
                    },
                },
                AgentState(repo_root=Path(tmp), user_request="test"),
            )
            agent.state.scratch["spec_hypothesis_options"] = {
                "accepted": [
                    {
                        "hypothesis_id": "hyp-parse",
                        "change_boundary": {
                            "regions": ["target.py::parse_item"],
                            "kind": "local_edit",
                        },
                    }
                ]
            }
            report = {"issue_codes": ["too_many_deliverables"], "issues": []}

            self.assertFalse(agent._spec_hypothesis_task_repair_needed(report))

    def test_spec_hypothesis_task_repair_failure_keeps_hard_block(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "target.py").write_text(
                    "def parse_item(value):\n    return value\n\n"
                    "def helper(value):\n    return value\n"
                )
                brief = self._valid_hypothesis_brief(
                    hypothesis_id="hyp-wide",
                    region="target.py::parse_item, target.py::helper",
                    kind="local_edit",
                    why_not_smaller="The claim spans both regions.",
                )
                bad_spec = self._valid_hypothesis_spec(
                    spec_id="bad-shrink",
                    hypothesis_id="hyp-wide",
                    region="target.py::parse_item",
                )
                bad_payload = json.loads(bad_spec)
                bad_task = bad_payload["task_graph"][0]
                bad_task["edit_scope"] = "Change parse_item implementation."
                bad_task["probe_plan"] = "Change parse_item implementation."
                bad_task["fallback_plan"] = "Revert the patch."
                bad_task["rollback_or_shrink_plan"] = "Revert the patch."
                bad_task["risk_evidence"] = {
                    "field": "edit_scope",
                    "quote": "Change parse_item implementation",
                    "explanation": "implementation change",
                }
                manager = _RoleSequenceModelManager(
                    {
                        "reasoner": [brief],
                        "spec_synth": [json.dumps(bad_payload), json.dumps(bad_payload)],
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {
                            "reasoner": "reasoner-model",
                            "spec_synth": "spec-model",
                            "default": "spec-model",
                        },
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_two_call_synthesis": True,
                            "spec_thinking_brief_enabled": True,
                            "spec_hypothesis_brief_enabled": True,
                            "spec_finalize_model_role": "spec_synth",
                            "spec_quality_gate": True,
                            "spec_quality_rewrite_attempts": 0,
                            "spec_gate_soft_fallback": True,
                            "spec_grounding_gate": True,
                            "writable_files": ["target.py"],
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = manager
                agent.state.plan_markdown = "Plan"
                agent.state.file_context = [
                    FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
                ]

                await agent._maybe_refresh_run_spec(force=True)

                self.assertFalse((repo / ".local_micro_agent" / "run_spec.json").exists())
                self.assertEqual(len(manager.seen["spec_synth"]), 2)
                progress = (repo / ".local_micro_agent" / "spec_progress.jsonl").read_text()
                self.assertNotIn("quality_soft_fallback", progress)
                self.assertIn("hypothesis_task_repair", progress)
                self.assertIn("soft fallback blocked", "\n".join(agent.state.notes))

        asyncio.run(run_case())

    def test_spec_hypothesis_brief_accepts_domain_terms_without_interpretation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_hypothesis_brief_enabled": True,
                        "spec_grounding_gate": True,
                        "writable_files": ["target.py"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.file_context = [
                FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
            ]
            agent._spec_grounding_facts_context()
            brief = self._valid_hypothesis_brief(
                domain_terms="VLIW slot-packer pressure around the parser guard"
            )

            context = agent._spec_hypothesis_options_context(brief)

            options = json.loads(
                (repo / ".local_micro_agent" / "spec_hypothesis_options.json").read_text()
            )
            self.assertEqual(options["accepted_count"], 1)
            self.assertIn("VLIW slot-packer", options["accepted"][0]["hypothesis"])
            self.assertIn("Accepted SPEC hypothesis options", context)

    def test_spec_hypothesis_brief_rejects_structural_action_as_local_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("def build(value):\n    return value\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_hypothesis_brief_enabled": True,
                        "spec_grounding_gate": True,
                        "writable_files": ["target.py"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.file_context = [
                FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
            ]
            agent._spec_grounding_facts_context()
            brief = self._valid_hypothesis_brief(
                hypothesis_id="hyp-pack-local",
                region="target.py::build",
                kind="local_edit",
                domain_terms=(
                    "group instructions into bundles and change scheduling order "
                    "inside build"
                ),
            )

            agent._spec_hypothesis_options_context(brief)

            options = json.loads(
                (repo / ".local_micro_agent" / "spec_hypothesis_options.json").read_text()
            )
            self.assertEqual(options["accepted_count"], 0)
            self.assertIn(
                "structural_hypothesis_boundary_kind_mismatch",
                options["rejected"][0]["issues"],
            )

    def test_spec_hypothesis_brief_rejects_freeform_tasks(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
                manager = _RoleModelManager(
                    {
                        "reasoner": ModelResponse(
                            "",
                            usage={"reasoning_only_response": True},
                            reasoning="We should rewrite target.py::parse_item broadly.",
                        ),
                        "spec_synth": self._valid_hypothesis_spec(),
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {
                            "reasoner": "reasoner-model",
                            "spec_synth": "spec-model",
                            "default": "spec-model",
                        },
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_two_call_synthesis": True,
                            "spec_thinking_brief_enabled": True,
                            "spec_hypothesis_brief_enabled": True,
                            "spec_finalize_model_role": "spec_synth",
                            "spec_quality_gate": True,
                            "spec_quality_rewrite_attempts": 0,
                            "spec_grounding_gate": True,
                            "writable_files": ["target.py"],
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = manager
                agent.state.plan_markdown = "Plan"
                agent.state.file_context = [
                    FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
                ]

                await agent._maybe_refresh_run_spec(force=True)

                self.assertFalse((repo / ".local_micro_agent" / "run_spec.json").exists())
                options = json.loads(
                    (repo / ".local_micro_agent" / "spec_hypothesis_options.json").read_text()
                )
                self.assertEqual(options["accepted_count"], 0)
                self.assertIn(
                    "missing_hypothesis_option_blocks",
                    options["rejected"][0]["issues"],
                )
                report = json.loads(
                    (repo / ".local_micro_agent" / "spec_quality_report.json").read_text()
                )
                self.assertIn("hypothesis_option_missing", report["issue_codes"])

        asyncio.run(run_case())

    def test_spec_hypothesis_brief_empty_output_clears_stale_options(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
                artifact_dir = repo / ".local_micro_agent"
                artifact_dir.mkdir()
                (artifact_dir / "spec_hypothesis_options.json").write_text(
                    json.dumps(
                        {
                            "version": 1,
                            "accepted": [
                                {
                                    "hypothesis_id": "hyp-parse-guard",
                                    "change_boundary": {
                                        "regions": ["target.py::parse_item"]
                                    },
                                }
                            ],
                            "accepted_count": 1,
                            "rejected": [],
                            "rejected_count": 0,
                        }
                    )
                    + "\n"
                )
                manager = _RoleModelManager(
                    {
                        "reasoner": ModelResponse("", usage={}, reasoning=""),
                        "spec_synth": self._valid_hypothesis_spec(),
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {
                            "reasoner": "reasoner-model",
                            "spec_synth": "spec-model",
                            "default": "spec-model",
                        },
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_two_call_synthesis": True,
                            "spec_thinking_brief_enabled": True,
                            "spec_hypothesis_brief_enabled": True,
                            "spec_finalize_model_role": "spec_synth",
                            "spec_quality_gate": True,
                            "spec_quality_rewrite_attempts": 0,
                            "spec_grounding_gate": True,
                            "writable_files": ["target.py"],
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = manager
                agent.state.plan_markdown = "Plan"
                agent.state.file_context = [
                    FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
                ]

                await agent._maybe_refresh_run_spec(force=True)

                self.assertFalse((repo / ".local_micro_agent" / "run_spec.json").exists())
                options = json.loads(
                    (repo / ".local_micro_agent" / "spec_hypothesis_options.json").read_text()
                )
                self.assertEqual(options["accepted_count"], 0)
                report = json.loads(
                    (repo / ".local_micro_agent" / "spec_quality_report.json").read_text()
                )
                self.assertIn("hypothesis_option_missing", report["issue_codes"])

        asyncio.run(run_case())

    def test_spec_hypothesis_brief_rejects_unknown_task_hypothesis_id(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
                manager = _RoleModelManager(
                    {
                        "reasoner": ModelResponse(
                            "",
                            usage={"reasoning_only_response": True},
                            reasoning=self._valid_hypothesis_brief(),
                        ),
                        "spec_synth": self._valid_hypothesis_spec(
                            hypothesis_id="hyp-unaccepted"
                        ),
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {
                            "reasoner": "reasoner-model",
                            "spec_synth": "spec-model",
                            "default": "spec-model",
                        },
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_two_call_synthesis": True,
                            "spec_thinking_brief_enabled": True,
                            "spec_hypothesis_brief_enabled": True,
                            "spec_finalize_model_role": "spec_synth",
                            "spec_quality_gate": True,
                            "spec_quality_rewrite_attempts": 0,
                            "spec_grounding_gate": True,
                            "writable_files": ["target.py"],
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = manager
                agent.state.plan_markdown = "Plan"
                agent.state.file_context = [
                    FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
                ]

                await agent._maybe_refresh_run_spec(force=True)

                self.assertFalse((repo / ".local_micro_agent" / "run_spec.json").exists())
                report = json.loads(
                    (repo / ".local_micro_agent" / "spec_quality_report.json").read_text()
                )
                self.assertIn("hypothesis_id_unknown", report["issue_codes"])

        asyncio.run(run_case())

    def test_spec_hypothesis_brief_rejects_broad_boundary_without_why(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text(
                "def parse_item(value):\n    return value\n\n"
                "def format_item(value):\n    return str(value)\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_hypothesis_brief_enabled": True,
                        "spec_grounding_gate": True,
                        "writable_files": ["target.py"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.file_context = [
                FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
            ]
            agent._spec_grounding_facts_context()
            brief = self._valid_hypothesis_brief(
                region="target.py::parse_item, target.py::format_item",
                kind="structural_probe",
                why_not_smaller="",
            )

            agent._spec_hypothesis_options_context(brief)

            options = json.loads(
                (repo / ".local_micro_agent" / "spec_hypothesis_options.json").read_text()
            )
            self.assertEqual(options["accepted_count"], 0)
            self.assertIn("missing_why_not_smaller", options["rejected"][0]["issues"])

    def test_spec_quality_rejects_structural_hypothesis_as_local_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("def build(value):\n    return value\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_quality_gate": True,
                        "spec_hypothesis_brief_enabled": True,
                        "writable_files": ["target.py"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["spec_hypothesis_options"] = {
                "accepted": [
                    {
                        "hypothesis_id": "hyp-broad-pack",
                        "change_boundary": {
                            "regions": ["target.py::build"],
                            "kind": "structural_probe",
                        },
                    }
                ]
            }
            spec = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "quality-hypothesis-stage-mismatch",
                    "invariants": ["build return contract stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "hypothesis_id": "hyp-broad-pack",
                            "title": "pack build work",
                            "strategy_axis": "general_edit",
                            "expected_signal": "pytest stays green",
                            "deliverables": ["target.py"],
                            "target_symbols": ["build"],
                            "target_regions": ["target.py::build"],
                            "preserved_invariants": ["return value remains unchanged"],
                            "edit_scope": "Pack independent operations in build.",
                            "risk_level": "local",
                            "tactic_stage": "local_edit",
                            "risk_evidence": {
                                "field": "edit_scope",
                                "quote": "Pack independent operations",
                                "explanation": "incorrectly labeled local",
                            },
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The original return path is preserved.",
                            "fallback_plan": "Revert the patch.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            report = agent._spec_quality_report(spec)

            self.assertEqual(report["status"], "fail")
            self.assertIn(
                "hypothesis_boundary_structural_task_mismatch",
                report["issue_codes"],
            )

    def test_spec_quality_allows_explicit_single_region_shrink_from_structural_hypothesis(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text(
                "def build(value):\n    return value\n\n"
                "def helper(value):\n    return value\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_quality_gate": True,
                        "spec_hypothesis_brief_enabled": True,
                        "writable_files": ["target.py"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["spec_hypothesis_options"] = {
                "accepted": [
                    {
                        "hypothesis_id": "hyp-wide-pack",
                        "change_boundary": {
                            "regions": ["target.py::build", "target.py::helper"],
                            "kind": "structural_probe",
                        },
                    }
                ]
            }
            spec = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "quality-hypothesis-local-shrink",
                    "invariants": ["build return contract stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "hypothesis_id": "hyp-wide-pack",
                            "title": "single guarded build probe",
                            "strategy_axis": "general_edit",
                            "expected_signal": "pytest stays green",
                            "deliverables": ["target.py"],
                            "target_symbols": ["build"],
                            "target_regions": ["target.py::build"],
                            "preserved_invariants": ["return value remains unchanged"],
                            "edit_scope": (
                                "Add one single guarded branch inside build as the "
                                "smaller probe."
                            ),
                            "risk_level": "local",
                            "tactic_stage": "local_edit",
                            "risk_evidence": {
                                "field": "edit_scope",
                                "quote": "one single guarded branch",
                                "explanation": "explicit narrowed local probe",
                            },
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The original return path is preserved.",
                            "fallback_plan": "Keep the old path as fallback.",
                            "rollback_or_shrink_plan": (
                                "Shrink to this single guarded branch; revert if it fails."
                            ),
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            report = agent._spec_quality_report(spec)

            self.assertEqual(report["status"], "pass")

    def test_spec_quality_rejects_multi_region_hypothesis_without_shrink_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text(
                "def build(value):\n    return value\n\n"
                "def helper(value):\n    return value\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_quality_gate": True,
                        "spec_hypothesis_brief_enabled": True,
                        "writable_files": ["target.py"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.scratch["spec_hypothesis_options"] = {
                "accepted": [
                    {
                        "hypothesis_id": "hyp-wide-pack",
                        "change_boundary": {
                            "regions": ["target.py::build", "target.py::helper"],
                            "kind": "structural_probe",
                        },
                    }
                ]
            }
            spec = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "quality-hypothesis-shrink-missing",
                    "invariants": ["build return contract stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "hypothesis_id": "hyp-wide-pack",
                            "title": "build probe",
                            "strategy_axis": "general_edit",
                            "expected_signal": "pytest stays green",
                            "deliverables": ["target.py"],
                            "target_symbols": ["build"],
                            "target_regions": ["target.py::build"],
                            "preserved_invariants": ["return value remains unchanged"],
                            "edit_scope": "Change build implementation.",
                            "risk_level": "structural",
                            "tactic_stage": "structural_probe",
                            "risk_evidence": {
                                "field": "edit_scope",
                                "quote": "Change build implementation",
                                "explanation": "structural implementation change",
                            },
                            "probe_plan": "Change build implementation.",
                            "invariant_evidence": ["The old return behavior is preserved."],
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The original return path is preserved.",
                            "fallback_plan": "Revert the patch.",
                            "rollback_or_shrink_plan": "Revert the patch.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            report = agent._spec_quality_report(spec)

            self.assertEqual(report["status"], "fail")
            self.assertIn(
                "hypothesis_boundary_shrink_plan_missing",
                report["issue_codes"],
            )

    def test_spec_hypothesis_brief_rejects_unresolved_symbol_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_hypothesis_brief_enabled": True,
                        "spec_grounding_gate": True,
                        "writable_files": ["target.py"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.file_context = [
                FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
            ]
            agent._spec_grounding_facts_context()
            brief = self._valid_hypothesis_brief(region="target.py::missing_symbol")

            agent._spec_hypothesis_options_context(brief)

            options = json.loads(
                (repo / ".local_micro_agent" / "spec_hypothesis_options.json").read_text()
            )
            self.assertEqual(options["accepted_count"], 0)
            self.assertIn(
                "unresolved_or_non_writable_boundary:target.py::missing_symbol",
                options["rejected"][0]["issues"],
            )

    def test_spec_quality_gate_accepts_single_target_spec(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
                final_spec = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "quality-normal",
                        "invariants": ["parse_item return contract stays unchanged"],
                        "known_facts": ["target.py::parse_item is writable"],
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "guard parse branch",
                                "strategy_axis": "general_edit",
                                "expected_signal": "python -m pytest stays green",
                                "deliverables": ["target.py"],
                                "read_hints": ["target.py"],
                                "target_symbols": ["parse_item"],
                                "target_regions": ["target.py::parse_item"],
                                "preserved_invariants": ["return value remains unchanged"],
                                "edit_scope": "Add one guarded branch inside parse_item.",
                                "risk_level": "local",
                                "tactic_stage": "local_edit",
                                "risk_evidence": {
                                    "field": "edit_scope",
                                    "quote": "one guarded branch",
                                    "explanation": "single guarded local edit",
                                },
                                "validator": {
                                    "kind": "command",
                                    "failure_condition": "pytest fails",
                                },
                                "correctness_rationale": "The original return path is preserved.",
                                "fallback_plan": "Revert the guarded branch.",
                                "acceptance": {
                                    "kind": "command",
                                    "commands": ["python -m pytest"],
                                },
                            }
                        ],
                    }
                )
                manager = _RoleModelManager(
                    {
                        "reasoner": "Ranked ideas:\n1. target.py::parse_item - writable.",
                        "spec_synth": final_spec,
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {
                            "reasoner": "reasoner-model",
                            "spec_synth": "spec-model",
                            "default": "spec-model",
                        },
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_two_call_synthesis": True,
                            "spec_quality_gate": True,
                            "spec_grounding_gate": True,
                            "writable_files": ["target.py"],
                            "test_commands": ["python -m pytest"],
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = manager
                agent.state.plan_markdown = "Plan"
                agent.state.file_context = [
                    FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
                ]

                await agent._maybe_refresh_run_spec(force=True)

                report = json.loads(
                    (repo / ".local_micro_agent" / "spec_quality_report.json").read_text()
                )
                spec = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
                self.assertEqual(report["status"], "pass")
                self.assertEqual(spec["task_graph"][0]["target_regions"], ["target.py::parse_item"])

        asyncio.run(run_case())

    def test_spec_quality_gate_allows_missing_idea_brief(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
                final_spec = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "quality-edge",
                        "invariants": ["parse_item return contract stays unchanged"],
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "guard parse branch",
                                "strategy_axis": "general_edit",
                                "expected_signal": "pytest stays green",
                                "deliverables": ["target.py"],
                                "target_symbols": ["parse_item"],
                                "target_regions": ["target.py::parse_item"],
                                "preserved_invariants": ["return value remains unchanged"],
                                "edit_scope": "Add one guarded branch inside parse_item.",
                                "risk_level": "local",
                                "tactic_stage": "local_edit",
                                "risk_evidence": {
                                    "field": "edit_scope",
                                    "quote": "one guarded branch",
                                    "explanation": "single guarded local edit",
                                },
                                "validator": {
                                    "kind": "command",
                                    "failure_condition": "pytest fails",
                                },
                                "correctness_rationale": "The original return path is preserved.",
                                "fallback_plan": "Revert the guarded branch.",
                                "acceptance": {
                                    "kind": "command",
                                    "commands": ["python -m pytest"],
                                },
                            }
                        ],
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {
                            "reasoner": "reasoner-model",
                            "spec_synth": "spec-model",
                            "default": "spec-model",
                        },
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_two_call_synthesis": True,
                            "spec_quality_gate": True,
                            "spec_grounding_gate": True,
                            "writable_files": ["target.py"],
                            "test_commands": ["python -m pytest"],
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = _SpecIdeaReasoningOnlyManager(final_spec)
                agent.state.plan_markdown = "Plan"
                agent.state.file_context = [
                    FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
                ]

                await agent._maybe_refresh_run_spec(force=True)

                report = json.loads(
                    (repo / ".local_micro_agent" / "spec_quality_report.json").read_text()
                )
                self.assertEqual(report["status"], "pass")
                self.assertFalse((repo / ".local_micro_agent" / "spec_idea.md").exists())

        asyncio.run(run_case())

    def test_spec_quality_gate_ignores_failed_design_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_quality_gate": True,
                        "spec_grounding_gate": True,
                        "writable_files": ["target.py"],
                        "test_commands": ["python -m pytest"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.file_context = [
                FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
            ]
            spec = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "quality-failed-task-edge",
                    "invariants": ["parse_item return contract stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-000",
                            "title": "failed broad rewrite",
                            "status": "failed_design",
                            "deliverables": ["target.py", "other.py"],
                            "target_regions": [],
                            "edit_scope": "Optimize the codebase.",
                            "fallback_plan": "pass",
                        },
                        {
                            "task_id": "task-001",
                            "title": "guard parse branch",
                            "strategy_axis": "general_edit",
                            "expected_signal": "pytest stays green",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["return value remains unchanged"],
                            "edit_scope": "Add one guarded branch inside parse_item.",
                            "risk_level": "local",
                            "tactic_stage": "local_edit",
                            "risk_evidence": {
                                "field": "edit_scope",
                                "quote": "one guarded branch",
                                "explanation": "single guarded local edit",
                            },
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The original return path is preserved.",
                            "fallback_plan": "Revert the guarded branch.",
                            "acceptance": {
                                "kind": "command",
                                "commands": ["python -m pytest"],
                            },
                        },
                    ],
                }
            )

            report = agent._spec_quality_report(spec)

            self.assertEqual(report["status"], "pass")

    def test_spec_quality_vague_edit_scope_hint_is_repairable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("def build(value):\n    return value\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_quality_gate": True,
                        "writable_files": ["target.py"],
                        "test_commands": ["python -m pytest"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.file_context = [
                FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
            ]
            spec = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "quality-vague-scope",
                    "invariants": ["build return contract stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "pack build work",
                            "strategy_axis": "general_edit",
                            "expected_signal": "pytest stays green",
                            "deliverables": ["target.py"],
                            "read_hints": ["target.py"],
                            "target_symbols": ["build"],
                            "target_regions": ["target.py::build"],
                            "preserved_invariants": ["return value remains unchanged"],
                            "edit_scope": (
                                "Refactor build() to group slots by engine and pack "
                                "them into fewer instructions"
                            ),
                            "risk_level": "local",
                            "tactic_stage": "local_edit",
                            "risk_evidence": {
                                "field": "edit_scope",
                                "quote": "group slots",
                                "explanation": "single target local edit",
                            },
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The original behavior is preserved.",
                            "fallback_plan": "Revert the guarded branch.",
                            "acceptance": {
                                "kind": "command",
                                "commands": ["python -m pytest"],
                            },
                        }
                    ],
                }
            )

            report = agent._spec_quality_report(spec)

            issue = next(
                issue for issue in report["issues"] if issue["code"] == "vague_edit_scope"
            )
            self.assertIn("Bad:", issue["rewrite_hint"])
            self.assertIn("Good:", issue["rewrite_hint"])
            self.assertIn("target.py::build", issue["rewrite_hint"])
            self.assertIn("name the one category", issue["rewrite_hint"])

    def test_spec_quality_soft_fallback_persists_last_spec_before_code(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "target.py").write_text("def build(value):\n    return value\n")
                bad_spec = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "soft-quality",
                        "invariants": ["build behavior stays unchanged"],
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "pack build",
                                "strategy_axis": "general_edit",
                                "expected_signal": "pytest stays green",
                                "deliverables": ["target.py"],
                                "read_hints": ["target.py"],
                                "target_symbols": ["build"],
                                "target_regions": ["target.py::build"],
                                "preserved_invariants": ["return value remains unchanged"],
                                "edit_scope": "Refactor build.",
                                "risk_level": "local",
                                "tactic_stage": "local_edit",
                                "risk_evidence": {
                                    "field": "edit_scope",
                                    "quote": "Refactor build",
                                    "explanation": "single target local edit",
                                },
                                "validator": {
                                    "kind": "command",
                                    "failure_condition": "pytest fails",
                                },
                                "correctness_rationale": "The old behavior remains.",
                                "fallback_plan": "Revert the change.",
                                "acceptance": {
                                    "kind": "command",
                                    "commands": ["python -m pytest"],
                                },
                            }
                        ],
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {
                            "spec_synth": "spec-model",
                            "default": "spec-model",
                        },
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "run_spec_model_role": "spec_synth",
                            "spec_quality_gate": True,
                            "spec_quality_rewrite_attempts": 0,
                            "spec_gate_soft_fallback": True,
                            "writable_files": ["target.py"],
                            "test_commands": ["python -m pytest"],
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = _RoleModelManager({"spec_synth": bad_spec})
                agent.state.plan_markdown = "Plan"
                agent.state.file_context = [
                    FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
                ]

                await agent._maybe_refresh_run_spec(force=True)

                spec = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
                progress = (repo / ".local_micro_agent" / "spec_progress.jsonl").read_text()
                self.assertEqual(spec["quality_gate_advisory"]["status"], "soft_fallback")
                self.assertIn("quality_soft_fallback", progress)

        asyncio.run(run_case())

    def test_spec_quality_soft_fallback_blocks_unknown_hypothesis_id(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
                manager = _RoleModelManager(
                    {
                        "reasoner": ModelResponse(
                            "",
                            usage={"reasoning_only_response": True},
                            reasoning=self._valid_hypothesis_brief(),
                        ),
                        "spec_synth": self._valid_hypothesis_spec(
                            spec_id="unknown-id-soft-fallback",
                            hypothesis_id="hyp-invented",
                        ),
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {
                            "reasoner": "reasoner-model",
                            "spec_synth": "spec-model",
                            "default": "spec-model",
                        },
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_two_call_synthesis": True,
                            "spec_thinking_brief_enabled": True,
                            "spec_hypothesis_brief_enabled": True,
                            "spec_finalize_model_role": "spec_synth",
                            "spec_quality_gate": True,
                            "spec_quality_rewrite_attempts": 0,
                            "spec_gate_soft_fallback": True,
                            "spec_grounding_gate": True,
                            "writable_files": ["target.py"],
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = manager
                agent.state.plan_markdown = "Plan"
                agent.state.file_context = [
                    FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
                ]

                await agent._maybe_refresh_run_spec(force=True)

                self.assertFalse((repo / ".local_micro_agent" / "run_spec.json").exists())
                report = json.loads(
                    (repo / ".local_micro_agent" / "spec_quality_report.json").read_text()
                )
                self.assertIn("hypothesis_id_unknown", report["issue_codes"])
                progress_path = repo / ".local_micro_agent" / "spec_progress.jsonl"
                progress = progress_path.read_text() if progress_path.exists() else ""
                self.assertNotIn("quality_soft_fallback", progress)
                self.assertIn(
                    "soft fallback blocked",
                    "\n".join(agent.state.notes),
                )

        asyncio.run(run_case())

    def test_spec_synth_call_budget_bounds_quality_rewrites(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "target.py").write_text("def build(value):\n    return value\n")
                bad_spec = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "budgeted-quality",
                        "invariants": ["build behavior stays unchanged"],
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "pack build",
                                "strategy_axis": "general_edit",
                                "expected_signal": "pytest stays green",
                                "deliverables": ["target.py"],
                                "target_symbols": ["build"],
                                "target_regions": ["target.py::build"],
                                "preserved_invariants": ["return value remains unchanged"],
                                "edit_scope": "Refactor build.",
                                "risk_level": "local",
                                "tactic_stage": "local_edit",
                                "risk_evidence": {
                                    "field": "edit_scope",
                                    "quote": "Refactor build",
                                    "explanation": "single target local edit",
                                },
                                "validator": {
                                    "kind": "command",
                                    "failure_condition": "pytest fails",
                                },
                                "correctness_rationale": "The old behavior remains.",
                                "fallback_plan": "Revert the change.",
                                "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                            }
                        ],
                    }
                )
                manager = _RoleModelManager({"reasoner": "unused", "spec_synth": bad_spec})
                agent = MicroAgent(
                    {
                        "models": {
                            "reasoner": "reasoner-model",
                            "spec_synth": "spec-model",
                            "default": "spec-model",
                        },
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "spec_two_call_synthesis": True,
                            "spec_quality_gate": True,
                            "spec_quality_rewrite_attempts": 2,
                            "spec_synth_call_budget": 1,
                            "writable_files": ["target.py"],
                            "test_commands": ["python -m pytest"],
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = manager
                agent.state.plan_markdown = "Plan"
                agent.state.file_context = [
                    FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
                ]

                await agent._maybe_refresh_run_spec(force=True)

                self.assertEqual(agent.state.scratch["spec_synth_call_count"], 1)
                self.assertTrue(agent.state.scratch["spec_synth_budget_exhausted"])
                self.assertEqual(len(manager.seen.get("spec_synth", [])), 1)
                self.assertNotIn("reasoner", manager.seen)
                self.assertFalse((repo / ".local_micro_agent" / "run_spec.json").exists())

        asyncio.run(run_case())

    def test_spec_quality_gate_accepts_valid_structural_probe_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_quality_gate": True,
                        "spec_grounding_gate": True,
                        "spec_design_contract_gate": True,
                        "spec_structural_risk_gate": True,
                        "spec_probe_diff_contract_required": True,
                        "writable_files": ["target.py"],
                        "test_commands": ["python -m pytest"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.file_context = [
                FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
            ]
            spec = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "quality-structural-normal",
                    "invariants": ["parse_item return contract stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "Probe parser branch scheduling",
                            "strategy_axis": "general_edit",
                            "expected_signal": "pytest stays green",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["existing parse outputs stay unchanged"],
                            "edit_scope": "Add one guarded branch before the parser loop.",
                            "risk_level": "structural",
                            "tactic_stage": "structural_probe",
                            "risk_evidence": {
                                "field": "title",
                                "quote": "scheduling",
                                "explanation": "Scheduling is a structural behavior risk.",
                            },
                            "probe_plan": "Add a single guarded branch and keep the old path.",
                            "probe_diff_contract": {
                                "allowed_files": ["target.py"],
                                "allowed_regions": ["target.py::parse_item"],
                                "expected_changed_regions": ["target.py::parse_item"],
                                "target_symbols": ["parse_item"],
                                "max_files_changed": 1,
                                "max_hunks": 1,
                                "max_changed_lines": 12,
                                "max_changed_functions": 1,
                                "allowed_change_kinds": ["add_guard"],
                                "observation": "pytest remains green",
                            },
                            "invariant_evidence": ["The old path remains the fallback path."],
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The fallback parser path is unchanged.",
                            "fallback_plan": "Disable the guarded branch.",
                            "rollback_or_shrink_plan": (
                                "Shrink to a smaller guarded probe or revert the branch."
                            ),
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            report = agent._spec_quality_report(spec)

            self.assertEqual(report["status"], "pass")

    def test_spec_quality_gate_rejects_design_contract_region_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text(
                "def parse_item(value):\n    return value\n\n"
                "def format_item(value):\n    return str(value)\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_quality_gate": True,
                        "spec_grounding_gate": True,
                        "spec_design_contract_gate": True,
                        "spec_structural_risk_gate": True,
                        "writable_files": ["target.py"],
                        "test_commands": ["python -m pytest"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.file_context = [
                FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
            ]
            spec = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "quality-design-preflight",
                    "invariants": ["public behavior stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "guard parse branch",
                            "strategy_axis": "general_edit",
                            "expected_signal": "pytest stays green",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["return value remains unchanged"],
                            "edit_scope": "Add one guarded branch inside parse_item.",
                            "risk_level": "local",
                            "tactic_stage": "local_edit",
                            "risk_evidence": {
                                "field": "edit_scope",
                                "quote": "one guarded branch",
                                "explanation": "single guarded local edit",
                            },
                            "probe_diff_contract": {
                                "allowed_files": ["target.py"],
                                "allowed_regions": ["target.py::parse_item"],
                                "expected_changed_regions": ["target.py::format_item"],
                            },
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The original return path is preserved.",
                            "fallback_plan": "Revert the guarded branch.",
                            "acceptance": {
                                "kind": "command",
                                "commands": ["python -m pytest"],
                            },
                        }
                    ],
                }
            )

            report = agent._spec_quality_report(spec)

            self.assertEqual(report["status"], "fail")
            self.assertIn(
                "design_contract_probe_contract_region_mismatch",
                report["issue_codes"],
            )
            self.assertIn(
                "target=parse_item with expected [parse_item]",
                report["issues"][-1]["rewrite_hint"],
            )

    def test_spec_quality_gate_rejects_rollback_only_structural_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_quality_gate": True,
                        "spec_grounding_gate": True,
                        "spec_design_contract_gate": True,
                        "spec_structural_risk_gate": True,
                        "spec_probe_diff_contract_required": True,
                        "writable_files": ["target.py"],
                        "test_commands": ["python -m pytest"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.file_context = [
                FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
            ]
            spec = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "quality-rollback-only",
                    "invariants": ["parse_item return contract stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "Probe parser branch scheduling",
                            "strategy_axis": "general_edit",
                            "expected_signal": "pytest stays green",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["existing parse outputs stay unchanged"],
                            "edit_scope": "Add one guarded branch before the parser loop.",
                            "risk_level": "structural",
                            "tactic_stage": "structural_probe",
                            "risk_evidence": {
                                "field": "title",
                                "quote": "scheduling",
                                "explanation": "Scheduling is a structural behavior risk.",
                            },
                            "probe_plan": "Add a single guarded branch and keep the old path.",
                            "probe_diff_contract": {
                                "allowed_files": ["target.py"],
                                "allowed_regions": ["target.py::parse_item"],
                                "expected_changed_regions": ["target.py::parse_item"],
                                "target_symbols": ["parse_item"],
                                "max_files_changed": 1,
                                "max_hunks": 1,
                                "max_changed_lines": 12,
                                "max_changed_functions": 1,
                                "allowed_change_kinds": ["add_guard"],
                                "observation": "pytest remains green",
                            },
                            "invariant_evidence": ["The old path remains the fallback path."],
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The fallback parser path is unchanged.",
                            "fallback_plan": "Disable the guarded branch.",
                            "rollback_or_shrink_plan": "Revert the single patch.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            report = agent._spec_quality_report(spec)

            self.assertEqual(report["status"], "fail")
            self.assertIn(
                "design_contract_rollback_or_shrink_plan_must_describe_a_smaller_guarded_probe",
                report["issue_codes"],
            )

    def test_spec_quality_feedback_includes_hypothesis_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_hypothesis_brief_enabled": True,
                        "spec_hypothesis_options_path": ".local_micro_agent/spec_hypothesis_options.json",
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            options = {
                "version": 1,
                "accepted": [],
                "rejected": [
                    {
                        "hypothesis_id": "broad-pack",
                        "issues": ["missing_why_not_smaller"],
                        "option": {
                            "hypothesis": "broad packing",
                            "change_boundary": {
                                "regions": ["target.py::Builder.build"],
                                "kind": "structural_probe",
                            },
                            "expected_signal": {"success_condition": ""},
                            "why_not_smaller": "",
                        },
                    }
                ],
            }
            (artifact_dir / "spec_hypothesis_options.json").write_text(
                json.dumps(options) + "\n"
            )
            report = {
                "status": "fail",
                "issues": [
                    {
                        "code": "hypothesis_option_missing",
                        "task_id": "task-001",
                        "detail": "no accepted option",
                        "rewrite_hint": "produce typed option",
                    }
                ],
            }

            feedback = agent._spec_quality_feedback_context(report)

            self.assertIn("BEGIN_HYPOTHESIS_OPTION", feedback)
            self.assertIn("Free-form prose ideas and rejected options are not runnable", feedback)
            self.assertIn("Latest SPEC hypothesis option validation summary", feedback)
            self.assertIn("missing_why_not_smaller", feedback)
            self.assertIn("broad-pack", feedback)

    def test_spec_quality_feedback_explains_smaller_guarded_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=Path(tmp), user_request="test"),
            )
            report = {
                "status": "fail",
                "issues": [
                    {
                        "code": "design_contract_rollback_or_shrink_plan_must_describe_a_smaller_guarded_probe",
                        "task_id": "task-001",
                        "detail": "rollback_or_shrink_plan must describe a smaller probe",
                        "rewrite_hint": "describe a smaller guarded probe",
                    }
                ],
            }

            feedback = agent._spec_quality_feedback_context(report)

            self.assertIn("describe the smaller guarded probe itself", feedback)
            self.assertIn("not only how to revert", feedback)

    def test_spec_quality_feedback_explains_hypothesis_id_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=Path(tmp), user_request="test"),
            )
            report = {
                "status": "fail",
                "issues": [
                    {
                        "code": "hypothesis_id_unknown",
                        "task_id": "task-001",
                        "detail": "unknown hypothesis id hyp-invented",
                    }
                ],
            }

            feedback = agent._spec_quality_feedback_context(report)

            self.assertIn("Copy one accepted hypothesis_id exactly", feedback)
            self.assertIn("do not invent", feedback)

    def test_shrink_probe_plan_rejects_rollback_only_single_phrases(self) -> None:
        self.assertFalse(MicroAgent._plan_mentions_shrink_or_probe("Revert the single patch."))
        self.assertFalse(
            MicroAgent._plan_mentions_shrink_or_probe("Restore the single changed branch.")
        )
        self.assertFalse(
            MicroAgent._plan_mentions_shrink_or_probe("Roll back to a single previous branch.")
        )
        self.assertTrue(
            MicroAgent._plan_mentions_shrink_or_probe(
                "Use a single guarded branch and keep fallback behavior."
            )
        )
        self.assertTrue(
            MicroAgent._plan_mentions_shrink_or_probe("Revert if the smaller probe fails.")
        )

    def test_spec_quality_gate_rejects_local_signature_callsite_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text(
                "def parse_item(value):\n    return value\n\n"
                "def format_item(value):\n    return parse_item(value)\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_quality_gate": True,
                        "spec_grounding_gate": True,
                        "spec_design_contract_gate": True,
                        "spec_structural_risk_gate": True,
                        "writable_files": ["target.py"],
                        "test_commands": ["python -m pytest"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.file_context = [
                FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
            ]
            spec = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "quality-local-risk-mismatch",
                    "invariants": ["public behavior stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "reuse parser argument",
                            "strategy_axis": "general_edit",
                            "expected_signal": "pytest stays green",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["return value remains unchanged"],
                            "edit_scope": (
                                "Change parse_item signature and update callsite in "
                                "format_item."
                            ),
                            "risk_level": "local",
                            "tactic_stage": "local_edit",
                            "risk_evidence": {
                                "field": "edit_scope",
                                "quote": "Change parse_item signature",
                                "explanation": "Incorrectly labeled as local.",
                            },
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The original return path is preserved.",
                            "fallback_plan": "Revert the signature change.",
                            "acceptance": {
                                "kind": "command",
                                "commands": ["python -m pytest"],
                            },
                        }
                    ],
                }
            )

            report = agent._spec_quality_report(spec)

            self.assertEqual(report["status"], "fail")
            self.assertIn(
                "design_contract_local_risk_level_contradicts_structural_action_in_task_scope",
                report["issue_codes"],
            )

    def test_spec_quality_gate_rejects_multi_action_structural_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("def parse_item(value):\n    return value\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "spec_mode": True,
                        "spec_quality_gate": True,
                        "spec_grounding_gate": True,
                        "spec_design_contract_gate": True,
                        "spec_structural_risk_gate": True,
                        "spec_probe_diff_contract_required": True,
                        "writable_files": ["target.py"],
                        "test_commands": ["python -m pytest"],
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            agent.state.file_context = [
                FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
            ]
            spec = agent._normalize_run_spec(
                {
                    "version": 2,
                    "spec_id": "quality-broad-structural",
                    "invariants": ["parse_item return contract stays unchanged"],
                    "task_graph": [
                        {
                            "task_id": "task-001",
                            "title": "Probe parser state scheduling",
                            "strategy_axis": "general_edit",
                            "expected_signal": "pytest stays green",
                            "deliverables": ["target.py"],
                            "target_symbols": ["parse_item"],
                            "target_regions": ["target.py::parse_item"],
                            "preserved_invariants": ["existing parse outputs stay unchanged"],
                            "edit_scope": (
                                "Allocate a cached value, initialize it before the loop, "
                                "replace all parser reads, increment the cursor, and "
                                "remove the old calculation."
                            ),
                            "risk_level": "structural",
                            "tactic_stage": "structural_probe",
                            "risk_evidence": {
                                "field": "title",
                                "quote": "scheduling",
                                "explanation": "Scheduling is a structural behavior risk.",
                            },
                            "probe_plan": "Add a single guarded branch and keep the old path.",
                            "probe_diff_contract": {
                                "allowed_files": ["target.py"],
                                "allowed_regions": ["target.py::parse_item"],
                                "expected_changed_regions": ["target.py::parse_item"],
                                "target_symbols": ["parse_item"],
                                "max_files_changed": 1,
                                "max_hunks": 2,
                                "max_changed_lines": 20,
                                "max_changed_functions": 1,
                                "allowed_change_kinds": ["add_guard"],
                                "observation": "pytest remains green",
                            },
                            "invariant_evidence": ["The old path remains the fallback path."],
                            "validator": {
                                "kind": "command",
                                "failure_condition": "pytest fails",
                            },
                            "correctness_rationale": "The fallback parser path is unchanged.",
                            "fallback_plan": "Disable the guarded branch.",
                            "rollback_or_shrink_plan": "Shrink to one guarded parser branch.",
                            "acceptance": {"kind": "command", "commands": ["python -m pytest"]},
                        }
                    ],
                }
            )

            report = agent._spec_quality_report(spec)

            self.assertEqual(report["status"], "fail")
            self.assertIn(
                "design_contract_structural_edit_scope_too_broad_start_with_one_reversible_probe",
                report["issue_codes"],
            )

    def test_spec_quality_gate_retries_after_silent_idea_drift(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "target.py").write_text(
                    "def parse_item(value):\n    return value\n\n"
                    "def format_item(value):\n    return str(value)\n"
                )
                bad_spec = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "quality-retry",
                        "invariants": ["public behavior stays unchanged"],
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "format branch",
                                "strategy_axis": "general_edit",
                                "expected_signal": "pytest stays green",
                                "deliverables": ["target.py"],
                                "target_symbols": ["format_item"],
                                "target_regions": ["target.py::format_item"],
                                "preserved_invariants": ["return value remains unchanged"],
                                "edit_scope": "Add one guarded branch inside format_item.",
                                "risk_level": "local",
                                "tactic_stage": "local_edit",
                                "risk_evidence": {
                                    "field": "edit_scope",
                                    "quote": "one guarded branch",
                                    "explanation": "single guarded local edit",
                                },
                                "validator": {
                                    "kind": "command",
                                    "failure_condition": "pytest fails",
                                },
                                "correctness_rationale": "The original return path is preserved.",
                                "fallback_plan": "Revert the guarded branch.",
                                "acceptance": {
                                    "kind": "command",
                                    "commands": ["python -m pytest"],
                                },
                            }
                        ],
                    }
                )
                fixed_spec = json.dumps(
                    {
                        "version": 2,
                        "spec_id": "quality-retry",
                        "invariants": ["public behavior stays unchanged"],
                        "task_graph": [
                            {
                                "task_id": "task-001",
                                "title": "parse branch",
                                "strategy_axis": "general_edit",
                                "expected_signal": "pytest stays green",
                                "deliverables": ["target.py"],
                                "target_symbols": ["parse_item"],
                                "target_regions": ["target.py::parse_item"],
                                "preserved_invariants": ["return value remains unchanged"],
                                "edit_scope": "Add one guarded branch inside parse_item.",
                                "risk_level": "local",
                                "tactic_stage": "local_edit",
                                "risk_evidence": {
                                    "field": "edit_scope",
                                    "quote": "one guarded branch",
                                    "explanation": "single guarded local edit",
                                },
                                "validator": {
                                    "kind": "command",
                                    "failure_condition": "pytest fails",
                                },
                                "correctness_rationale": "The original return path is preserved.",
                                "fallback_plan": "Revert the guarded branch.",
                                "acceptance": {
                                    "kind": "command",
                                    "commands": ["python -m pytest"],
                                },
                            }
                        ],
                    }
                )
                manager = _RoleSequenceModelManager(
                    {
                        "reasoner": [
                            "Ranked ideas:\n1. target.py::parse_item - writable.",
                        ],
                        "spec_synth": [bad_spec, fixed_spec],
                    }
                )
                agent = MicroAgent(
                    {
                        "models": {
                            "reasoner": "reasoner-model",
                            "spec_synth": "spec-model",
                            "default": "spec-model",
                        },
                        "providers": {},
                        "mcp_servers": {},
                        "workflow": {
                            "spec_mode": True,
                            "run_spec_enabled": True,
                            "run_spec_after_read": True,
                            "run_spec_path": ".local_micro_agent/run_spec.json",
                            "spec_two_call_synthesis": True,
                            "spec_quality_gate": True,
                            "spec_quality_rewrite_attempts": 1,
                            "spec_grounding_gate": True,
                            "writable_files": ["target.py"],
                            "test_commands": ["python -m pytest"],
                        },
                    },
                    AgentState(repo_root=repo, user_request="test"),
                )
                agent.models = manager
                agent.state.plan_markdown = "Plan"
                agent.state.file_context = [
                    FileSnapshot(path="target.py", content=(repo / "target.py").read_text())
                ]

                await agent._maybe_refresh_run_spec(force=True)

                spec = json.loads((repo / ".local_micro_agent" / "run_spec.json").read_text())
                report = json.loads(
                    (repo / ".local_micro_agent" / "spec_quality_report.json").read_text()
                )
                progress = (repo / ".local_micro_agent" / "spec_progress.jsonl").read_text()
                graph_events = [
                    json.loads(line)
                    for line in (
                        repo / ".local_micro_agent" / "spec_graph_candidates.jsonl"
                    ).read_text().splitlines()
                    if line.strip()
                ]
                second_prompt = manager.seen["spec_synth"][1][-1]["content"]
                self.assertEqual(spec["task_graph"][0]["target_regions"], ["target.py::parse_item"])
                self.assertEqual(spec["search"]["graph_id"], "graph-0002")
                self.assertEqual(report["status"], "pass")
                self.assertIn("quality_rejected", progress)
                self.assertIn("idea_alignment_failed", second_prompt)
                self.assertEqual(
                    [(event["event"], event["status"], event["graph_id"]) for event in graph_events],
                    [
                        ("candidate_rejected", "rejected_quality", "graph-0001"),
                        ("candidate_selected", "selected", "graph-0002"),
                    ],
                )
                self.assertIn("idea_alignment_failed", graph_events[0]["issue_codes"])
                self.assertTrue(
                    (repo / ".local_micro_agent" / "spec_graph_candidates" / "graph-0001.json").exists()
                )
                self.assertTrue(
                    (repo / ".local_micro_agent" / "spec_graph_candidates" / "graph-0002.json").exists()
                )

        asyncio.run(run_case())

    def test_spec_graph_signature_preserves_all_graph_targets(self) -> None:
        spec = {
            "version": 2,
            "task_graph": [
                {
                    "task_id": "task-001",
                    "target_regions": ["alpha.py::pack", "beta.py::bundle"],
                    "tactic_stage": "structural_probe",
                },
                {
                    "task_id": "task-002",
                    "deliverables": ["gamma.py", "delta.py"],
                    "risk_level": "local",
                },
            ],
        }

        self.assertEqual(
            MicroAgent._spec_graph_signature(spec),
            [
                "alpha.py::pack:structural_probe",
                "beta.py::bundle:structural_probe",
                "delta.py:local",
                "gamma.py:local",
            ],
        )

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

    def test_reflect_before_retry_skips_structured_patch_miss_until_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            state.notes.append("Replacement target not found: target.py")
            state.scratch["applied_changes"] = 0
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "reflect_before_retry": True,
                    "reflect_after_repeated_failure_class": 3,
                },
            }
            agent = MicroAgent(config, state)

            self.assertFalse(agent._should_reflect_after_failure())
            self.assertFalse(agent._should_reflect_after_failure())
            self.assertTrue(agent._should_reflect_after_failure())
            self.assertFalse(agent._should_reflect_after_failure())
            self.assertIn("Skipping REFLECT for structured retry failure patch_miss", "\n".join(state.notes))

    def test_reflect_conditionally_false_preserves_legacy_reflect_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(repo_root=Path(tmp), user_request="test")
            state.notes.append("Replacement target not found: target.py")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "reflect_before_retry": True,
                    "reflect_conditionally": False,
                },
            }
            agent = MicroAgent(config, state)

            self.assertTrue(agent._should_reflect_after_failure())

    def test_shipped_deterministic_configs_enable_rejected_candidate_retries(self) -> None:
        for config_path in Path("config").glob("*.json"):
            with self.subTest(config=str(config_path)):
                workflow = json.loads(config_path.read_text()).get("workflow", {})
                if workflow.get("deterministic_test_decision"):
                    self.assertTrue(
                        workflow.get("retry_rejected_candidates"),
                        f"{config_path} can stop after one rejected candidate",
                    )

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

    def test_missing_diagnostic_script_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            state = AgentState(repo_root=repo, user_request="test")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "diagnostic_commands": [
                        {
                            "name": "missing-stats",
                            "command": "python3 tools/generated_kernel_stats.py",
                        }
                    ]
                },
            }
            agent = MicroAgent(config, state)

            results = asyncio.run(agent._run_diagnostic_commands())

            self.assertEqual(results[0]["exit_code"], 0)
            self.assertTrue(results[0]["skipped"])
            self.assertIn("missing diagnostic file", results[0]["stdout"])
            self.assertIn("Diagnostic missing-stats skipped", "\n".join(state.notes))

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

    def test_current_source_context_uses_line_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("alpha = 1\nbeta = 2\n")
            state = AgentState(repo_root=repo, user_request="test")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {"writable_files": ["target.py"]},
            }
            agent = MicroAgent(config, state)

            async def format_context() -> str:
                await agent.mcp.start()
                try:
                    return await agent._format_current_source_context()
                finally:
                    await agent.mcp.close()

            context = asyncio.run(format_context())

            self.assertIn("1: alpha = 1", context)
            self.assertIn("2: beta = 2", context)

    def test_symbol_source_context_extracts_named_method_span(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text(
                "class Builder:\n"
                "    def build(self):\n"
                "        return []\n"
                "\n"
                "    def add(self):\n"
                "        return None\n"
            )
            state = AgentState(repo_root=repo, user_request="Replace Builder.build")
            state.plan_markdown = "Implement Builder.build only."
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {"writable_files": ["target.py"]},
            }
            agent = MicroAgent(config, state)

            async def format_context() -> str:
                await agent.mcp.start()
                try:
                    return await agent._format_symbol_source_context()
                finally:
                    await agent.mcp.close()

            context = asyncio.run(format_context())

            self.assertIn("Symbols: Builder.build", context)
            self.assertIn("    def build(self):", context)
            self.assertIn("        return []", context)
            self.assertNotIn("    def add(self):", context)

    def test_target_not_found_repair_context_anchors_near_stale_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text(
                "def unrelated():\n"
                "    return 0\n\n"
                "def hot_path():\n"
                "    value = 'old'\n"
                "    return value\n"
            )
            state = AgentState(repo_root=repo, user_request="test")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["target.py"],
                    "repair_source_context_char_limit": 12000,
                    "repair_anchor_context_lines": 2,
                },
            }
            agent = MicroAgent(config, state)
            candidate = CodeCandidate(
                "miss",
                [
                    CodeChange(
                        path="target.py",
                        reason="edit hot path",
                        target="def hot_path():\n    value = 'missing'\n",
                        replacement="def hot_path():\n    value = 'fast'\n",
                    )
                ],
                "edit hot path",
            )

            async def repair_context() -> str:
                await agent.mcp.start()
                try:
                    return await agent._candidate_repair_source_context(
                        candidate, {"target.py"}
                    )
                finally:
                    await agent.mcp.close()

            context = asyncio.run(repair_context())

            self.assertIn("Original missing target/search text follows", context)
            self.assertIn("Best current-source excerpt", context)
            self.assertIn("4: def hot_path():", context)
            self.assertIn("5:     value = 'old'", context)

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
            self.assertEqual(record["candidate_id"], "miss")
            self.assertEqual(record["status"], "rejected_no_changes")
            self.assertIn("Replacement target not found", record["failure_detail"])
            self.assertIn(
                "target-not-found repair rejected: repaired target still not found",
                "\n".join(state.notes),
            )

    def test_target_not_found_repair_retargets_unique_whitespace_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text(
                "class Builder:\n"
                "    def build(self):\n"
                "        # Simple slot packing\n"
                "        return []\n"
                "\n"
                "    def add(self):\n"
                "        return None\n"
            )
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
                            "print('cycles: 80' if 'return [1]' in t else 'cycles: 120')\""
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
<reason>Repair near-exact whitespace miss.</reason>
<change>
<path>target.py</path>
<search>
    def build(self):
         # Simple slot packing
        return []
</search>
<replace>
    def build(self):
        # Simple slot packing
        return [1]
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
                        target="missing",
                        replacement="new",
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

            self.assertIn("return [1]", target.read_text())
            self.assertIn(
                "Retargeted repaired search block to exact current source whitespace",
                "\n".join(state.notes),
            )
            rows = [
                json.loads(line)
                for line in (repo / ".local_micro_agent" / "candidates.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(rows[0]["candidate_id"], "miss-repair1")
            self.assertEqual(rows[0]["status"], "improved")

    def test_apply_replacement_retargets_unique_whitespace_miss_without_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text(
                "class Builder:\n"
                "    def build(self):\n"
                "        # Simple slot packing\n"
                "        return []\n"
            )
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=repo, user_request="test"),
            )
            change = CodeChange(
                path="target.py",
                reason="replace build method",
                target=(
                    "    def build(self):\n"
                    "         # Simple slot packing\n"
                    "        return []\n"
                ),
                replacement=(
                    "    def build(self):\n"
                    "        # Simple slot packing\n"
                    "        return [1]\n"
                ),
            )

            async def apply_once() -> int:
                await agent.mcp.start()
                try:
                    result = await agent._apply_changes([change], {"target.py"})
                    return result.applied
                finally:
                    await agent.mcp.close()

            applied = asyncio.run(apply_once())

            self.assertEqual(applied, 1)
            self.assertIn("return [1]", target.read_text())
            self.assertIn(
                "Retargeted replacement target to exact current source whitespace",
                "\n".join(agent.state.notes),
            )

    def test_patch_miss_retarget_normal_uses_line_anchor_for_repeated_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text(
                "def first():\n"
                "    value = 1\n"
                "    return value\n"
                "\n"
                "def second():\n"
                "    value = 1\n"
                "    return value\n"
            )
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=repo, user_request="test"),
            )
            change = CodeChange(
                path="target.py",
                reason="edit second only",
                target="    value = 1\n",
                replacement="    value = 2\n",
                target_region="target.py::second",
                start_line=6,
                end_line=6,
                anchor_before="def second():\n",
                anchor_after="    return value\n",
            )

            async def apply_once() -> int:
                await agent.mcp.start()
                try:
                    result = await agent._apply_changes([change], {"target.py"})
                    return result.applied
                finally:
                    await agent.mcp.close()

            applied = asyncio.run(apply_once())

            self.assertEqual(applied, 1)
            self.assertIn("def first():\n    value = 1", target.read_text())
            self.assertIn("def second():\n    value = 2", target.read_text())
            self.assertIn("via line_anchor", "\n".join(agent.state.notes))

    def test_patch_miss_retarget_edge_recovers_from_stale_line_with_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text(
                "def first():\n"
                "    value = 1\n"
                "    return value\n"
                "\n"
                "def second():\n"
                "    value = 1\n"
                "    return value\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {"patch_line_anchor_context_lines": 0},
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            change = CodeChange(
                path="target.py",
                reason="edit second with stale line hint",
                target="    value = 1\n",
                replacement="    value = 3\n",
                target_region="target.py::second",
                start_line=2,
                end_line=2,
                anchor_before="def second():\n",
                anchor_after="    return value\n",
            )

            async def apply_once() -> int:
                await agent.mcp.start()
                try:
                    result = await agent._apply_changes([change], {"target.py"})
                    return result.applied
                finally:
                    await agent.mcp.close()

            applied = asyncio.run(apply_once())

            self.assertEqual(applied, 1)
            self.assertIn("def first():\n    value = 1", target.read_text())
            self.assertIn("def second():\n    value = 3", target.read_text())
            self.assertIn("via anchor", "\n".join(agent.state.notes))

    def test_patch_miss_retarget_error_records_ambiguous_anchor_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text(
                "def target():\n"
                "    value = 1\n"
                "    value = 1\n"
                "    return value\n"
            )
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["target.py"],
                    "candidate_queue": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            agent = MicroAgent(config, state)
            candidate = CodeCandidate(
                "ambiguous",
                [
                    CodeChange(
                        path="target.py",
                        reason="ambiguous bounded edit",
                        target="    value = 1\n",
                        replacement="    value = 2\n",
                        target_region="target.py::target",
                        anchor_before="def target():\n",
                        anchor_after="    return value\n",
                    )
                ],
                "ambiguous bounded edit",
            )

            async def evaluate_once() -> None:
                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([candidate], {"target.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(evaluate_once())

            self.assertEqual(target.read_text().count("value = 1"), 2)
            record = json.loads(
                (repo / ".local_micro_agent" / "candidates.jsonl")
                .read_text()
                .splitlines()[0]
            )
            self.assertEqual(record["failure_class"], "patch_miss")
            self.assertEqual(record["patch_miss_kind"], "ambiguous_target")
            self.assertEqual(record["patch_miss_path"], "target.py")
            self.assertEqual(record["matches_found"], 2)
            self.assertFalse(record["repair_attempted"])
            self.assertEqual(record["target_region"], "target.py::target")

    def test_patch_miss_partial_apply_is_restored_before_candidate_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\nflag = 'old'\n")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["target.py"],
                    "candidate_queue": True,
                    "test_commands": [
                        (
                            "python3 -c \"from pathlib import Path; "
                            "print('cycles: 50' if 'value = \\'new\\'' in "
                            "Path('target.py').read_text() else 'cycles: 120')\""
                        )
                    ],
                    "metric_regex": r"cycles: (\d+)",
                    "baseline_metric": 100,
                    "accept_if_improved": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                    "repair_target_not_found": False,
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            agent = MicroAgent(config, state)
            candidate = CodeCandidate(
                "partial",
                [
                    CodeChange(
                        path="target.py",
                        reason="valid edit",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    ),
                    CodeChange(
                        path="target.py",
                        reason="missing edit",
                        target="flag = 'missing'\n",
                        replacement="flag = 'new'\n",
                    ),
                ],
                "partial should not be tested",
            )

            async def evaluate_once() -> None:
                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([candidate], {"target.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(evaluate_once())

            self.assertEqual(target.read_text(), "value = 'old'\nflag = 'old'\n")
            self.assertIn("partial apply; restored before repair", "\n".join(state.notes))
            rows = [
                json.loads(line)
                for line in (repo / ".local_micro_agent" / "candidates.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["candidate_id"], "partial")
            self.assertEqual(rows[0]["status"], "rejected_no_changes")
            self.assertEqual(rows[0]["failure_class"], "patch_miss")
            self.assertEqual(rows[0]["patch_miss_kind"], "target_not_found")

    def test_patch_miss_partial_patch_failure_is_restored_before_candidate_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\nflag = 'old'\n")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["target.py"],
                    "candidate_queue": True,
                    "test_commands": [
                        (
                            "python3 -c \"from pathlib import Path; "
                            "print('cycles: 50' if 'value = \\'new\\'' in "
                            "Path('target.py').read_text() else 'cycles: 120')\""
                        )
                    ],
                    "metric_regex": r"cycles: (\d+)",
                    "baseline_metric": 100,
                    "accept_if_improved": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                    "repair_target_not_found": False,
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            agent = MicroAgent(config, state)
            candidate = CodeCandidate(
                "partial-patch",
                [
                    CodeChange(
                        path="target.py",
                        reason="valid edit",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    ),
                    CodeChange(
                        path="target.py",
                        reason="stale patch edit",
                        patch=(
                            "diff --git a/target.py b/target.py\n"
                            "--- a/target.py\n"
                            "+++ b/target.py\n"
                            "@@ -1,2 +1,2 @@\n"
                            " value = 'old'\n"
                            "-flag = 'missing'\n"
                            "+flag = 'new'\n"
                        ),
                    ),
                ],
                "partial patch should not be tested",
            )

            async def evaluate_once() -> None:
                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([candidate], {"target.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(evaluate_once())

            self.assertEqual(target.read_text(), "value = 'old'\nflag = 'old'\n")
            self.assertIn("partial apply; restored before repair", "\n".join(state.notes))
            rows = [
                json.loads(line)
                for line in (repo / ".local_micro_agent" / "candidates.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["candidate_id"], "partial-patch")
            self.assertEqual(rows[0]["status"], "rejected_no_changes")
            self.assertEqual(rows[0]["failure_class"], "patch_miss")
            self.assertEqual(rows[0]["patch_miss_kind"], "patch_rejected")

    def test_patch_miss_patch_reject_records_actual_touched_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            forbidden = repo / "forbidden.py"
            target.write_text("value = 'old'\n")
            forbidden.write_text("secret = 'old'\n")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["target.py"],
                    "candidate_queue": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                    "repair_target_not_found": False,
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            agent = MicroAgent(config, state)
            candidate = CodeCandidate(
                "patch-forbidden",
                [
                    CodeChange(
                        path="target.py",
                        reason="valid edit before bad patch",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    ),
                    CodeChange(
                        path="target.py",
                        reason="patch touches a different file",
                        patch=(
                            "diff --git a/forbidden.py b/forbidden.py\n"
                            "--- a/forbidden.py\n"
                            "+++ b/forbidden.py\n"
                            "@@ -1 +1 @@\n"
                            "-secret = 'old'\n"
                            "+secret = 'new'\n"
                        ),
                    ),
                ],
                "patch path should report actual touched file",
            )

            async def evaluate_once() -> None:
                await agent.mcp.start()
                try:
                    await agent._evaluate_code_candidates([candidate], {"target.py"})
                finally:
                    await agent.mcp.close()

            asyncio.run(evaluate_once())

            self.assertEqual(target.read_text(), "value = 'old'\n")
            self.assertEqual(forbidden.read_text(), "secret = 'old'\n")
            rows = [
                json.loads(line)
                for line in (repo / ".local_micro_agent" / "candidates.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["candidate_id"], "patch-forbidden")
            self.assertEqual(rows[0]["failure_class"], "patch_miss")
            self.assertEqual(rows[0]["patch_miss_kind"], "patch_rejected")
            self.assertEqual(rows[0]["patch_miss_path"], "forbidden.py")
            self.assertEqual(rows[0]["patch_touched_files"], ["forbidden.py"])
            self.assertEqual(rows[0]["patch_rejected_files"], ["forbidden.py"])

    def test_single_candidate_partial_apply_is_restored_before_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\nflag = 'old'\n")
            config = {
                "models": {"default": "static"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["target.py"],
                    "test_commands": [
                        (
                            "python3 -c \"from pathlib import Path; "
                            "print('cycles: 50' if 'value = \\'new\\'' in "
                            "Path('target.py').read_text() else 'cycles: 120')\""
                        )
                    ],
                    "deterministic_test_decision": True,
                    "metric_regex": r"cycles: (\d+)",
                    "baseline_metric": 100,
                    "accept_if_improved": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                    "repair_target_not_found": False,
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.planned_files = ["target.py"]
            agent = MicroAgent(config, state)
            agent.models = _SequenceModelManager(
                [
                    json.dumps(
                        {
                            "changes": [
                                {
                                    "path": "target.py",
                                    "reason": "valid edit",
                                    "target": "value = 'old'\n",
                                    "replacement": "value = 'new'\n",
                                },
                                {
                                    "path": "target.py",
                                    "reason": "missing edit",
                                    "target": "flag = 'missing'\n",
                                    "replacement": "flag = 'new'\n",
                                },
                            ]
                        }
                    )
                ]
            )

            async def code_and_test() -> None:
                await agent.mcp.start()
                try:
                    await agent.code()
                    await agent.test()
                finally:
                    await agent.mcp.close()

            asyncio.run(code_and_test())

            self.assertEqual(target.read_text(), "value = 'old'\nflag = 'old'\n")
            self.assertIn(
                "Single CODE candidate had patch miss after partial apply; "
                "restored before repair",
                "\n".join(state.notes),
            )
            rows = [
                json.loads(line)
                for line in (repo / ".local_micro_agent" / "candidates.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "rejected")
            self.assertEqual(rows[0]["applied"], 0)
            self.assertEqual(rows[0]["failure_class"], "patch_miss")
            self.assertEqual(rows[0]["patch_miss_kind"], "target_not_found")

    def test_single_candidate_patch_miss_runs_same_loop_repair(self) -> None:
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
                    "metric_regex": r"cycles: (\d+)",
                    "baseline_metric": 100,
                    "accept_if_improved": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                    "repair_target_not_found": True,
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.planned_files = ["target.py"]
            agent = MicroAgent(config, state)
            agent.models = _SequenceModelManager(
                [
                    json.dumps(
                        {
                            "changes": [
                                {
                                    "path": "target.py",
                                    "reason": "stale edit",
                                    "target": "value = 'missing'\n",
                                    "replacement": "value = 'fast'\n",
                                }
                            ]
                        }
                    ),
                    json.dumps(
                        {
                            "candidates": [
                                {
                                    "id": "fixed",
                                    "reason": "same intent with current target",
                                    "changes": [
                                        {
                                            "path": "target.py",
                                            "reason": "current edit",
                                            "target": "value = 'old'\n",
                                            "replacement": "value = 'fast'\n",
                                        }
                                    ],
                                }
                            ]
                        }
                    ),
                ]
            )

            async def code_and_test() -> None:
                await agent.mcp.start()
                try:
                    await agent.code()
                    await agent.test()
                finally:
                    await agent.mcp.close()

            asyncio.run(code_and_test())

            self.assertEqual(target.read_text(), "value = 'fast'\n")
            self.assertIn(
                "Single CODE candidate target-not-found repair generated",
                "\n".join(state.notes),
            )
            record = json.loads(
                (repo / ".local_micro_agent" / "candidates.jsonl")
                .read_text()
                .splitlines()[0]
            )
            self.assertEqual(record["status"], "improved")
            self.assertEqual(record["repair_parent_id"], "single")
            self.assertTrue(record["repair_attempted"])
            self.assertEqual(record["repair_status"], "applied")
            self.assertEqual(record["repair_parent_patch_miss_kind"], "target_not_found")
            self.assertEqual(record["repair_parent_patch_miss_path"], "target.py")

    def test_apply_replacement_retargeted_noop_is_not_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text(
                "class Builder:\n"
                "    def build(self):\n"
                "        return []\n"
            )
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=repo, user_request="test"),
            )
            change = CodeChange(
                path="target.py",
                reason="near miss noop",
                target="    def build(self):\n         return []\n",
                replacement="    def build(self):\n        return []\n",
            )

            async def apply_once() -> int:
                await agent.mcp.start()
                try:
                    result = await agent._apply_changes([change], {"target.py"})
                    return result.applied
                finally:
                    await agent.mcp.close()

            applied = asyncio.run(apply_once())

            self.assertEqual(applied, 0)
            self.assertIn("no-op after retarget", "\n".join(agent.state.notes))

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

    def test_adaptive_search_memory_cools_down_repeated_failed_region_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text(
                "def hot_path():\n"
                "    value = 'old'\n"
                "    return value\n"
            )
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": takehome_workflow(
                    writable_files=["target.py"],
                    adaptive_search_memory=True,
                    adaptive_search_reject_cooled_axes=True,
                    adaptive_search_axis_failure_threshold=99,
                    adaptive_search_region_failure_threshold=3,
                    adaptive_search_region_cooldown_loops=4,
                ),
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": target.read_text()}
            agent = MicroAgent(config, state)
            candidate = CodeCandidate(
                "hot",
                [
                    CodeChange(
                        path="target.py",
                        reason="hot path tweak",
                        target="    value = 'old'\n",
                        replacement="    value = 'slow'\n",
                    )
                ],
                "hot path tweak",
                strategy_axis="performance",
            )

            for _ in range(3):
                agent._record_strategy_attempt(
                    candidate,
                    status="rejected",
                    metric=120,
                    applied=1,
                    failed=True,
                )

            cooled = agent._cooled_candidate_regions(candidate)
            self.assertEqual(cooled, ["target.py::hot_path::performance"])
            memory = agent._format_adaptive_search_memory()
            self.assertIn("target.py::hot_path::performance", memory)

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

    def test_rejected_candidate_writes_failure_memory_lesson(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": takehome_workflow(
                    adaptive_search_memory=True,
                    failure_memory=True,
                ),
            }
            state = AgentState(repo_root=repo, user_request="test")
            agent = MicroAgent(config, state)
            candidate = CodeCandidate(
                "bad",
                [
                    CodeChange(
                        path="target.py",
                        reason="pack store with address calculation",
                        target="old",
                        replacement="new",
                    )
                ],
                "diagnostic bundle count improves by packing a dependent store",
                strategy_axis="issue_slot_pressure",
            )

            extra = agent._candidate_history_extra(
                candidate,
                status="rejected",
                metric=None,
                applied=1,
                failed=True,
                patch_text="",
                results=[
                    TestResult(
                        "python3 test.py",
                        1,
                        stderr="AssertionError: store read stale address",
                    )
                ],
                failure_detail="AssertionError: store read stale address",
                diagnostic_results=[
                    {
                        "name": "stats",
                        "command": "python3 stats.py",
                        "exit_code": 0,
                        "stdout": '{"non_debug_bundles": 106774}',
                        "stderr": "",
                    }
                ],
            )

            memory_path = repo / ".local_micro_agent" / "failure_memory.jsonl"
            self.assertTrue(memory_path.exists())
            record = json.loads(memory_path.read_text())
            self.assertEqual(record["failure_class"], "correctness_failure")
            self.assertEqual(record["next_rule"], "repair_with_constraint")
            self.assertIn("106774", record["observed_signal"])
            self.assertIn("stale address", record["why_invalid"])
            self.assertEqual(extra["failure_memory_path"], ".local_micro_agent/failure_memory.jsonl")

    def test_candidate_syntax_failure_after_restore_is_not_repair_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": takehome_workflow(
                    adaptive_search_memory=True,
                    failure_memory=True,
                ),
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": "value = 1\n"}
            agent = MicroAgent(config, state)
            candidate = CodeCandidate(
                "syntax-bad",
                [
                    CodeChange(
                        path="target.py",
                        reason="try a local edit",
                        target="value = 1\n",
                        replacement="value =\n",
                    )
                ],
                "try a local edit",
                strategy_axis="general_edit",
            )

            extra = agent._candidate_history_extra(
                candidate,
                status="rejected",
                metric=None,
                applied=1,
                failed=True,
                patch_text="",
                results=[
                    TestResult(
                        "preflight:syntax target.py",
                        1,
                        stderr="SyntaxError in target.py:1:8: invalid syntax",
                    )
                ],
                failure_detail="Candidate preflight failed for target.py: SyntaxError line 1",
            )

            self.assertEqual(extra["failure_origin"], "candidate_validation")
            self.assertEqual(extra["issue_scope"], "candidate_delta")
            self.assertTrue(extra["repo_valid_after_restore"])
            self.assertFalse(extra["repair_task_eligible"])
            self.assertEqual(extra["memory_use"], "avoid_shape")
            record = json.loads((repo / ".local_micro_agent" / "failure_memory.jsonl").read_text())
            self.assertEqual(record["issue_scope"], "candidate_delta")
            self.assertFalse(record["repair_task_eligible"])
            formatted = agent._format_failure_memory()
            self.assertIn("rejected_candidate_lessons", formatted)
            self.assertIn("SyntaxError", formatted)
            self.assertIn('"current_repo_issues": []', formatted)

    def test_post_restore_validation_failure_is_repair_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": takehome_workflow(
                    adaptive_search_memory=True,
                    failure_memory=True,
                ),
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.scratch["pre_code_snapshot"] = {"target.py": "value = 1\n"}
            agent = MicroAgent(config, state)
            candidate = CodeCandidate(
                "restore-bad",
                [],
                "restore validation failed",
                strategy_axis="general_edit",
            )

            extra = agent._candidate_history_extra(
                candidate,
                status="rejected",
                metric=None,
                applied=0,
                failed=True,
                patch_text="",
                results=[TestResult("python3 test.py", 1, stderr="SyntaxError")],
                failure_detail="post-restore validation failed: SyntaxError",
            )

            self.assertEqual(extra["failure_origin"], "post_restore_validation")
            self.assertEqual(extra["issue_scope"], "current_repo")
            self.assertFalse(extra["repo_valid_after_restore"])
            self.assertTrue(extra["repair_task_eligible"])
            self.assertEqual(extra["memory_use"], "create_repair_task")
            formatted = agent._format_failure_memory()
            self.assertIn("current_repo_issues", formatted)
            self.assertIn("create_repair_task", formatted)

    def test_spec_candidate_failure_scope_context_separates_current_repo_issues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "failure_memory.jsonl").write_text(
                json.dumps(
                    {
                        "loop": 1,
                        "strategy_axis": "general_edit",
                        "failure_class": "correctness_failure",
                        "why_invalid": "candidate SyntaxError",
                        "failure_origin": "candidate_validation",
                        "issue_scope": "candidate_delta",
                        "repo_valid_after_restore": True,
                        "repair_task_eligible": False,
                        "memory_use": "avoid_shape",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "loop": 2,
                        "strategy_axis": "general_edit",
                        "failure_class": "correctness_failure",
                        "why_invalid": "post-restore SyntaxError",
                        "failure_origin": "post_restore_validation",
                        "issue_scope": "current_repo",
                        "repo_valid_after_restore": False,
                        "repair_task_eligible": True,
                        "memory_use": "create_repair_task",
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
                        failure_memory=True,
                    ),
                },
                AgentState(repo_root=repo, user_request="test"),
            )

            context = agent._spec_candidate_failure_scope_context()

            self.assertIn("Current repo issues", context)
            self.assertIn("Rejected candidate lessons", context)
            self.assertIn("current_repo_issues", context)
            self.assertIn("post-restore SyntaxError", context)
            self.assertIn("rejected_candidate_lessons", context)
            self.assertIn("candidate SyntaxError", context)
            self.assertIn("do not turn", context)

    def test_spec_rewrite_focus_for_candidate_delta_forbids_repair_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = MicroAgent(
                {"models": {}, "providers": {}, "mcp_servers": {}, "workflow": {}},
                AgentState(repo_root=Path(tmp), user_request="test"),
            )

            focus = agent._spec_design_rewrite_focus(
                {
                    "task_id": "task-001",
                    "title": "Fix syntax",
                    "edit_scope": "Fix syntax in target.py",
                    "last_observation": {
                        "failure_class": "correctness_failure",
                        "issue_scope": "candidate_delta",
                        "repair_task_eligible": False,
                        "summary": "candidate SyntaxError",
                    },
                },
                ["repeated correctness_failure"],
            )

            self.assertIn("do not convert", focus)
            self.assertIn("repair/syntax-fix task", focus)
            self.assertIn("current_repo", focus)

    def test_correct_candidate_persists_last_correct_survivor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("value = 'old'\n")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "preserve_correct_survivors": True,
                    "writable_files": ["target.py"],
                },
            }
            state = AgentState(repo_root=repo, user_request="test")
            agent = MicroAgent(config, state)
            candidate = CodeCandidate(
                "clean",
                [
                    CodeChange(
                        path="target.py",
                        reason="keep tests passing",
                        target="value = 'old'\n",
                        replacement="value = 'new'\n",
                    )
                ],
                "correct but no metric gain",
                strategy_axis="general_edit",
            )

            extra = agent._persist_correct_survivor(
                candidate,
                status="rejected",
                metric=147734,
                patch_text="--- a/target.py\n+++ b/target.py\n@@\n-value = 'old'\n+value = 'new'\n",
                results=[TestResult("python3 test.py", 0, stdout="ok\n")],
                observation={
                    "failure_class": "no_improvement",
                    "stage_result": "no_improvement",
                    "summary": "correctness passed without metric gain",
                },
            )

            state_path = repo / ".local_micro_agent" / "last_correct_state.json"
            patch_path = repo / ".local_micro_agent" / "last_correct.patch"
            self.assertTrue(state_path.exists())
            self.assertTrue(patch_path.exists())
            self.assertEqual(extra["last_correct_patch_path"], ".local_micro_agent/last_correct.patch")
            self.assertEqual(state.scratch["last_correct_metric"], 147734)
            self.assertIn("+value = 'new'", patch_path.read_text())
            record = json.loads(state_path.read_text())
            self.assertEqual(record["candidate_id"], "clean")
            self.assertEqual(record["failure_class"], "no_improvement")

    def test_candidate_history_restores_episode_failure_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            history_dir = repo / ".local_micro_agent"
            history_dir.mkdir()
            rows = []
            for loop in range(3):
                rows.append(
                    {
                        "loop": loop,
                        "candidate_id": f"c{loop}",
                        "status": "rejected",
                        "metric": 147734,
                        "applied": 1,
                        "failed": False,
                        "strategy_axes": ["general_edit"],
                        "family_aliases": ["call_path_probe"],
                        "region_keys": ["target.py::build::general_edit+call_path_probe"],
                        "failure_class": "no_improvement",
                    }
                )
            (history_dir / "candidates.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": takehome_workflow(
                    adaptive_search_memory=True,
                    candidate_history_path=".local_micro_agent/candidates.jsonl",
                    adaptive_search_axis_window=8,
                    adaptive_search_axis_failure_threshold=3,
                    adaptive_search_axis_cooldown_loops=5,
                ),
            }
            state = AgentState(repo_root=repo, user_request="test")
            state.loop_count = 4
            agent = MicroAgent(config, state)

            memory = agent._adaptive_search_memory_from_history()

            self.assertIsNotNone(memory)
            assert memory is not None
            axis = memory["axes"]["general_edit"]
            self.assertEqual(axis["failure_classes"], {"no_improvement": 3})
            self.assertEqual(axis["cooldown_until_loop"], 9)
            self.assertEqual(memory["recent"][-1]["family_aliases"], ["call_path_probe"])
            formatted = agent._format_adaptive_search_memory()
            self.assertIn("no_improvement", formatted)
            self.assertIn("call_path_probe", formatted)

    def test_single_candidate_test_records_history_and_survivor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'new'\n")
            config = {
                "models": {},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "writable_files": ["target.py"],
                    "test_commands": ["python3 -c \"print('CYCLES: 100')\""],
                    "metric_regex": r"CYCLES:\s*(\d+)",
                    "baseline_metric": 50,
                    "require_metric": True,
                    "accept_if_improved": True,
                    "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                    "preserve_correct_survivors": True,
                    "deterministic_test_decision": True,
                },
            }
            state = AgentState(
                repo_root=repo,
                user_request="test",
                current=AgentStateName.TEST,
                max_loops=1,
            )
            state.scratch["pre_code_snapshot"] = {"target.py": "value = 'old'\n"}
            state.scratch["applied_changes"] = 1
            state.proposed_changes = [
                CodeChange(
                    path="target.py",
                    reason="change value",
                    target="value = 'old'\n",
                    replacement="value = 'new'\n",
                )
            ]
            agent = MicroAgent(config, state)

            async def test_once() -> None:
                await agent.mcp.start()
                try:
                    await agent.test()
                finally:
                    await agent.mcp.close()

            asyncio.run(test_once())

            history_path = repo / ".local_micro_agent" / "candidates.jsonl"
            survivor_path = repo / ".local_micro_agent" / "last_correct.patch"
            self.assertTrue(history_path.exists())
            self.assertTrue(survivor_path.exists())
            record = json.loads(history_path.read_text())
            self.assertEqual(record["candidate_id"], "loop-000-single")
            self.assertEqual(record["status"], "rejected")
            self.assertEqual(record["failure_class"], "no_improvement")
            self.assertEqual(record["metric"], 100)
            self.assertEqual(
                record["last_correct_patch_path"],
                ".local_micro_agent/last_correct.patch",
            )
            self.assertIn("+value = 'new'", survivor_path.read_text())
            self.assertEqual(target.read_text(), "value = 'old'\n")

    def test_repeated_candidate_delta_correctness_failure_bans_semantic_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text(
                "class Builder:\n"
                "    def build(self, slots):\n"
                "        return [{engine: [slot]} for engine, slot in slots]\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "adaptive_search_memory": True,
                        "candidate_history_path": ".local_micro_agent/candidates.jsonl",
                        "semantic_failure_family_threshold": 2,
                    },
                },
                AgentState(repo_root=repo, user_request="test", max_loops=10),
            )
            agent.state.scratch["pre_code_snapshot"] = {
                "target.py": (repo / "target.py").read_text()
            }
            result = TestResult(
                "python3 target.py",
                1,
                "",
                "AssertionError: stale slot value at pc=12",
            )

            def failed_candidate(candidate_id: str, replacement: str) -> CodeCandidate:
                return CodeCandidate(
                    candidate_id,
                    [
                        CodeChange(
                            "target.py",
                            "group slots by engine into VLIW bundles to reduce cycles",
                            target="    def build(self, slots):\n",
                            replacement=replacement,
                            target_region="target.py::Builder.build",
                        )
                    ],
                    "group slots by engine into VLIW bundles",
                    "performance",
                )

            first = failed_candidate("first", "    def build(self, slots):\n        return []\n")
            first_extra = agent._candidate_history_extra(
                first,
                status="rejected",
                metric=None,
                applied=1,
                failed=True,
                patch_text="diff",
                results=[result],
                failure_detail=agent._candidate_failure_detail([], [result], failed=True),
            )
            agent._append_candidate_history(first, "rejected", None, 1, True, first_extra)

            agent.state.loop_count = 1
            second = failed_candidate(
                "second",
                "    def build(self, slots):\n        return list(slots)\n",
            )
            second_extra = agent._candidate_history_extra(
                second,
                status="rejected",
                metric=None,
                applied=1,
                failed=True,
                patch_text="diff",
                results=[result],
                failure_detail=agent._candidate_failure_detail([], [result], failed=True),
            )

            self.assertEqual(
                first_extra["semantic_family_key"],
                second_extra["semantic_family_key"],
            )
            self.assertEqual(
                second_extra["semantic_family_action"],
                "ban_family_and_retarget",
            )
            self.assertEqual(
                second_extra["failure_memory_next_rule"],
                "ban_family_and_retarget",
            )
            agent._append_candidate_history(second, "rejected", None, 1, True, second_extra)

            agent.state.loop_count = 2
            third = failed_candidate(
                "third",
                "    def build(self, slots):\n        return tuple(slots)\n",
            )
            rejection = agent._candidate_semantic_family_ban_rejection(third)
            self.assertIsNotNone(rejection)
            assert rejection is not None
            status, note, extra = rejection
            self.assertEqual(status, "rejected_semantic_family_banned")
            self.assertIn("retarget outside", note)
            self.assertEqual(extra["semantic_family_key"], second_extra["semantic_family_key"])

    def test_semantic_family_ban_does_not_block_different_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text(
                "class Builder:\n"
                "    def build(self, slots):\n"
                "        return slots\n"
                "    def other(self, slots):\n"
                "        return slots\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {"adaptive_search_memory": True},
                },
                AgentState(repo_root=repo, user_request="test", loop_count=3),
            )
            banned = CodeCandidate(
                "banned",
                [
                    CodeChange(
                        "target.py",
                        "group slots by engine into VLIW bundles",
                        target_region="target.py::Builder.build",
                    )
                ],
                "group slots by engine into VLIW bundles",
                "performance",
            )
            family = agent._candidate_semantic_family(banned)
            agent.state.scratch["semantic_failure_families"] = {
                family["semantic_family_key"]: {
                    **family,
                    "banned_until_loop": 10,
                    "failures": [{"loop": 1, "candidate_id": "banned"}],
                }
            }
            other = CodeCandidate(
                "other",
                [
                    CodeChange(
                        "target.py",
                        "group slots by engine into VLIW bundles",
                        target_region="target.py::Builder.other",
                    )
                ],
                "group slots by engine into VLIW bundles",
                "performance",
            )

            self.assertIsNone(agent._candidate_semantic_family_ban_rejection(other))

    def test_current_repo_failure_does_not_ban_candidate_delta_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "target.py").write_text("value = 'old'\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {
                        "adaptive_search_memory": True,
                        "semantic_failure_family_threshold": 1,
                    },
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            candidate = CodeCandidate(
                "current-repo",
                [CodeChange("target.py", "group slots by engine", target_region="target.py")],
                "group slots by engine",
                "performance",
            )
            result = TestResult("python3 target.py", 1, "", "SyntaxError: already broken")
            extra = agent._candidate_history_extra(
                candidate,
                status="rejected",
                metric=None,
                applied=0,
                failed=True,
                patch_text="",
                results=[result],
                failure_detail=agent._candidate_failure_detail([], [result], failed=True),
            )

            self.assertEqual(extra["issue_scope"], "current_repo")
            self.assertNotIn("semantic_family_action", extra)
            self.assertIsNone(agent._candidate_semantic_family_ban_rejection(candidate))

    def test_code_prompt_includes_semantic_family_ban_retarget_contract(self) -> None:
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
                loop_count=3,
                max_loops=1,
            )
            state.plan_markdown = "seeded"
            state.planned_files = ["target.py"]
            state.scratch["semantic_failure_families"] = {
                "family123": {
                    "semantic_family_terms": ["region:target.py::value", "term:bundle"],
                    "failure_classes": {"correctness_failure": 2},
                    "banned_until_loop": 8,
                    "failures": [{"loop": 2, "candidate_id": "old"}],
                }
            }
            models = _RoleModelManager(
                {
                    "coder": (
                        '{"changes":[{"path":"target.py","target":"value = '
                        "'old'\\n\",\"replacement\":\"value = 'new'\\n\","
                        '"reason":"try different small probe"}]}'
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
            self.assertIn("Semantic failure-family bans", joined)
            self.assertIn("Candidate-delta correctness failures are negative evidence", joined)
            self.assertIn("Retarget to a smaller metric-bearing probe", joined)

    def test_code_prompt_includes_failure_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            artifact_dir = repo / ".local_micro_agent"
            artifact_dir.mkdir()
            (artifact_dir / "failure_memory.jsonl").write_text(
                json.dumps(
                    {
                        "loop": 7,
                        "strategy_axis": "issue_slot_pressure",
                        "failure_class": "correctness_failure",
                        "failure_signature": ["pack", "store"],
                        "observed_signal": "diagnostics=non_debug_bundles=106774",
                        "why_invalid": "store read an address written in the same bundle",
                        "next_rule": "repair_with_constraint",
                        "repair_hint": "split dependent reads from same-bundle writes",
                    }
                )
                + "\n"
            )
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
                    "failure_memory": True,
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
            self.assertIn("Failure memory follows", joined)
            self.assertIn("repair_with_constraint", joined)
            self.assertIn("same-bundle writes", joined)

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

    def test_brainstorm_selection_can_refresh_read_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("value = 'old'\n")
            history_dir = repo / ".local_micro_agent"
            history_dir.mkdir()
            (history_dir / "candidates.jsonl").write_text(
                '{"status":"rejected","strategy_axis":"issue_slot_pressure","strategy_axes":["issue_slot_pressure"],"reason":"bad bundle"}\n'
                '{"status":"rejected","strategy_axis":"issue_slot_pressure","strategy_axes":["issue_slot_pressure"],"reason":"same hazard"}\n'
            )
            (history_dir / "failure_memory.jsonl").write_text(
                json.dumps(
                    {
                        "loop": 2,
                        "strategy_axis": "issue_slot_pressure",
                        "failure_class": "correctness_failure",
                        "why_invalid": "diagnostic improved but same-bundle read broke tests",
                        "next_rule": "repair_with_constraint",
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
                    brainstorm_refresh_read_after_selection=True,
                    adaptive_search_memory=True,
                    candidate_history_path=".local_micro_agent/candidates.jsonl",
                ),
            }
            state = AgentState(repo_root=repo, user_request="test", current=AgentStateName.REFLECT)
            state.plan_markdown = "seeded"
            state.file_context = [FileSnapshot("target.py", target.read_text())]
            state.scratch["best_metric"] = 110870
            models = _RoleModelManager(
                {
                    "brainstorm": (
                        "1. strategy_axis: issue_slot_pressure\n"
                        "family_key: bundle_dependency_repair\n"
                        "Repair the dependency hazard before trying the lower bundle signal.\n"
                        "2. strategy_axis: data_flow\nother\n"
                        "3. strategy_axis: resource_management\nthird"
                    )
                }
            )
            agent = MicroAgent(config, state)
            agent.models = models

            asyncio.run(agent.reflect())

            self.assertEqual(state.current, AgentStateName.READ)
            focus = state.scratch["focused_read_context"]
            self.assertIn("bundle_dependency_repair", focus)
            self.assertIn("failure_memory", focus)
            self.assertIn("repair_with_constraint", focus)
            self.assertIn("110870", focus)
            self.assertIn("promoted to focused READ/SPEC refresh", "\n".join(state.notes))

    def test_read_prompt_includes_focused_read_context(self) -> None:
        state = AgentState(repo_root=Path("."), user_request="test")
        state.plan_markdown = "seeded"
        state.scratch["focused_read_context"] = "selected tactic and failure memory"

        messages = read_prompt(state)

        self.assertIn("Focused read context", messages[1]["content"])
        self.assertIn("selected tactic and failure memory", messages[1]["content"])

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
            self.assertIn("never include them in target/search/replace text", dynamic_content)
            self.assertIn("value = 'current'", dynamic_content)
            self.assertNotIn("value = 'stale'", dynamic_content)
            self.assertEqual(target.read_text(), "value = 'new'\n")

    def test_code_attempt_includes_exact_symbol_span_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text(
                "class Builder:\n"
                "    def build(self):\n"
                "        value = 'current'\n"
                "        return value\n"
                "\n"
                "    def add(self):\n"
                "        return None\n"
            )
            config = {
                "models": {"default": "roles"},
                "providers": {},
                "mcp_servers": {},
                "workflow": {
                    "plan_markdown": "Replace Builder.build.",
                    "seed_files": ["target.py"],
                    "writable_files": ["target.py"],
                    "prompt_cache_friendly_layout": True,
                },
            }
            state = AgentState(
                repo_root=repo,
                user_request="Optimize Builder.build",
                current=AgentStateName.CODE,
                max_loops=1,
            )
            state.plan_markdown = "Replace Builder.build."
            state.planned_files = ["target.py"]
            state.file_context = [FileSnapshot("target.py", "class Builder:\n    pass\n")]
            models = _RoleModelManager(
                {
                    "coder": (
                        '{"changes":[{"path":"target.py",'
                        '"target":"        value = \'current\'\\n",'
                        '"replacement":"        value = \'new\'\\n",'
                        '"reason":"edit symbol"}]}'
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

            dynamic_content = models.seen["coder"][0][-1]["content"]
            self.assertIn("Exact writable symbol spans follow", dynamic_content)
            self.assertIn("Symbols: Builder.build", dynamic_content)
            self.assertIn("    def build(self):", dynamic_content)
            self.assertIn("        value = 'current'", dynamic_content)
            self.assertEqual(target.read_text().count("value = 'new'"), 1)

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
            history_records = [
                json.loads(line)
                for line in (artifact_dir / "candidates.jsonl").read_text().splitlines()
            ]
            attempts = [
                json.loads(line)
                for line in (artifact_dir / "todo_attempts.jsonl").read_text().splitlines()
            ]
            updated = json.loads((artifact_dir / "todo_plan.json").read_text())
            self.assertEqual(history_records[0]["status"], "rejected_todo_axis_drift")
            self.assertFalse(history_records[0]["budget_counted"])
            self.assertEqual(history_records[0]["failure_class"], "active_task_drift")
            self.assertEqual(history_records[0]["failure_origin"], "pre_apply_contract")
            self.assertEqual(attempts[0]["todo_id"], "todo-001-phase_interleave")
            self.assertFalse(attempts[0]["budget_counted"])
            self.assertEqual(updated["active_todo_id"], "todo-001-phase_interleave")
            self.assertEqual(updated["todos"][0]["status"], "attempted")
            self.assertEqual(updated["todos"][0]["attempts"], 1)
            self.assertEqual(updated["todos"][0]["non_budget_attempts"], 1)
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
            history_records = [
                json.loads(line)
                for line in (artifact_dir / "candidates.jsonl").read_text().splitlines()
            ]
            attempts = [
                json.loads(line)
                for line in (artifact_dir / "todo_attempts.jsonl").read_text().splitlines()
            ]
            updated = json.loads((artifact_dir / "todo_plan.json").read_text())
            self.assertEqual(history_records[0]["status"], "rejected_todo_family_drift")
            self.assertFalse(history_records[0]["budget_counted"])
            self.assertEqual(history_records[0]["failure_class"], "active_task_drift")
            self.assertFalse(attempts[0]["budget_counted"])
            self.assertEqual(updated["active_todo_id"], "todo-001-phase_interleave")
            self.assertEqual(updated["todos"][0]["status"], "attempted")
            self.assertEqual(updated["todos"][0]["attempts"], 1)
            self.assertEqual(updated["todos"][0]["non_budget_attempts"], 1)
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

    def test_candidate_preflight_rejects_escaped_python_entities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("if value &lt; 2:\n    result = value\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {"writable_files": ["target.py"]},
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            candidate = CodeCandidate(
                "escaped",
                [
                    CodeChange(
                        path="target.py",
                        reason="escaped operator",
                        target="if value < 2:\n    result = value\n",
                        replacement="if value &lt; 2:\n    result = value\n",
                    )
                ],
                "escaped operator",
            )

            async def run_preflight() -> list[TestResult]:
                await agent.mcp.start()
                try:
                    return await agent._run_candidate_preflight(candidate, {"target.py"})
                finally:
                    await agent.mcp.close()

            results = asyncio.run(run_preflight())

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].command, "preflight:html-entities target.py")
            self.assertIn("HTML entity '&lt;'", results[0].stderr)

    def test_candidate_preflight_rejects_python_syntax_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text("def build():\n    if True print('bad')\n")
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {"writable_files": ["target.py"]},
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            candidate = CodeCandidate(
                "syntax",
                [
                    CodeChange(
                        path="target.py",
                        reason="bad syntax",
                        target="def build():\n    pass\n",
                        replacement="def build():\n    if True print('bad')\n",
                    )
                ],
                "bad syntax",
            )

            async def run_preflight() -> list[TestResult]:
                await agent.mcp.start()
                try:
                    return await agent._run_candidate_preflight(candidate, {"target.py"})
                finally:
                    await agent.mcp.close()

            results = asyncio.run(run_preflight())

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].command, "preflight:syntax target.py")
            self.assertIn("SyntaxError in target.py", results[0].stderr)
            self.assertIn("if True print('bad')", results[0].stderr)

    def test_exact_context_refresh_is_queued_after_target_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "target.py"
            target.write_text(
                "def build():\n"
                "    value = 1\n"
                "    return value\n"
            )
            agent = MicroAgent(
                {
                    "models": {},
                    "providers": {},
                    "mcp_servers": {},
                    "workflow": {"writable_files": ["target.py"]},
                },
                AgentState(repo_root=repo, user_request="test"),
            )
            candidate = CodeCandidate(
                "miss",
                [
                    CodeChange(
                        path="target.py",
                        reason="retarget from current source",
                        target="def build():\n    value = 2\n    return value\n",
                        replacement="def build():\n    value = 3\n    return value\n",
                    )
                ],
                "retarget from current source",
            )

            async def queue_refresh() -> None:
                await agent.mcp.start()
                try:
                    await agent._record_exact_context_refresh_request(
                        candidate,
                        "Replacement target not found: target.py",
                        {"target.py"},
                    )
                finally:
                    await agent.mcp.close()

            asyncio.run(queue_refresh())

            refresh = agent.state.scratch["exact_context_refresh"]
            self.assertIn("Replacement target not found: target.py", refresh)
            self.assertIn("Best current-source excerpt", refresh)
            self.assertIn("value = 1", refresh)


if __name__ == "__main__":
    unittest.main()
