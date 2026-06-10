from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

import local_micro_agent.mixins as mixins_package
from local_micro_agent.decisions import CodeCandidate
from local_micro_agent.mixins import (
    AdaptiveSearchMixin,
    BrainstormTacticsMixin,
    CandidateRecordsMixin,
    ModelRuntimeMixin,
    PromptContextMixin,
    TelemetryMixin,
    TodoLifecycleMixin,
)
from local_micro_agent.orchestrator import MicroAgent
from local_micro_agent.state import AgentState, CodeChange, TestResult

ALL_MIXINS = (
    TelemetryMixin,
    ModelRuntimeMixin,
    BrainstormTacticsMixin,
    AdaptiveSearchMixin,
    TodoLifecycleMixin,
    PromptContextMixin,
    CandidateRecordsMixin,
)

MIXIN_MODULES = (
    "candidates",
    "context",
    "model_runtime",
    "search_memory",
    "tactics",
    "telemetry",
    "todos",
)


def make_agent(tmp: str, workflow: dict | None = None) -> MicroAgent:
    state = AgentState(repo_root=Path(tmp), user_request="test")
    config = {
        "models": {},
        "providers": {},
        "mcp_servers": {},
        "workflow": workflow or {},
    }
    return MicroAgent(config, state)


class MixinPackageStructureTests(unittest.TestCase):
    def test_each_mixin_module_imports_independently(self) -> None:
        for name in MIXIN_MODULES:
            with self.subTest(module=name):
                module = importlib.import_module(f"local_micro_agent.mixins.{name}")
                self.assertIsNotNone(module)

    def test_package_all_matches_exported_classes(self) -> None:
        exported = {cls.__name__ for cls in ALL_MIXINS}
        self.assertEqual(set(mixins_package.__all__), exported)
        for name in mixins_package.__all__:
            self.assertTrue(hasattr(mixins_package, name))

    def test_micro_agent_composes_all_mixins(self) -> None:
        for mixin in ALL_MIXINS:
            with self.subTest(mixin=mixin.__name__):
                self.assertIn(mixin, MicroAgent.__mro__)

    def test_no_method_collisions_across_mixins(self) -> None:
        seen: dict[str, str] = {}
        for mixin in ALL_MIXINS:
            for name, member in vars(mixin).items():
                if name.startswith("__"):
                    continue
                self.assertNotIn(
                    name,
                    seen,
                    f"{name} defined in both {seen.get(name)} and {mixin.__name__}",
                )
                seen[name] = mixin.__name__

    def test_orchestrator_core_does_not_shadow_mixin_methods(self) -> None:
        mixin_methods = {
            name
            for mixin in ALL_MIXINS
            for name in vars(mixin)
            if not name.startswith("__")
        }
        core_methods = {
            name for name in vars(MicroAgent) if not name.startswith("__")
        }
        self.assertEqual(core_methods & mixin_methods, set())

    def test_mixins_are_stateless(self) -> None:
        for mixin in ALL_MIXINS:
            with self.subTest(mixin=mixin.__name__):
                self.assertNotIn("__init__", vars(mixin))
                for name, member in vars(mixin).items():
                    if name.startswith("__") or callable(member):
                        continue
                    if isinstance(member, (staticmethod, classmethod, property)):
                        continue
                    self.fail(
                        f"{mixin.__name__}.{name} is mutable class state: {member!r}"
                    )


class TelemetryMixinTests(unittest.TestCase):
    def test_safe_stream_label_sanitizes_unsafe_characters(self) -> None:
        self.assertEqual(
            MicroAgent._safe_stream_label("Role: coder / json_call!"),
            "role_coder_json_call",
        )

    def test_safe_stream_label_falls_back_for_empty_input(self) -> None:
        self.assertEqual(MicroAgent._safe_stream_label("  ./- "), "item")


class ModelRuntimeMixinTests(unittest.TestCase):
    def test_input_token_budget_subtracts_max_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            self.assertEqual(
                agent._input_token_budget({"num_ctx": 64000, "max_tokens": 8192}),
                55808,
            )

    def test_input_token_budget_requires_num_ctx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            self.assertIsNone(agent._input_token_budget({"max_tokens": 8192}))
            self.assertIsNone(agent._input_token_budget({"num_ctx": "64000"}))

    def test_input_token_budget_treats_bad_max_tokens_as_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            self.assertEqual(agent._input_token_budget({"num_ctx": 1000}), 1000)
            self.assertEqual(
                agent._input_token_budget({"num_ctx": 1000, "max_tokens": -5}), 1000
            )


class PromptContextMixinTests(unittest.TestCase):
    def test_line_numbered_text_pads_to_widest_line_number(self) -> None:
        text = "\n".join(f"line{i}" for i in range(1, 12))
        numbered = MicroAgent._line_numbered_text(text)
        lines = numbered.splitlines()
        self.assertEqual(lines[0], " 1: line1")
        self.assertEqual(lines[10], "11: line11")

    def test_line_numbered_text_honors_start_line(self) -> None:
        numbered = MicroAgent._line_numbered_text("alpha\nbeta", start_line=41)
        self.assertEqual(numbered.splitlines(), ["41: alpha", "42: beta"])

    def test_slice_text_keeps_head_and_tail_with_marker(self) -> None:
        text = "A" * 600 + "B" * 600
        sliced = MicroAgent._slice_text(text, 200)
        self.assertIn("[...truncated...]", sliced)
        self.assertTrue(sliced.startswith("A"))
        self.assertTrue(sliced.endswith("B"))
        self.assertLess(len(sliced), len(text))

    def test_best_anchor_excerpt_targets_matching_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            content = (
                "def unrelated():\n"
                "    return 0\n\n"
                "def hot_path():\n"
                "    value = compute_value()\n"
                "    return value\n"
            )
            excerpt = agent._best_anchor_excerpt(
                content,
                "value = compute_value()",
                context_lines=1,
                limit=4000,
            )
            self.assertIn("hot_path", excerpt)
            self.assertIn("5:     value = compute_value()", excerpt)
            self.assertNotIn("unrelated", excerpt)

    def test_repo_path_key_normalizes_inside_repo_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            self.assertEqual(agent._repo_path_key("src/../src/a.py"), "src/a.py")
            absolute = str(Path(tmp) / "src" / "a.py")
            self.assertEqual(agent._repo_path_key(absolute), "src/a.py")


class AdaptiveSearchMixinTests(unittest.TestCase):
    def test_python_symbol_at_line_returns_innermost_symbol(self) -> None:
        content = (
            "class Service:\n"
            "    def outer(self):\n"
            "        pass\n"
            "    def inner(self):\n"
            "        return 1\n"
        )
        self.assertEqual(MicroAgent._python_symbol_at_line(content, 5), "inner")

    def test_python_symbol_at_line_handles_syntax_errors(self) -> None:
        self.assertEqual(MicroAgent._python_symbol_at_line("def broken(:", 1), "")

    def test_change_region_key_uses_symbol_then_bucket_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            python_source = "def hot_path():\n    value = 'old'\n"
            agent.state.scratch["pre_code_snapshot"] = {
                "target.py": python_source,
                "data.txt": "plain text\n" * 60,
            }
            symbol_key = agent._change_region_key(
                CodeChange(
                    path="target.py",
                    reason="tweak",
                    target="    value = 'old'\n",
                    replacement="    value = 'new'\n",
                )
            )
            self.assertEqual(symbol_key, "target.py::hot_path")
            bucket_key = agent._change_region_key(
                CodeChange(
                    path="data.txt",
                    reason="tweak",
                    target="plain text\n",
                    replacement="other text\n",
                )
            )
            self.assertEqual(bucket_key, "data.txt::lines_1_50")

    def test_failure_statuses_cover_gate_rejections(self) -> None:
        statuses = MicroAgent._adaptive_search_failure_statuses()
        for status in (
            "rejected",
            "rejected_cooled_axis",
            "rejected_cooled_region",
            "rejected_no_changes",
            "rejected_repeated_pattern",
        ):
            self.assertIn(status, statuses)


class TodoLifecycleMixinTests(unittest.TestCase):
    def test_todo_status_validated_for_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            self.assertEqual(
                agent._todo_status_after_attempt({"status": "improved"}, None, 1),
                "validated",
            )

    def test_todo_status_failed_when_budget_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp, workflow={"todo_attempt_budget": 2})
            attempt = {"status": "rejected", "failure_class": "no_improvement"}
            self.assertEqual(
                agent._todo_status_after_attempt(attempt, "attempted", 1), "attempted"
            )
            self.assertEqual(
                agent._todo_status_after_attempt(attempt, "attempted", 2), "failed"
            )

    def test_todo_status_keeps_validated_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            attempt = {"status": "rejected", "failure_class": "no_improvement"}
            self.assertEqual(
                agent._todo_status_after_attempt(attempt, "validated", 5), "validated"
            )


class BrainstormTacticsMixinTests(unittest.TestCase):
    def test_signature_similarity_bounds(self) -> None:
        signature = MicroAgent._tactic_signature(
            "pack independent slots in the build method"
        )
        self.assertGreater(len(signature), 0)
        self.assertEqual(MicroAgent._signature_similarity(signature, signature), 1.0)
        other = MicroAgent._tactic_signature("vectorize the parser hash loop")
        self.assertLess(MicroAgent._signature_similarity(signature, other), 1.0)


class CandidateRecordsMixinTests(unittest.TestCase):
    def test_failure_class_patch_miss_for_no_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            self.assertEqual(
                agent._candidate_failure_class(
                    status="rejected_no_changes",
                    metric=None,
                    applied=0,
                    failed=True,
                    results=[],
                    failure_detail="Replacement target not found",
                ),
                "patch_miss",
            )

    def test_failure_class_duplicate_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            self.assertEqual(
                agent._candidate_failure_class(
                    status="rejected_repeated_pattern",
                    metric=None,
                    applied=0,
                    failed=True,
                    results=[],
                ),
                "duplicate_variant",
            )

    def test_failure_class_correctness_failure_on_failing_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            results = [TestResult(command="pytest", exit_code=1, stdout="", stderr="boom")]
            self.assertEqual(
                agent._candidate_failure_class(
                    status="rejected",
                    metric=None,
                    applied=1,
                    failed=True,
                    results=results,
                ),
                "correctness_failure",
            )

    def test_summarize_changes_reports_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            candidate = CodeCandidate(
                "c1",
                [
                    CodeChange(
                        path="src/a.py",
                        reason="edit",
                        target="x",
                        replacement="y",
                    )
                ],
                "edit",
            )
            summary = agent._summarize_changes(candidate.changes)
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary[0].get("path"), "src/a.py")


if __name__ == "__main__":
    unittest.main()
