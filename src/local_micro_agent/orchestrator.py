from __future__ import annotations

import argparse
import ast
import asyncio
import difflib
import hashlib
import json
import re
import shlex
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .candidate_output import candidate_output_message
from .decisions import CodeCandidate, CodeDecision, ReadDecision, TestDecision
from .mcp_client import McpServerSpec, McpToolClient
from .models import ModelManager
from .prompts import (
    PROMPT_MARKDOWN,
    code_prompt,
    plan_prompt,
    read_prompt,
    reflect_prompt,
    simple_thinking_brief_prompt,
    test_prompt,
)
from .state import (
    AgentState,
    AgentStateName,
    CodeChange,
    FileSnapshot,
    TestResult,
)
from .validators import JsonValidationError
from .presets import apply_workflow_preset
from .mixins import (
    AdaptiveSearchMixin,
    BrainstormTacticsMixin,
    CandidateRecordsMixin,
    ModelRuntimeMixin,
    PromptContextMixin,
    TelemetryMixin,
    TodoLifecycleMixin,
)


@dataclass
class ApplyResult:
    applied: int = 0
    failed_changes: list[dict[str, Any]] = field(default_factory=list)
    patch_miss_events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        return bool(self.failed_changes)


@dataclass
class PatchApplyResult:
    applied: bool = False
    touched_files: set[str] = field(default_factory=set)
    rejected_files: list[str] = field(default_factory=list)
    failure_detail: str = ""


class MicroAgent(
    TelemetryMixin,
    ModelRuntimeMixin,
    BrainstormTacticsMixin,
    AdaptiveSearchMixin,
    TodoLifecycleMixin,
    PromptContextMixin,
    CandidateRecordsMixin,
):
    def __init__(self, config: dict[str, Any], state: AgentState):
        self.config = apply_workflow_preset(config)
        workflow = self.config.get("workflow")
        if isinstance(workflow, dict):
            defaulted = workflow.get("preset_defaulted_keys")
            preset_loops = workflow.get("max_code_test_loops")
            if (
                isinstance(defaulted, list)
                and "max_code_test_loops" in defaulted
                and isinstance(preset_loops, int)
                and state.max_loops_defaulted
            ):
                # The loop budget came from the preset, not the caller, even
                # if the config was already expanded by load_config(). A state
                # whose max_loops was omitted could not have seen it; an
                # explicitly supplied max_loops stays authoritative even when
                # it equals DEFAULT_MAX_LOOPS.
                state.max_loops = preset_loops
                state.max_loops_defaulted = False
        self.state = state
        self.models = ModelManager(self.config)
        self.mcp = McpToolClient(
            {
                name: McpServerSpec(command=spec["command"], args=spec.get("args", []))
                for name, spec in self.config.get("mcp_servers", {}).items()
            }
        )

    async def run(self) -> AgentState:
        await self.mcp.start()
        try:
            while self.state.current not in {AgentStateName.DONE, AgentStateName.FAILED}:
                self.state.fsm_step_count += 1
                if self.state.current == AgentStateName.PLAN:
                    self._log("PLAN")
                    await self._profiled_phase("PLAN", self.plan)
                elif self.state.current == AgentStateName.SPEC_SYNTH:
                    self._log("SPEC_SYNTH")
                    await self._profiled_phase("SPEC_SYNTH", self.spec_synth)
                elif self.state.current == AgentStateName.SCHEDULE:
                    self._log("SCHEDULE")
                    await self._profiled_phase("SCHEDULE", self.schedule)
                elif self.state.current == AgentStateName.TASK_READ:
                    self._log("TASK_READ")
                    await self._profiled_phase("TASK_READ", self.task_read)
                elif self.state.current == AgentStateName.ACCEPT_SYNTH:
                    self._log("ACCEPT_SYNTH")
                    await self._profiled_phase("ACCEPT_SYNTH", self.accept_synth)
                elif self.state.current == AgentStateName.READ:
                    self._log("READ")
                    await self._profiled_phase("READ", self.read)
                elif self.state.current == AgentStateName.REFLECT:
                    self._log(f"REFLECT loop={self.state.loop_count}")
                    await self._profiled_phase("REFLECT", self.reflect)
                elif self.state.current == AgentStateName.CODE:
                    self._log(f"CODE loop={self.state.loop_count}")
                    await self._profiled_phase("CODE", self.code)
                elif self.state.current == AgentStateName.TEST:
                    self._log(f"TEST loop={self.state.loop_count}")
                    await self._profiled_phase("TEST", self.test)
                else:
                    self.state.current = AgentStateName.FAILED
            self._persist_spec_report()
        finally:
            await self.mcp.close()
        return self.state

    async def plan(self) -> None:
        await self._load_external_contexts()
        seeded_plan = self.config.get("workflow", {}).get("plan_markdown")
        if seeded_plan:
            self.state.plan_markdown = seeded_plan.strip()
            self.state.current = AgentStateName.READ
            return

        project_context = await self._load_project_context()
        workflow_context = self._workflow_plan_context()
        if workflow_context:
            project_context = "\n\n".join(part for part in [project_context, workflow_context] if part)
        try:
            output = await self._model_chat(
                "planner",
                plan_prompt(self.state, project_context),
                call_site="plan",
            )
        except Exception as exc:
            self.state.notes.append(
                f"PLAN model call failed; using fallback plan: {type(exc).__name__}: {exc}"
            )
            output = self._fallback_plan_markdown(project_context)
        self.state.plan_markdown = output.strip()
        self.state.current = AgentStateName.READ

    def _fallback_plan_markdown(self, project_context: str = "") -> str:
        workflow = self.config.get("workflow", {})
        files = workflow.get("read_fallback_files") or workflow.get("writable_files") or []
        file_lines = "\n".join(f"- `{path}`" for path in files) or "- Use READ fallback files."
        tests = workflow.get("test_commands") or []
        test_lines = "\n".join(f"- `{command}`" for command in tests) or "- Run configured tests."
        return (
            "# Fallback Plan\n\n"
            "## Files to read or modify\n"
            f"{file_lines}\n\n"
            "## Ordered implementation steps\n"
            "1. Read the configured source files and current workflow constraints.\n"
            "2. Make the smallest correctness-preserving change requested by the task.\n"
            "3. Validate with the configured test and metric commands.\n\n"
            "## Test commands\n"
            f"{test_lines}\n"
        )

    async def read(self) -> None:
        seeded_files = self.config.get("workflow", {}).get("seed_files")
        if seeded_files is not None:
            decision = ReadDecision(files=seeded_files, reason="seeded by workflow config")
        else:
            try:
                decision = await self._json_call("planner", read_prompt(self.state), ReadDecision)
            except JsonValidationError as exc:
                fallback_files = self._read_fallback_files()
                self.state.notes.append(
                    "READ decision failed; falling back to configured files: "
                    f"{exc}"
                )
                if not fallback_files:
                    self.state.notes.append(
                        "READ fallback file list is empty; CODE will run without source context"
                    )
                decision = ReadDecision(files=fallback_files, reason="fallback after READ failure")
        self.state.planned_files = self._filter_read_files(decision.files)
        self.state.file_context = []
        for rel_path in self.state.planned_files:
            abs_path = self.state.repo_root / rel_path
            content = await self.mcp.read_file(str(abs_path))
            content = self._context_for_file(rel_path, content)
            self.state.file_context.append(FileSnapshot(path=rel_path, content=content))
        await self._load_external_contexts()
        await self._maybe_refresh_semantic_analysis()
        await self._maybe_refresh_simple_thinking_brief()
        await self._maybe_refresh_run_spec()
        self.state.current = (
            AgentStateName.SCHEDULE if self._spec_mode_enabled() else AgentStateName.CODE
        )

    async def _maybe_refresh_simple_thinking_brief(self, focus: str = "") -> None:
        workflow = self.config.get("workflow", {})
        if self._spec_mode_enabled() or not workflow.get("simple_thinking_brief_enabled"):
            return
        if self.state.scratch.get("simple_thinking_brief") and not focus:
            return
        role = str(workflow.get("simple_thinking_brief_model_role") or "reasoner")
        call_site = "simple_thinking_brief" if not focus else "simple_thinking_brief_refresh"
        meta = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "call_site": call_site,
            "status": "pending",
        }
        try:
            parts = await self._model_thinking_brief(
                role,
                simple_thinking_brief_prompt(self.state, focus=focus),
                call_site=call_site,
            )
        except Exception as exc:
            meta.update(
                {
                    "status": "model_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            self._persist_simple_thinking_brief_meta(meta)
            self.state.notes.append(
                f"Simple thinking brief model call failed: {type(exc).__name__}: {exc}"
            )
            return
        selected = str(parts.source or parts.usage.get("thinking_brief_selected_source") or "")
        accept_reasoning = bool(
            workflow.get("simple_thinking_brief_accept_reasoning_only", True)
        )
        if selected == "reasoning" and not accept_reasoning:
            brief = ""
        else:
            brief = parts.content if selected == "content" else parts.reasoning
        brief = self._slice_text(
            brief.strip(),
            int(workflow.get("simple_thinking_brief_char_limit", 3000) or 3000),
        )
        meta.update(
            {
                "status": "ok" if brief else "empty",
                "selected_source": selected,
                "selected_chars": len(brief),
                "usage": parts.usage,
            }
        )
        path = self._workflow_artifact_path(
            "simple_thinking_brief_path",
            ".local_micro_agent/simple_thinking_brief.md",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if brief:
            path.write_text(brief + "\n")
            self.state.scratch["simple_thinking_brief"] = brief
            self.state.notes.append("Simple thinking brief added for CODE context")
        else:
            self.state.scratch.pop("simple_thinking_brief", None)
            if path.exists():
                try:
                    path.unlink()
                except OSError as exc:
                    self.state.notes.append(
                        f"Failed to clear stale simple thinking brief: {type(exc).__name__}: {exc}"
                    )
        self._persist_simple_thinking_brief_meta(meta)

    def _persist_simple_thinking_brief_meta(self, meta: dict[str, Any]) -> None:
        path = self._workflow_artifact_path(
            "simple_thinking_brief_meta_path",
            ".local_micro_agent/simple_thinking_brief_meta.json",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    def _format_simple_thinking_brief_context(self) -> str:
        workflow = self.config.get("workflow", {})
        if self._spec_mode_enabled() or not workflow.get("simple_thinking_brief_enabled"):
            return ""
        brief = self.state.scratch.get("simple_thinking_brief")
        if not isinstance(brief, str) or not brief.strip():
            path = self._workflow_artifact_path(
                "simple_thinking_brief_path",
                ".local_micro_agent/simple_thinking_brief.md",
            )
            if path.exists():
                brief = path.read_text(errors="replace")
            else:
                return ""
        limit = int(workflow.get("simple_thinking_brief_code_char_limit", 1200) or 1200)
        return self._slice_text(brief.strip(), limit)

    async def spec_synth(self) -> None:
        await self._maybe_refresh_run_spec(force=True)
        self.state.current = AgentStateName.SCHEDULE

    async def schedule(self) -> None:
        self._schedule_spec_task()

    async def task_read(self) -> None:
        await self._read_current_spec_task_context()

    async def accept_synth(self) -> None:
        await self._ensure_current_spec_task_acceptance()

    def _read_fallback_files(self) -> list[str]:
        workflow = self.config.get("workflow", {})
        for key in ("read_fallback_files", "writable_files", "project_context_files"):
            value = workflow.get(key)
            if isinstance(value, list) and value:
                return [str(item) for item in value]
        return []

    async def reflect(self) -> None:
        if self._should_brainstorm():
            await self._brainstorm()
            if self._brainstorm_refresh_read_enabled() and self._selected_tactic_for_current_loop():
                self._prepare_brainstorm_refresh_epoch()
                self.state.current = AgentStateName.READ
            else:
                self.state.current = AgentStateName.CODE
            return
        feedback_notes_limit = int(
            self.config.get("workflow", {}).get("feedback_notes_limit", 12)
        )
        try:
            self.state.scratch["todo_observation_chain"] = (
                self._format_todo_observation_chain()
            )
            output = await self._model_chat(
                "reflector",
                reflect_prompt(self.state, feedback_notes_limit),
                call_site="reflect",
            )
        except Exception as exc:
            self.state.notes.append(
                f"Reflect model call failed: {type(exc).__name__}: {exc}"
            )
            self.state.current = AgentStateName.CODE
            return
        reflection = output.strip()
        if reflection:
            self.state.scratch["reflection"] = reflection
            self.state.notes.append("Reflect summary added for next CODE attempt")
        self.state.current = AgentStateName.CODE

    async def code(self) -> None:
        if self._block_active_todo_if_micro_probe_not_executable():
            return
        if self._brainstorm_all_tactics_failed_for_current_loop():
            self.state.notes.append(
                "Skipping CODE because all brainstorm tactics matched failed families"
            )
            self._append_candidate_history(
                CodeCandidate(
                    "brainstorm-all-skipped",
                    [],
                    "All brainstorm tactics matched failed tactic families",
                    "general_edit",
                ),
                status="rejected_brainstorm_all_failed_families",
                metric=None,
                applied=0,
                failed=True,
            )
            self.state.proposed_changes = []
            self.state.scratch["applied_changes"] = 0
            self.state.scratch.pop("brainstorm_all_tactics_failed_loop", None)
            if self.state.loop_count + 1 >= self.state.max_loops:
                self.state.current = AgentStateName.FAILED
            else:
                self.state.loop_count += 1
                self.state.current = AgentStateName.REFLECT
            return
        seeded_changes = self.config.get("workflow", {}).get("seed_changes")
        seeded_change_mode = bool(seeded_changes)
        if seeded_changes:
            decision = CodeDecision(changes=[CodeChange.from_dict(c) for c in seeded_changes])
        else:
            try:
                feedback_notes_limit = int(
                    self.config.get("workflow", {}).get("feedback_notes_limit", 12)
                )
                output_format = str(
                    self.config.get("workflow", {}).get("code_output_format", "json")
                )
                cache_friendly_layout = bool(
                    self.config.get("workflow", {}).get(
                        "prompt_cache_friendly_layout", True
                    )
                )
                messages = code_prompt(
                    self.state,
                    feedback_notes_limit,
                    output_format,
                    cache_friendly_layout=cache_friendly_layout,
                )
                dynamic_suffix_blocks: list[str] = []
                if (
                    cache_friendly_layout
                    and messages
                    and messages[-1].get("role") == "user"
                    and messages[-1]
                    .get("content", "")
                    .startswith("Dynamic context for this CODE attempt:")
                ):
                    dynamic_suffix_blocks.append(messages.pop()["content"])

                def add_runtime_context(content: str) -> None:
                    if cache_friendly_layout:
                        dynamic_suffix_blocks.append(content)
                    else:
                        messages.append({"role": "system", "content": content})

                current_source_context = await self._format_current_source_context()
                if current_source_context:
                    add_runtime_context(
                        "Current writable source context follows. This is reread "
                        "immediately before this CODE attempt and supersedes the "
                        "stable Source files block above when they differ. Copy "
                        "target/search text from this current context. Line numbers "
                        "and the 'N: ' prefix are not part of the file content; never "
                        "include them in target/search/replace text.\n"
                        f"{current_source_context}"
                    )
                simple_thinking_brief = self._format_simple_thinking_brief_context()
                if simple_thinking_brief:
                    add_runtime_context(
                        "Simple thinking brief follows. It is advisory only: prefer "
                        "the smallest writable edit, obey current source context, "
                        "latest failure, validation commands, and deterministic "
                        "apply/test gates. Do not treat this as a run_spec, task "
                        "graph, or structural contract.\n"
                        f"{simple_thinking_brief}"
                    )
                symbol_source_context = await self._format_symbol_source_context()
                if symbol_source_context:
                    add_runtime_context(
                        "Exact writable symbol spans follow. These spans were "
                        "extracted from the current source immediately before CODE "
                        "because the active task names these symbols. Prefer copying "
                        "target/search text verbatim from these unnumbered blocks "
                        "when editing the named symbol; do not include fence lines.\n"
                        f"{symbol_source_context}"
                    )
                exact_refresh = self.state.scratch.pop("exact_context_refresh", "")
                if exact_refresh:
                    add_runtime_context(
                        "Exact context refresh follows because a previous candidate "
                        "missed its target/search block or changed only comments/blank "
                        "lines. The next candidate must copy target/search text from "
                        "the current-source excerpt below, not from stale memory.\n"
                        f"{exact_refresh}"
                    )
                if self.config.get("workflow", {}).get("candidate_queue"):
                    messages = [
                        *messages,
                        candidate_output_message(output_format, mode="queue"),
                    ]
                axis_contract = self._format_axis_contract()
                if axis_contract:
                    add_runtime_context(
                        "Strategy axis contract follows. Candidate output must obey it. "
                        "A candidate with a missing, unknown, cooled, or wrong strategy_axis "
                        "will be rejected before edits or tests.\n"
                        f"{axis_contract}"
                    )
                search_memory = self._format_adaptive_search_memory()
                if search_memory:
                    add_runtime_context(
                        "Adaptive search memory follows. Use it to allocate search budget. "
                        "Do not repeat cooled-down strategy axes unless the user request "
                        "explicitly requires them; prefer under-explored axes and explain "
                        "the chosen axis in the candidate reason.\n"
                        f"{search_memory}"
                    )
                gate_memory = self._format_adaptive_gate_memory()
                if gate_memory:
                    add_runtime_context(
                        "Adaptive gate controller telemetry follows. Use it to "
                        "notice when controller gates may be overblocking useful "
                        "search. If gates are in shadow or soft mode, choose a "
                        "small evidence-producing probe instead of renaming old "
                        "ideas.\n"
                        f"{gate_memory}"
                    )
                semantic_family_bans = self._format_semantic_failure_family_bans()
                if semantic_family_bans:
                    add_runtime_context(
                        "Semantic failure-family bans follow. Candidate-delta "
                        "correctness failures are negative evidence, not current "
                        "repository repair tasks. Do not submit another candidate "
                        "inside an active banned family. Retarget to a smaller "
                        "metric-bearing probe outside the banned family unless a "
                        "current-repo issue or validated checkpoint gives new "
                        "evidence.\n"
                        f"{semantic_family_bans}"
                    )
                failure_memory = self._format_failure_memory()
                if failure_memory:
                    add_runtime_context(
                        "Failure memory follows. Treat it as durable negative "
                        "evidence from prior candidates, grouped by issue scope. "
                        "Only current_repo_issues describe problems that may justify "
                        "repair work in the current source. Rejected candidate lessons "
                        "describe discarded candidate deltas, patch misses, contract "
                        "rejects, or metric gates; do not turn their SyntaxError/test "
                        "text into current-code repair tasks. Do not treat improved "
                        "diagnostics as valid unless tests also passed. Avoid candidate "
                        "variants whose failure_signature, failure_class, or next_rule "
                        "says avoid; when next_rule is repair_with_constraint, address "
                        "why_invalid before optimizing further.\n"
                        f"{failure_memory}"
                    )
                todo_observation_chain = self._format_todo_observation_chain()
                if todo_observation_chain:
                    add_runtime_context(
                        "Observation-backed todo continuation follows. Treat this "
                        "as the current experiment chain, not as background. The next "
                        "candidate should either repair the latest actionable failure "
                        "inside the same hypothesis, or explicitly use the observations "
                        "to choose a narrower measured probe. Do not reset to a generic "
                        "tactic when the chain contains a concrete invariant, no-signal, "
                        "or diagnostic observation.\n"
                        f"{todo_observation_chain}"
                    )
                structural_state = self._format_structural_state_context()
                if structural_state:
                    add_runtime_context(
                        "Validated structural checkpoints follow. These are "
                        "correctness-preserving intermediate patches from this run, "
                        "kept separate from metric best state. If the current tactic "
                        "continues one checkpoint, rebuild the candidate from that "
                        "checkpoint and add only one guarded next step; do not copy "
                        "unrelated checkpoints.\n"
                        f"{structural_state}"
                    )
                tactic_library = self._format_tactic_library()
                if tactic_library:
                    add_runtime_context(
                        "Stagnation brainstorm tactics follow. Prefer one tactic that "
                        "matches the required strategy axis and has not been rejected.\n"
                        f"{tactic_library}"
                    )
                simple_report = self._format_simple_report_context()
                if simple_report:
                    add_runtime_context(
                        "Simple report follows. It is advisory-only compression of "
                        "current candidates.jsonl for this simple classic run. Use it "
                        "to avoid repeated no-improvement shapes and to refresh exact "
                        "source after patch misses; do not treat it as a run_spec, "
                        "task graph, scheduler, hypothesis, structural contract, or "
                        "blocking gate.\n"
                        f"{simple_report}"
                    )
                history = self._format_candidate_history()
                if history:
                    add_runtime_context(
                        "Recent candidate history follows. Avoid repeating rejected changes. "
                        "Preserve ideas that were accepted unless the current plan says otherwise.\n"
                        f"{history}"
                    )
                active_todo = self._format_active_todo()
                if active_todo:
                    add_runtime_context(
                        "Active durable todo follows. This is the current contract "
                        "and supersedes broader run-local spec tasks, candidate "
                        "history, and failure memory. Implement only this todo. "
                        "Candidate strategy_axis must exactly match the todo "
                        "strategy_axis. Candidate reason and change reasons must "
                        "preserve the todo context, stay on its family_key when one "
                        "is present, mention the todo_id, and edit only the named "
                        "target_symbols/target_regions. If this todo is a "
                        "structural_probe, its actual diff is checked after apply "
                        "and before tests against probe_diff_contract; stay inside "
                        "its file, hunk, line, symbol, and forbidden-region limits. "
                        "Todo drift is rejected before edits or tests.\n"
                        f"{active_todo}"
                    )
                if dynamic_suffix_blocks:
                    dynamic_suffix_blocks = self._shrink_dynamic_suffix_blocks(
                        messages,
                        dynamic_suffix_blocks,
                        role="coder",
                    )
                    messages = [
                        *messages,
                        {
                            "role": "user",
                            "content": "\n\n".join(dynamic_suffix_blocks),
                        },
                    ]
                decision = await self._json_call("coder", messages, CodeDecision)
            except JsonValidationError as exc:
                self.state.notes.append(f"Coder output rejected after repair: {exc}")
                decision = CodeDecision(changes=[])
        self.state.proposed_changes = decision.changes
        self.state.scratch["applied_changes"] = 0
        allowed = self._writable_files()
        self.state.scratch["pre_code_snapshot"] = await self._snapshot_files(sorted(allowed))
        if self.config.get("workflow", {}).get("candidate_queue"):
            await self._evaluate_code_candidates(decision.candidates, allowed)
            self.state.current = AgentStateName.TEST
            return
        scope_rejection = self._active_todo_change_scope_rejection(decision.changes)
        if scope_rejection is not None:
            status, note = scope_rejection
            self.state.notes.append(
                f"Single CODE candidate rejected before apply: {note}"
            )
            self.state.scratch["pre_apply_candidate_rejection"] = {
                "status": status,
                "note": note,
            }
            self.state.scratch["applied_changes"] = 0
            self.state.current = AgentStateName.TEST
            return
        semantic_family_rejection = self._candidate_semantic_family_ban_rejection(
            self._single_code_candidate(decision.changes)
        )
        if semantic_family_rejection is not None:
            status, note, extra = semantic_family_rejection
            self.state.notes.append(
                f"Single CODE candidate rejected before apply: {note}"
            )
            self.state.scratch["pre_apply_candidate_rejection"] = {
                "status": status,
                "note": note,
                **extra,
            }
            self.state.scratch["applied_changes"] = 0
            self.state.current = AgentStateName.TEST
            return
        previous_snapshot = self.state.scratch.get("pre_code_snapshot")
        self.state.scratch["patch_miss_repair_attempted"] = False
        self.state.scratch["patch_miss_repair_status"] = "not_attempted"
        self.state.scratch.pop("repair_parent_patch_miss_events", None)
        note_start = len(self.state.notes)
        apply_result = await self._apply_changes(
            decision.changes,
            allowed,
            skip_disallowed=seeded_change_mode,
        )
        applied = apply_result.applied
        parent_patch_miss_events = apply_result.patch_miss_events
        if apply_result.has_failures:
            if applied > 0 and isinstance(previous_snapshot, dict):
                self.state.notes.append(
                    "Single CODE candidate had patch miss after partial apply; "
                    "restored before repair"
                )
                await self._restore_snapshot(previous_snapshot)
                self.state.scratch["applied_changes"] = 0
                applied = 0
        if applied == 0:
            previous_notes = self.state.notes[note_start:]
            failure_detail = self._candidate_failure_detail(
                previous_notes,
                [],
                failed=True,
            )
            repaired = await self._repair_target_not_found_candidate(
                self._single_code_candidate(decision.changes),
                failure_detail=failure_detail,
                allowed=allowed,
            )
            if repaired is not None:
                self.state.notes.append(
                    "Single CODE candidate target-not-found repair generated "
                    f"{repaired.candidate_id}"
                )
                if isinstance(previous_snapshot, dict):
                    await self._restore_snapshot(previous_snapshot)
                if parent_patch_miss_events:
                    self.state.scratch["repair_parent_patch_miss_events"] = (
                        parent_patch_miss_events
                    )
                self.state.proposed_changes = repaired.changes
                self.state.scratch["single_repair_parent_id"] = "single"
                repair_apply_result = await self._apply_changes(repaired.changes, allowed)
                applied = repair_apply_result.applied
                if (
                    repair_apply_result.has_failures
                    and applied > 0
                    and isinstance(previous_snapshot, dict)
                ):
                    self.state.notes.append(
                        "Single CODE repair candidate had patch miss after partial apply; "
                        "restored before test"
                    )
                    await self._restore_snapshot(previous_snapshot)
                    self.state.scratch["applied_changes"] = 0
                    applied = 0
                if applied > 0:
                    self.state.scratch["patch_miss_repair_status"] = "applied"
                elif repair_apply_result.has_failures:
                    self.state.scratch["patch_miss_repair_status"] = "apply_failed"
                elif self.state.scratch.get("patch_miss_repair_status") == "generated":
                    self.state.scratch["patch_miss_repair_status"] = "still_missing"
        if isinstance(previous_snapshot, dict):
            probe_rejection = self._active_probe_diff_contract_rejection(
                previous_snapshot,
                allowed,
            )
            if probe_rejection is not None:
                status, note, extra = probe_rejection
                current_snapshot = await self._snapshot_files(sorted(allowed))
                patch_text = self._snapshot_patch(previous_snapshot, current_snapshot)
                self.state.notes.append(
                    f"Single CODE candidate rejected after diff check: {note}"
                )
                self.state.scratch["pre_apply_candidate_rejection"] = {
                    "status": status,
                    "note": note,
                    "patch_text": patch_text,
                    "candidate_delta_applied": applied,
                    **extra,
                }
                await self._restore_snapshot(previous_snapshot)
                self.state.scratch["applied_changes"] = 0
        self.state.current = AgentStateName.TEST

    async def _apply_changes(
        self,
        changes: list[CodeChange],
        allowed: set[str],
        *,
        skip_disallowed: bool = False,
    ) -> ApplyResult:
        result = ApplyResult()
        self.state.scratch["patch_miss_events"] = []
        self.state.proposed_changes = changes
        for change in changes:
            if change.path not in allowed:
                if skip_disallowed and not self._is_spec_acceptance_path(change.path):
                    self.state.notes.append(
                        f"Skipped out-of-plan seed change: {change.path}"
                    )
                    continue
                event = self._record_patch_miss_event(
                    change, "out_of_plan", matches_found=0
                )
                result.failed_changes.append(event)
                self.state.notes.append(f"Rejected out-of-plan change: {change.path}")
                continue
            if change.target is not None and change.replacement is not None:
                before_events = len(self._patch_miss_events())
                if await self._apply_replacement(change):
                    result.applied += 1
                else:
                    event = self._latest_patch_miss_event_since(before_events)
                    if event is None:
                        event = self._record_patch_miss_event(
                            change, "replacement_failed", matches_found=0
                        )
                    result.failed_changes.append(event)
                continue
            if change.patch:
                patch_result = await self._apply_patch(change.patch, allowed)
                if patch_result.applied:
                    result.applied += 1
                else:
                    patch_path = (
                        patch_result.rejected_files[0]
                        if patch_result.rejected_files
                        else next(iter(sorted(patch_result.touched_files)), None)
                    )
                    event = self._record_patch_miss_event(
                        change,
                        "patch_rejected",
                        matches_found=0,
                        patch_miss_path=patch_path,
                        patch_touched_files=sorted(patch_result.touched_files),
                        patch_rejected_files=patch_result.rejected_files,
                        failure_detail=patch_result.failure_detail,
                    )
                    result.failed_changes.append(event)
                continue
            if change.content is not None:
                await self.mcp.write_file(str(self.state.repo_root / change.path), change.content)
                result.applied += 1
                continue
            event = self._record_patch_miss_event(
                change, "empty_change", matches_found=0
            )
            result.failed_changes.append(event)
            self.state.notes.append(f"Skipped empty change: {change.path}")
        result.patch_miss_events = self._patch_miss_events()
        self.state.scratch["applied_changes"] = result.applied
        self.state.scratch["apply_failed_changes"] = result.failed_changes
        return result

    def _single_code_candidate(self, changes: list[CodeChange]) -> CodeCandidate:
        active_todo = self.state.scratch.get("active_todo")
        if not isinstance(active_todo, dict):
            active_todo = self._load_active_todo()
        strategy_axis = ""
        reason = "single CODE candidate"
        if isinstance(active_todo, dict):
            strategy_axis = self._normalize_strategy_axis(
                str(active_todo.get("strategy_axis", ""))
            )
            reason = str(active_todo.get("title", "") or reason)
        return CodeCandidate("single", changes, reason, strategy_axis=strategy_axis)

    def _patch_miss_events(self) -> list[dict[str, Any]]:
        events = self.state.scratch.get("patch_miss_events")
        if not isinstance(events, list):
            return []
        return [event for event in events if isinstance(event, dict)]

    def _latest_patch_miss_event_since(self, previous_count: int) -> dict[str, Any] | None:
        events = self._patch_miss_events()
        if len(events) <= previous_count:
            return None
        return events[-1]

    async def _evaluate_code_candidates(
        self, candidates: list[CodeCandidate], allowed: set[str]
    ) -> None:
        baseline_snapshot = self.state.scratch.get("pre_code_snapshot")
        if not isinstance(baseline_snapshot, dict):
            self.state.notes.append("Candidate queue missing baseline snapshot")
            return

        workflow = self.config.get("workflow", {})
        baseline_metric = self.state.scratch.get("best_metric", workflow.get("baseline_metric"))
        iteration_best_metric = int(baseline_metric) if baseline_metric is not None else None
        best_snapshot: dict[str, str | None] | None = None
        best_results: list[TestResult] = []
        best_changes: list[CodeChange] = []
        best_applied = 0

        for candidate in candidates:
            # Candidate-queue gates are intentionally ordered as:
            # active-todo metadata contract, active-todo change scope, duplicate
            # todo variants, axis/family/novelty, apply, probe diff, tests.
            # Single-candidate CODE uses the same change-scope helper before
            # apply; keep scope enforcement in this explicit step rather than
            # nesting it inside the metadata contract gate.
            todo_rejection = self._active_todo_contract_rejection(candidate)
            if todo_rejection is not None:
                status, note = todo_rejection
                self.state.notes.append(f"Candidate {candidate.candidate_id} rejected: {note}")
                extra = self._candidate_rejection_extra(candidate, status, note)
                self._append_candidate_history(
                    candidate,
                    status=status,
                    metric=None,
                    applied=0,
                    failed=True,
                    extra=extra,
                )
                self._record_strategy_attempt(
                    candidate,
                    status=status,
                    metric=None,
                    applied=0,
                    failed=True,
                )
                continue

            scope_rejection = self._active_todo_change_scope_rejection(candidate.changes)
            if scope_rejection is not None:
                status, note = scope_rejection
                self.state.notes.append(f"Candidate {candidate.candidate_id} rejected: {note}")
                extra = self._candidate_rejection_extra(candidate, status, note)
                self._append_candidate_history(
                    candidate,
                    status=status,
                    metric=None,
                    applied=0,
                    failed=True,
                    extra=extra,
                )
                self._record_strategy_attempt(
                    candidate,
                    status=status,
                    metric=None,
                    applied=0,
                    failed=True,
                )
                continue

            duplicate_todo_variant = self._active_todo_duplicate_variant_rejection(candidate)
            if duplicate_todo_variant is not None:
                status, note = duplicate_todo_variant
                self.state.notes.append(f"Candidate {candidate.candidate_id} rejected: {note}")
                extra = self._candidate_rejection_extra(candidate, status, note)
                self._append_candidate_history(
                    candidate,
                    status=status,
                    metric=None,
                    applied=0,
                    failed=True,
                    extra=extra,
                )
                self._record_strategy_attempt(
                    candidate,
                    status=status,
                    metric=None,
                    applied=0,
                    failed=True,
                )
                continue

            axis_rejection = self._candidate_axis_contract_rejection(candidate)
            if axis_rejection is not None:
                status, note = axis_rejection
                self.state.notes.append(f"Candidate {candidate.candidate_id} rejected: {note}")
                extra = self._candidate_rejection_extra(candidate, status, note)
                self._append_candidate_history(
                    candidate,
                    status=status,
                    metric=None,
                    applied=0,
                    failed=True,
                    extra=extra,
                )
                self._record_strategy_attempt(
                    candidate,
                    status=status,
                    metric=None,
                    applied=0,
                    failed=True,
                )
                continue

            family_rejection = self._candidate_family_contract_rejection(candidate)
            if family_rejection is not None:
                status, note = family_rejection
                self.state.notes.append(f"Candidate {candidate.candidate_id} rejected: {note}")
                extra = self._candidate_rejection_extra(candidate, status, note)
                self._append_candidate_history(
                    candidate,
                    status=status,
                    metric=None,
                    applied=0,
                    failed=True,
                    extra=extra,
                )
                self._record_strategy_attempt(
                    candidate,
                    status=status,
                    metric=None,
                    applied=0,
                    failed=True,
                )
                continue

            duplicate_fingerprint = self._rejected_candidate_fingerprint(candidate)
            if duplicate_fingerprint is not None:
                self.state.notes.append(
                    "Candidate "
                    f"{candidate.candidate_id} rejected: forbidden repeated pattern "
                    f"{duplicate_fingerprint}"
                )
                extra = self._candidate_rejection_extra(
                    candidate,
                    "rejected_repeated_pattern",
                    f"forbidden repeated pattern {duplicate_fingerprint}",
                )
                self._append_candidate_history(
                    candidate,
                    status="rejected_repeated_pattern",
                    metric=None,
                    applied=0,
                    failed=True,
                    extra=extra,
                )
                self._record_strategy_attempt(
                    candidate,
                    status="rejected_repeated_pattern",
                    metric=None,
                    applied=0,
                    failed=True,
                )
                continue

            semantic_family_rejection = self._candidate_semantic_family_ban_rejection(
                candidate
            )
            if semantic_family_rejection is not None:
                status, note, rejection_extra = semantic_family_rejection
                self.state.notes.append(f"Candidate {candidate.candidate_id} rejected: {note}")
                extra = self._candidate_rejection_extra(candidate, status, note)
                extra.update(rejection_extra)
                self._append_candidate_history(
                    candidate,
                    status=status,
                    metric=None,
                    applied=0,
                    failed=True,
                    extra=extra,
                )
                self._record_strategy_attempt(
                    candidate,
                    status=status,
                    metric=None,
                    applied=0,
                    failed=True,
                )
                continue

            cooled_axes = self._cooled_candidate_axes(candidate)
            if cooled_axes:
                self.state.notes.append(
                    "Candidate "
                    f"{candidate.candidate_id} rejected: cooled strategy axes "
                    f"{', '.join(cooled_axes)}"
                )
                extra = self._candidate_rejection_extra(
                    candidate,
                    "rejected_cooled_axis",
                    f"cooled strategy axes {', '.join(cooled_axes)}",
                )
                self._append_candidate_history(
                    candidate,
                    status="rejected_cooled_axis",
                    metric=None,
                    applied=0,
                    failed=True,
                    extra=extra,
                )
                self._record_strategy_attempt(
                    candidate,
                    status="rejected_cooled_axis",
                    metric=None,
                    applied=0,
                    failed=True,
                )
                continue

            cooled_regions = self._cooled_candidate_regions(candidate)
            if cooled_regions:
                self.state.notes.append(
                    "Candidate "
                    f"{candidate.candidate_id} rejected: cooled edit regions "
                    f"{', '.join(cooled_regions)}"
                )
                extra = self._candidate_rejection_extra(
                    candidate,
                    "rejected_cooled_region",
                    f"cooled edit regions {', '.join(cooled_regions)}",
                )
                self._append_candidate_history(
                    candidate,
                    status="rejected_cooled_region",
                    metric=None,
                    applied=0,
                    failed=True,
                    extra=extra,
                )
                self._record_strategy_attempt(
                    candidate,
                    status="rejected_cooled_region",
                    metric=None,
                    applied=0,
                    failed=True,
                )
                continue

            await self._restore_snapshot(baseline_snapshot)
            candidate_for_record = candidate
            repair_parent_id = ""
            note_start = len(self.state.notes)
            self.state.scratch["patch_miss_repair_attempted"] = False
            self.state.scratch["patch_miss_repair_status"] = "not_attempted"
            self.state.scratch.pop("repair_parent_patch_miss_events", None)
            apply_result = await self._apply_changes(candidate.changes, allowed)
            applied = apply_result.applied
            current_snapshot = await self._snapshot_files(sorted(allowed))
            patch_text = self._snapshot_patch(baseline_snapshot, current_snapshot)
            parent_patch_miss_events = apply_result.patch_miss_events
            if apply_result.has_failures:
                if applied > 0:
                    self.state.notes.append(
                        "Candidate "
                        f"{candidate.candidate_id} had patch miss after partial apply; "
                        "restored before repair"
                    )
                    await self._restore_snapshot(baseline_snapshot)
                    current_snapshot = await self._snapshot_files(sorted(allowed))
                    patch_text = self._snapshot_patch(baseline_snapshot, current_snapshot)
                    self.state.scratch["applied_changes"] = 0
                    applied = 0
            if applied == 0:
                failure_detail = self._candidate_failure_detail(
                    self.state.notes[note_start:],
                    [],
                    failed=True,
                )
                no_change_reason = failure_detail or "No writable file content changed"
                repaired_candidate = await self._repair_target_not_found_candidate(
                    candidate,
                    failure_detail=no_change_reason,
                    allowed=allowed,
                )
                if repaired_candidate is not None:
                    repair_parent_id = candidate.candidate_id
                    candidate_for_record = repaired_candidate
                    self.state.notes.append(
                        "Candidate "
                        f"{candidate.candidate_id} target-not-found repair generated "
                        f"{repaired_candidate.candidate_id}"
                    )
                    await self._restore_snapshot(baseline_snapshot)
                    if parent_patch_miss_events:
                        self.state.scratch["repair_parent_patch_miss_events"] = (
                            parent_patch_miss_events
                        )
                    note_start = len(self.state.notes)
                    repair_apply_result = await self._apply_changes(
                        repaired_candidate.changes, allowed
                    )
                    applied = repair_apply_result.applied
                    current_snapshot = await self._snapshot_files(sorted(allowed))
                    patch_text = self._snapshot_patch(baseline_snapshot, current_snapshot)
                    if repair_apply_result.has_failures and applied > 0:
                        self.state.notes.append(
                            "Candidate "
                            f"{repaired_candidate.candidate_id} had patch miss after "
                            "partial apply; restored before test"
                        )
                        await self._restore_snapshot(baseline_snapshot)
                        current_snapshot = await self._snapshot_files(sorted(allowed))
                        patch_text = self._snapshot_patch(
                            baseline_snapshot, current_snapshot
                        )
                        self.state.scratch["applied_changes"] = 0
                        applied = 0
                    if applied > 0:
                        self.state.scratch["patch_miss_repair_status"] = "applied"
                    elif repair_apply_result.has_failures:
                        self.state.scratch["patch_miss_repair_status"] = "apply_failed"
                    elif self.state.scratch.get("patch_miss_repair_status") == "generated":
                        self.state.scratch["patch_miss_repair_status"] = "still_missing"
                    if applied == 0:
                        failure_detail = self._candidate_failure_detail(
                            self.state.notes[note_start:],
                            [],
                            failed=True,
                        )
                        no_change_reason = failure_detail or "No writable file content changed"
                if applied == 0:
                    self.state.notes.append(
                        "Candidate "
                        f"{candidate_for_record.candidate_id} rejected: no changes applied"
                        f" ({no_change_reason})"
                    )
                    self._remember_rejected_candidate(candidate_for_record)
                    extra = self._candidate_history_extra(
                        candidate_for_record,
                        status="rejected_no_changes",
                        metric=None,
                        applied=0,
                        failed=True,
                        patch_text=patch_text,
                        results=[],
                        failure_detail=failure_detail,
                        no_change_reason=no_change_reason,
                        repair_parent_id=repair_parent_id,
                    )
                    self._append_candidate_history(
                        candidate_for_record,
                        status="rejected_no_changes",
                        metric=None,
                        applied=0,
                        failed=True,
                        extra=extra,
                    )
                    self._record_strategy_attempt(
                        candidate_for_record,
                        status="rejected_no_changes",
                        metric=None,
                        applied=0,
                        failed=True,
                    )
                    await self._record_exact_context_refresh_request(
                        candidate_for_record,
                        no_change_reason,
                        allowed,
                    )
                    continue
                candidate = candidate_for_record
            else:
                candidate_for_record = candidate

            candidate = candidate_for_record
            if repair_parent_id:
                self.state.notes.append(
                    f"Candidate {candidate.candidate_id} is repair of {repair_parent_id}"
                )

            probe_rejection = self._active_probe_diff_contract_rejection(
                baseline_snapshot,
                allowed,
            )
            if probe_rejection is not None:
                status, note, rejection_extra = probe_rejection
                self.state.notes.append(
                    f"Candidate {candidate.candidate_id} rejected after diff check: {note}"
                )
                extra = self._candidate_history_extra(
                    candidate,
                    status=status,
                    metric=None,
                    applied=0,
                    failed=True,
                    patch_text=patch_text,
                    results=[],
                    failure_detail=note,
                    repair_parent_id=repair_parent_id,
                )
                extra.update(rejection_extra)
                self._append_candidate_history(
                    candidate,
                    status=status,
                    metric=None,
                    applied=0,
                    failed=True,
                    extra=extra,
                )
                self._record_strategy_attempt(
                    candidate,
                    status=status,
                    metric=None,
                    applied=0,
                    failed=True,
                )
                await self._restore_snapshot(baseline_snapshot)
                self.state.scratch["applied_changes"] = 0
                continue

            results = await self._run_candidate_preflight(candidate, allowed)
            if not results:
                results = await self._run_test_commands()
            failed = any(result.exit_code != 0 for result in results)
            metric = self._metric_from_results(results)
            if metric is None:
                failed = failed or bool(workflow.get("require_metric"))
                self.state.notes.append(
                    f"Candidate {candidate.candidate_id} metric not found"
                )
            improved = metric is not None and self._metric_improved(metric, iteration_best_metric)
            failure_detail = self._candidate_failure_detail(
                self.state.notes[note_start:],
                results,
                failed=failed,
            )
            diagnostic_results = await self._run_diagnostic_commands(when="after_test")
            diagnostic_summary = self._diagnostic_summary(diagnostic_results, limit=900)
            if diagnostic_summary:
                self.state.notes.append(
                    f"Candidate {candidate.candidate_id} diagnostics: {diagnostic_summary}"
                )
            self.state.notes.append(
                f"Candidate {candidate.candidate_id} applied={applied} "
                f"metric={metric} failed={failed} improved={improved}"
            )
            status = "improved" if improved and not failed else "rejected"
            extra = self._candidate_history_extra(
                candidate,
                status=status,
                metric=metric,
                applied=applied,
                failed=failed,
                patch_text=patch_text,
                results=results,
                failure_detail=failure_detail,
                repair_parent_id=repair_parent_id,
                diagnostic_results=diagnostic_results,
            )
            if not failed and applied > 0:
                extra.update(
                    self._persist_correct_survivor(
                        candidate,
                        status=status,
                        metric=metric,
                        patch_text=patch_text,
                        results=results,
                        observation=extra,
                    )
                )
            self._append_candidate_history(
                candidate,
                status=status,
                metric=metric,
                applied=applied,
                failed=failed,
                extra=extra,
            )
            self._record_structural_checkpoint(
                candidate,
                status=status,
                metric=metric,
                applied=applied,
                failed=failed,
                patch_text=patch_text,
                extra=extra,
            )
            self._record_strategy_attempt(
                candidate,
                status=status,
                metric=metric,
                applied=applied,
                failed=failed,
            )
            if failed or not improved:
                await self._record_exact_context_refresh_request(
                    candidate,
                    failure_detail,
                    allowed,
                )
                self._remember_rejected_candidate(candidate)
                continue

            iteration_best_metric = metric
            best_snapshot = await self._snapshot_files(sorted(allowed))
            best_results = results
            best_changes = candidate.changes
            best_applied = applied

        if best_snapshot is None:
            await self._restore_snapshot(baseline_snapshot)
            self.state.scratch["applied_changes"] = 0
            self.state.proposed_changes = []
            self.state.notes.append("Candidate queue rejected all candidates")
            return

        await self._restore_snapshot(best_snapshot)
        self.state.scratch["applied_changes"] = best_applied
        self.state.proposed_changes = best_changes
        self.state.test_results = best_results
        if iteration_best_metric is not None:
            self.state.scratch["last_metric"] = iteration_best_metric
        self._append_candidate_history(
            CodeCandidate("accepted", best_changes, "candidate queue accepted best"),
            status="accepted",
            metric=iteration_best_metric,
            applied=best_applied,
            failed=False,
        )
        self.state.notes.append(f"Candidate queue accepted metric={iteration_best_metric}")

    async def _run_test_commands(self) -> list[TestResult]:
        commands = self._test_commands_for_current_scope()
        workflow = self.config.get("workflow", {})
        results = []
        for command in commands:
            start = self._profile_span_start()
            try:
                result = await self.mcp.run_command(
                    command,
                    cwd=str(self.state.repo_root),
                    timeout_seconds=workflow.get("command_timeout_seconds", 120),
                    output_limit=workflow.get("command_output_limit", 200_000),
                )
            except Exception as exc:
                self._record_profile_span(
                    "test_command",
                    start,
                    {
                        "command": command,
                        "cwd": str(self.state.repo_root),
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                raise
            self._record_profile_span(
                "test_command",
                start,
                {
                    "command": command,
                    "cwd": str(self.state.repo_root),
                    "exit_code": result.get("exit_code"),
                    "stdout_chars": len(str(result.get("stdout", ""))),
                    "stderr_chars": len(str(result.get("stderr", ""))),
                    "success": result.get("exit_code") == 0,
                },
            )
            results.append(TestResult(**result))
        return results

    async def _run_candidate_preflight(
        self, candidate: CodeCandidate, allowed: set[str]
    ) -> list[TestResult]:
        workflow = self.config.get("workflow", {})
        if workflow.get("candidate_syntax_preflight", True) is False:
            return []
        paths = sorted(
            {
                change.path
                for change in candidate.changes
                if change.path in allowed and str(change.path).endswith(".py")
            }
        )
        results: list[TestResult] = []
        for rel_path in paths:
            try:
                content = await self.mcp.read_file(str(self.state.repo_root / rel_path))
            except FileNotFoundError:
                continue
            entity_error = ""
            if workflow.get("candidate_html_entity_preflight", True) is not False:
                entity_error = self._html_entity_preflight_error(content)
            if entity_error:
                result = TestResult(
                    command=f"preflight:html-entities {rel_path}",
                    exit_code=1,
                    stderr=entity_error,
                )
                self.state.notes.append(
                    f"Candidate preflight failed for {rel_path}: {entity_error}"
                )
                results.append(result)
                continue
            try:
                ast.parse(content, filename=rel_path)
            except SyntaxError as exc:
                line = exc.lineno or 0
                excerpt = self._source_error_excerpt(content, line)
                stderr = (
                    f"SyntaxError in {rel_path}:{line}:{exc.offset or 0}: {exc.msg}\n"
                    f"{excerpt}"
                )
                result = TestResult(
                    command=f"preflight:syntax {rel_path}",
                    exit_code=1,
                    stderr=stderr,
                )
                self.state.notes.append(
                    f"Candidate preflight failed for {rel_path}: SyntaxError line {line}"
                )
                results.append(result)
        return results

    async def _run_preflight_for_proposed_changes(self) -> list[TestResult]:
        if not self.state.proposed_changes:
            return []
        candidate = CodeCandidate(
            "proposed",
            self.state.proposed_changes,
            "preflight proposed changes",
        )
        return await self._run_candidate_preflight(candidate, self._writable_files())

    @staticmethod
    def _html_entity_preflight_error(content: str) -> str:
        entity_pattern = re.compile(r"&(?:lt|gt|amp|quot|apos);")
        match = entity_pattern.search(content)
        if match is None:
            return ""
        line = content.count("\n", 0, match.start()) + 1
        return (
            f"HTML entity {match.group(0)!r} found in Python source at line {line}; "
            "model likely escaped code. Use literal Python operators instead."
        )

    @staticmethod
    def _source_error_excerpt(content: str, line: int, context: int = 3) -> str:
        lines = content.splitlines()
        if line <= 0:
            return ""
        start = max(line - context, 1)
        end = min(line + context, len(lines))
        return "\n".join(
            f"{index}: {lines[index - 1]}" for index in range(start, end + 1)
        )

    async def _record_exact_context_refresh_request(
        self, candidate: CodeCandidate, failure_detail: str, allowed: set[str]
    ) -> None:
        workflow = self.config.get("workflow", {})
        if workflow.get("exact_context_refresh_after_patch_miss", True) is False:
            return
        indicators = workflow.get(
            "exact_context_refresh_indicators",
            [
                "target not found",
                "comment",
                "blank",
                "no writable file content changed",
            ],
        )
        if isinstance(indicators, str):
            indicators = [indicators]
        detail_lower = failure_detail.lower()
        if not any(str(indicator).lower() in detail_lower for indicator in indicators):
            return
        context = await self._candidate_repair_source_context(candidate, allowed)
        if not context:
            return
        limit = int(workflow.get("exact_context_refresh_char_limit", 12000) or 12000)
        self.state.scratch["exact_context_refresh"] = (
            f"Failure detail:\n{self._truncate_text(failure_detail, 1600)}\n\n"
            f"{self._truncate_text(context, limit)}"
        )
        self.state.notes.append("Queued exact context refresh for next CODE attempt")

    async def _run_diagnostic_commands(self, when: str = "after_test") -> list[dict[str, Any]]:
        specs = self._diagnostic_command_specs(when=when)
        if not specs:
            return []
        workflow = self.config.get("workflow", {})
        default_timeout = int(workflow.get("diagnostic_timeout_seconds", 120) or 120)
        default_limit = int(workflow.get("diagnostic_output_limit", 12_000) or 12_000)
        results: list[dict[str, Any]] = []
        for index, spec in enumerate(specs, start=1):
            command = str(spec.get("command", "")).strip()
            if not command:
                continue
            name = str(spec.get("name") or f"diagnostic-{index}")
            output_limit = int(spec.get("output_limit", default_limit) or default_limit)
            timeout = int(spec.get("timeout_seconds", default_timeout) or default_timeout)
            skip_reason = self._diagnostic_skip_reason(command)
            if skip_reason:
                skipped = self.state.scratch.setdefault("_skipped_diagnostics", [])
                if isinstance(skipped, list) and command not in skipped:
                    skipped.append(command)
                    self.state.notes.append(
                        f"Diagnostic {name} skipped: {skip_reason}"
                    )
                results.append(
                    {
                        "name": name,
                        "command": command,
                        "exit_code": 0,
                        "stdout": f"skipped: {skip_reason}",
                        "stderr": "",
                        "skipped": True,
                    }
                )
                continue
            start = self._profile_span_start()
            try:
                result = await self.mcp.run_command(
                    command,
                    cwd=str(self.state.repo_root),
                    timeout_seconds=timeout,
                    output_limit=output_limit,
                )
            except Exception as exc:
                result = {
                    "command": command,
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": f"{type(exc).__name__}: {exc}",
                }
            self._record_profile_span(
                "diagnostic_command",
                start,
                {
                    "name": name,
                    "command": command,
                    "cwd": str(self.state.repo_root),
                    "exit_code": result.get("exit_code"),
                    "stdout_chars": len(str(result.get("stdout", ""))),
                    "stderr_chars": len(str(result.get("stderr", ""))),
                    "success": result.get("exit_code") == 0,
                },
            )
            results.append(
                {
                    "name": name,
                    "command": command,
                    "exit_code": result.get("exit_code"),
                    "stdout": self._truncate_text(str(result.get("stdout", "")), output_limit),
                    "stderr": self._truncate_text(str(result.get("stderr", "")), output_limit),
                }
            )
        return results

    def _diagnostic_skip_reason(self, command: str) -> str:
        try:
            parts = shlex.split(command)
        except ValueError:
            return ""
        for part in parts[1:]:
            if part.startswith("-"):
                continue
            if not part.endswith((".py", ".sh")):
                continue
            candidate = Path(part)
            if candidate.is_absolute():
                exists = candidate.exists()
            else:
                exists = (self.state.repo_root / candidate).exists()
            if not exists:
                return f"missing diagnostic file {part}"
            return ""
        return ""

    def _diagnostic_command_specs(self, when: str = "after_test") -> list[dict[str, Any]]:
        configured = self.config.get("workflow", {}).get("diagnostic_commands", [])
        if configured in (None, "", []):
            return []
        if not isinstance(configured, list):
            configured = [configured]
        specs: list[dict[str, Any]] = []
        for item in configured:
            if isinstance(item, dict):
                command = str(item.get("command", "")).strip()
                if not command:
                    continue
                item_when = str(item.get("when", "after_test") or "after_test")
                if item_when != when:
                    continue
                spec = {
                    "name": str(item.get("name") or ""),
                    "command": command,
                    "when": item_when,
                }
                for key in ("timeout_seconds", "output_limit"):
                    if item.get(key) is not None:
                        spec[key] = item[key]
                specs.append(spec)
                continue
            command = str(item).strip()
            if command:
                specs.append({"name": "", "command": command, "when": "after_test"})
        return specs

    def _diagnostic_summary(
        self, diagnostic_results: list[dict[str, Any]], limit: int = 1200
    ) -> str:
        if not diagnostic_results:
            return ""
        items: list[str] = []
        for result in diagnostic_results:
            output = "\n".join(
                part
                for part in (
                    str(result.get("stdout", "")).strip(),
                    str(result.get("stderr", "")).strip(),
                )
                if part
            )
            name = str(result.get("name") or "diagnostic")
            items.append(
                f"{name} exit={result.get('exit_code')}: "
                f"{self._truncate_text(output, 500) if output else 'no output'}"
            )
        return self._truncate_text(" | ".join(items), limit)

    async def test(self) -> None:
        self.state.scratch.pop("spec_regression_failed", None)
        frozen_acceptance_failed = self._frozen_acceptance_changed()
        if not frozen_acceptance_failed:
            self.state.test_results = await self._run_preflight_for_proposed_changes()
            if not self.state.test_results:
                self.state.test_results = await self._run_test_commands()
        failed = False
        for result in self.state.test_results:
            failed = failed or result.exit_code != 0
        if frozen_acceptance_failed:
            failed = True
        if self.state.scratch.get("applied_changes", 0) == 0:
            failed = True
            self.state.notes.append("No code changes were applied")
        correctness_failed = failed
        metric_failed = self._evaluate_metric_acceptance()
        failed = failed or metric_failed
        if self._spec_mode_enabled() and not failed:
            regression_results = await self._run_spec_regression_gate()
            if regression_results:
                self.state.test_results.extend(regression_results)
                failed = any(result.exit_code != 0 for result in regression_results)
                if failed:
                    self.state.scratch["spec_regression_failed"] = True
                    correctness_failed = True
        await self._record_single_candidate_observation(
            failed=failed,
            correctness_passed=not correctness_failed,
        )
        if failed:
            if not self.state.scratch.get("spec_regression_failed"):
                await self._restore_pre_code_snapshot()
            else:
                self.state.notes.append(
                    "Keeping current task changes after regression gate failure"
                )
        else:
            await self._persist_current_best_state()
        if self._spec_mode_enabled():
            await self._handle_spec_task_test_result(failed)
            return
        if self.config.get("workflow", {}).get("deterministic_test_decision"):
            if failed and self._should_retry_rejected_candidate():
                self.state.loop_count += 1
                self.state.current = self._retry_state_after_failure()
                return
            if not failed and self._should_continue_after_improvement():
                self._create_validated_pattern_followup_todo()
                self.state.loop_count += 1
                self.state.current = AgentStateName.CODE
                self.state.notes.append(
                    f"Continuing after improvement with baseline={self.state.scratch.get('best_metric')}"
                )
                return
            self.state.current = AgentStateName.FAILED if failed else AgentStateName.DONE
            return

        try:
            decision = await self._json_call("tester", test_prompt(self.state), TestDecision)
        except JsonValidationError as exc:
            self.state.notes.append(
                "TEST decision failed; using deterministic test result: "
                f"{exc}"
            )
            decision = TestDecision(
                status="retry" if failed else "pass",
                reason="fallback after TEST decision failure",
                next_focus="Use concrete test output and avoid malformed tester JSON.",
            )
        if not failed and decision.status == "pass":
            if self._should_continue_after_improvement():
                self._create_validated_pattern_followup_todo()
                self.state.loop_count += 1
                self.state.current = AgentStateName.CODE
                self.state.notes.append(
                    f"Continuing after improvement with baseline={self.state.scratch.get('best_metric')}"
                )
                return
            self.state.current = AgentStateName.DONE
            return

        self.state.loop_count += 1
        if self.state.loop_count >= self.state.max_loops or decision.status == "fail":
            self.state.current = AgentStateName.FAILED
            return

        self.state.notes.append(f"Retry focus: {decision.next_focus or decision.reason}")
        self.state.current = self._retry_state_after_failure(
            decision.next_focus or decision.reason
        )

    async def _record_single_candidate_observation(
        self, failed: bool, correctness_passed: bool
    ) -> None:
        workflow = self.config.get("workflow", {})
        if workflow.get("candidate_queue"):
            return
        if self._candidate_history_path() is None:
            return
        changes = self.state.proposed_changes
        if not changes:
            return
        active_todo = self.state.scratch.get("active_todo")
        if not isinstance(active_todo, dict):
            active_todo = self._load_active_todo()
            if active_todo:
                self.state.scratch["active_todo"] = active_todo
        strategy_axis = ""
        reason = "single CODE candidate"
        if isinstance(active_todo, dict):
            strategy_axis = self._normalize_strategy_axis(
                str(active_todo.get("strategy_axis", ""))
            )
            title = str(active_todo.get("title", "")).strip()
            if title:
                reason = title
        candidate = CodeCandidate(
            f"loop-{self.state.loop_count:03d}-single",
            changes,
            reason,
            strategy_axis=strategy_axis,
        )
        applied = int(self.state.scratch.get("applied_changes", 0) or 0)
        metric = self.state.scratch.get("last_metric")
        if not isinstance(metric, int):
            metric_acceptance = self.state.scratch.get("metric_acceptance")
            if isinstance(metric_acceptance, dict) and isinstance(
                metric_acceptance.get("metric"), int
            ):
                metric = metric_acceptance["metric"]
            else:
                metric = None
        previous_snapshot = self.state.scratch.get("pre_code_snapshot")
        patch_text = ""
        if isinstance(previous_snapshot, dict):
            current_snapshot = await self._snapshot_files(sorted(self._writable_files()))
            patch_text = self._snapshot_patch(previous_snapshot, current_snapshot)
        pre_apply_rejection = self.state.scratch.pop(
            "pre_apply_candidate_rejection", None
        )
        status = (
            str(pre_apply_rejection.get("status", "rejected"))
            if isinstance(pre_apply_rejection, dict)
            else (
                "improved"
                if self.state.scratch.get("metric_improved") is True and not failed
                else "rejected"
            )
        )
        history_failed = (
            True
            if isinstance(pre_apply_rejection, dict)
            else failed and (not correctness_passed or metric is None)
        )
        failure_detail = self._candidate_failure_detail(
            self.state.notes[-12:],
            self.state.test_results,
            failed=history_failed,
        )
        if isinstance(pre_apply_rejection, dict):
            note = str(pre_apply_rejection.get("note", "")).strip()
            if note:
                failure_detail = (
                    f"{note}; {failure_detail}" if failure_detail else note
                )
            rejected_patch_text = str(pre_apply_rejection.get("patch_text") or "")
            if rejected_patch_text:
                patch_text = rejected_patch_text
        no_change_reason = failure_detail if applied == 0 else ""
        repair_parent_id = str(self.state.scratch.pop("single_repair_parent_id", ""))
        extra = self._candidate_history_extra(
            candidate,
            status=status,
            metric=metric,
            applied=applied,
            failed=history_failed,
            patch_text=patch_text,
            results=self.state.test_results,
            failure_detail=failure_detail,
            no_change_reason=no_change_reason,
            repair_parent_id=repair_parent_id,
        )
        if isinstance(pre_apply_rejection, dict):
            extra.update(
                {
                    key: value
                    for key, value in pre_apply_rejection.items()
                    if key not in {"status", "note", "patch_text"}
                    and value not in (None, "", [], {})
                }
            )
        self.state.scratch["last_candidate_observation"] = extra
        if correctness_passed and applied > 0:
            extra.update(
                self._persist_correct_survivor(
                    candidate,
                    status=status,
                    metric=metric,
                    patch_text=patch_text,
                    results=self.state.test_results,
                    observation=extra,
                )
            )
        self._append_candidate_history(
            candidate,
            status=status,
            metric=metric,
            applied=applied,
            failed=history_failed,
            extra=extra,
        )
        self._record_strategy_attempt(
            candidate,
            status=status,
            metric=metric,
            applied=applied,
            failed=history_failed,
        )

    def _writable_files(self) -> set[str]:
        workflow = self.config.get("workflow", {})
        task_writable = self._current_spec_task_writable_files()
        candidates = task_writable or workflow.get("writable_files") or self.state.planned_files
        writable = {str(path) for path in candidates}
        if workflow.get("allow_external_context_writes"):
            return writable
        external_paths = self._external_context_path_keys()
        return {
            path
            for path in writable
            if self._repo_path_key(path) not in external_paths
            and not self._is_spec_acceptance_path(path)
        }

    async def _snapshot_files(self, paths: list[str]) -> dict[str, str | None]:
        snapshot = {}
        for rel_path in paths:
            try:
                snapshot[rel_path] = await self.mcp.read_file(str(self.state.repo_root / rel_path))
            except FileNotFoundError:
                snapshot[rel_path] = None
                self.state.notes.append(f"Snapshot missing file: {rel_path}")
        return snapshot

    async def _restore_pre_code_snapshot(self) -> None:
        snapshot = self.state.scratch.get("pre_code_snapshot")
        if not isinstance(snapshot, dict):
            return
        await self._restore_snapshot(snapshot)
        self.state.notes.append("Restored writable files after rejected candidate")

    async def _restore_snapshot(self, snapshot: dict[str, str | None]) -> None:
        for rel_path, content in snapshot.items():
            abs_path = str(self.state.repo_root / rel_path)
            if content is None:
                await self.mcp.delete_file(abs_path)
                continue
            await self.mcp.write_file(abs_path, str(content))

    def _evaluate_metric_acceptance(self) -> bool:
        workflow = self.config.get("workflow", {})
        self.state.scratch["metric_improved"] = False
        requires_improvement = self._spec_metric_task_requires_improvement()
        self.state.scratch["metric_acceptance"] = {
            "requires_improvement": requires_improvement,
        }
        if not workflow.get("metric_regex"):
            if requires_improvement:
                self.state.scratch["metric_acceptance"].update(
                    {
                        "failed": True,
                        "failure_class": "metric_missing",
                        "summary": (
                            "Spec metric task requires improvement, but "
                            "workflow.metric_regex is not configured."
                        ),
                    }
                )
                self.state.notes.append(
                    "Spec metric task requires improvement, but workflow.metric_regex is not configured"
                )
                return True
            return False
        metric = self._metric_from_results(self.state.test_results)
        if metric is None:
            self.state.notes.append(
                f"Metric not found with regex: {workflow.get('metric_regex')}"
            )
            self.state.scratch["metric_acceptance"].update(
                {
                    "failed": bool(
                        workflow.get("require_metric") or requires_improvement
                    ),
                    "failure_class": "metric_missing",
                    "summary": "Metric output was not parseable.",
                }
            )
            return bool(workflow.get("require_metric") or requires_improvement)
        self.state.scratch["last_metric"] = metric

        baseline = self.state.scratch.get(
            "best_metric", workflow.get("baseline_metric")
        )
        if baseline is None:
            self.state.scratch["best_metric"] = metric
            self.state.notes.append(f"Recorded initial metric: {metric}")
            self.state.scratch["metric_acceptance"].update(
                {
                    "metric": metric,
                    "baseline": None,
                    "improved": False,
                    "failed": requires_improvement,
                    "failure_class": (
                        "metric_baseline_missing" if requires_improvement else None
                    ),
                    "summary": (
                        "Recorded initial metric; no baseline was available "
                        "to prove improvement."
                    ),
                }
            )
            if requires_improvement:
                self.state.notes.append(
                    "Spec metric task requires improvement, but no baseline metric is available"
                )
            return requires_improvement

        baseline_int = int(baseline)
        improved = self._metric_improved(metric, baseline_int)
        self.state.notes.append(
            f"Metric candidate={metric} baseline={baseline_int} improved={improved}"
        )
        metric_summary = (
            f"Metric candidate={metric} baseline={baseline_int} improved={improved}."
        )
        self.state.scratch["metric_acceptance"].update(
            {
                "metric": metric,
                "baseline": baseline_int,
                "improved": improved,
                "failed": bool(
                    not improved
                    and (workflow.get("accept_if_improved") or requires_improvement)
                ),
                "failure_class": None if improved else "no_improvement",
                "summary": metric_summary
                if improved
                else (
                    metric_summary
                    + " Treat this as an inert/no-signal edit; ensure the "
                    "changed code path is actually executed."
                ),
            }
        )
        if improved:
            self.state.scratch["best_metric"] = metric
            self.state.scratch["metric_improved"] = True
            return False
        if requires_improvement:
            self.state.notes.append(
                "Spec metric task requires improvement before close; rejecting unchanged metric"
            )
        return bool(workflow.get("accept_if_improved") or requires_improvement)

    def _spec_metric_task_requires_improvement(self) -> bool:
        if not self._spec_mode_enabled():
            return False
        workflow = self.config.get("workflow", {})
        if workflow.get("spec_metric_requires_improvement", True) is False:
            return False
        task = self._current_spec_task()
        acceptance = task.get("acceptance") if isinstance(task, dict) else None
        return isinstance(acceptance, dict) and str(acceptance.get("kind")) == "metric"

    def _metric_from_results(self, results: list[TestResult]) -> int | None:
        metric_regex = self.config.get("workflow", {}).get("metric_regex")
        if not metric_regex:
            return None
        joined_output = "\n".join(f"{result.stdout}\n{result.stderr}" for result in results)
        return self._extract_metric(joined_output, str(metric_regex))

    def _metric_improved(self, metric: int, baseline: int | None) -> bool:
        if baseline is None:
            return True
        if self.config.get("workflow", {}).get("metric_goal", "minimize") == "maximize":
            return metric > baseline
        return metric < baseline

    def _should_retry_rejected_candidate(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("retry_rejected_candidates")) and (
            self.state.loop_count + 1 < self.state.max_loops
        )

    def _retry_state_after_failure(self, retry_focus: str = "") -> AgentStateName:
        if self._should_reflect_after_failure(retry_focus):
            return AgentStateName.REFLECT
        return AgentStateName.CODE

    def _should_reflect_after_failure(self, retry_focus: str = "") -> bool:
        workflow = self.config.get("workflow", {})
        if not workflow.get("reflect_before_retry"):
            return False
        if workflow.get("reflect_conditionally", True) is False:
            return True
        failure_class = self._retry_failure_class(retry_focus)
        counts = self.state.scratch.setdefault("retry_failure_class_counts", {})
        if not isinstance(counts, dict):
            counts = {}
            self.state.scratch["retry_failure_class_counts"] = counts
        count = int(counts.get(failure_class, 0) or 0) + 1
        counts[failure_class] = count

        simple_failures = {
            "patch_miss",
            "duplicate_variant",
            "axis_mismatch",
            "family_mismatch",
            "contract_mismatch",
        }
        if failure_class in simple_failures:
            threshold = int(
                workflow.get("reflect_after_repeated_failure_class", 3) or 3
            )
            if count < threshold:
                self.state.notes.append(
                    "Skipping REFLECT for structured retry failure "
                    f"{failure_class} count={count}/{threshold}"
                )
                return False
            counts[failure_class] = 0
        return True

    def _retry_failure_class(self, retry_focus: str = "") -> str:
        recent_notes = " | ".join(self.state.notes[-8:])
        detail = " | ".join(part for part in (recent_notes, retry_focus) if part)
        return self._candidate_failure_class(
            status="rejected",
            metric=self.state.scratch.get("last_metric"),
            applied=int(self.state.scratch.get("applied_changes", 0) or 0),
            failed=True,
            results=self.state.test_results,
            failure_detail=detail,
            no_change_reason=detail if self.state.scratch.get("applied_changes", 0) == 0 else "",
        )

    def _should_continue_after_improvement(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("continue_after_improvement")) and (
            self.state.scratch.get("metric_improved") is True
            and self.state.loop_count + 1 < self.state.max_loops
        )

    def _has_current_run_improvement(self) -> bool:
        if self.state.scratch.get("metric_improved") is True:
            return True
        return bool(
            self._latest_candidate_record_with_status("improved")
            or self._latest_candidate_record_with_status("accepted")
        )

    async def _persist_current_best_state(self) -> None:
        workflow = self.config.get("workflow", {})
        should_persist = (
            bool(workflow.get("continue_after_improvement"))
            or workflow.get("best_state_path") is not None
        )
        if not should_persist or self.state.scratch.get("metric_improved") is not True:
            return

        state_path = self._workflow_artifact_path("best_state_path", ".local_micro_agent/best_state.json")
        patch_path = self._workflow_artifact_path("best_patch_path", ".local_micro_agent/best.patch")
        allowed = sorted(self._writable_files())
        current_snapshot = await self._snapshot_files(allowed)
        previous_snapshot = self.state.scratch.get("pre_code_snapshot")
        patch_text = ""
        if isinstance(previous_snapshot, dict):
            patch_text = self._snapshot_patch(previous_snapshot, current_snapshot)

        state_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "loop": self.state.loop_count,
            "metric": self.state.scratch.get("best_metric", self.state.scratch.get("last_metric")),
            "last_metric": self.state.scratch.get("last_metric"),
            "changes": self._summarize_changes(self.state.proposed_changes),
            "adaptive_search_memory": self.state.scratch.get("adaptive_search_memory", {}),
            "tactic_library": self.state.scratch.get("tactic_library", ""),
            "notes_tail": self.state.notes[-12:],
        }
        state_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        patch_path.write_text(patch_text)
        self.state.notes.append(f"Persisted best state: {state_path}")

    @staticmethod
    def _snapshot_patch(
        before: dict[str, str | None], after: dict[str, str | None]
    ) -> str:
        hunks = []
        for path in sorted(set(before) | set(after)):
            before_text = before.get(path) or ""
            after_text = after.get(path) or ""
            if before_text == after_text:
                continue
            hunks.extend(
                difflib.unified_diff(
                    before_text.splitlines(keepends=True),
                    after_text.splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
            )
        return "".join(hunks)

    @staticmethod
    def _extract_metric(text: str, pattern: str) -> int | None:
        matches = re.findall(pattern, text)
        if not matches:
            return None
        value = matches[-1]
        if isinstance(value, tuple):
            value = next((part for part in value if part), "")
        try:
            return int(str(value))
        except ValueError:
            return None

    async def _apply_replacement(self, change: CodeChange) -> bool:
        path = change.path
        target = change.target or ""
        replacement = change.replacement or ""
        abs_path = self.state.repo_root / path
        if target == replacement:
            self._record_patch_miss_event(change, "patch_noop", matches_found=0)
            self.state.notes.append(f"Replacement is a no-op: {path}")
            return False
        if self._without_comment_lines(target) == self._without_comment_lines(replacement):
            self._record_patch_miss_event(change, "patch_noop", matches_found=0)
            self.state.notes.append(f"Replacement only changes comments or blank lines: {path}")
            return False
        original = await self.mcp.read_file(str(abs_path))
        resolved = self._resolve_replacement_target(original, change)
        if resolved is None:
            return False
        start, end, target, retarget_mode = resolved
        if retarget_mode:
            if target.endswith("\n") and not replacement.endswith("\n"):
                replacement += "\n"
            if target == replacement:
                self._record_patch_miss_event(change, "patch_noop", matches_found=1)
                self.state.notes.append(f"Replacement is a no-op after retarget: {path}")
                return False
            if self._without_comment_lines(target) == self._without_comment_lines(
                replacement
            ):
                self._record_patch_miss_event(change, "patch_noop", matches_found=1)
                self.state.notes.append(
                    f"Replacement only changes comments or blank lines after retarget: {path}"
                )
                return False
        await self.mcp.write_file(str(abs_path), original[:start] + replacement + original[end:])
        return True

    def _resolve_replacement_target(
        self, content: str, change: CodeChange
    ) -> tuple[int, int, str, str] | None:
        target = change.target or ""
        path = change.path
        exact_spans = self._literal_spans(content, target)
        if len(exact_spans) == 1:
            start, end = exact_spans[0]
            return start, end, content[start:end], ""

        line_ranges = self._line_hint_ranges(content, change)
        anchor_ranges, anchor_before_found, anchor_after_found = self._anchor_hint_ranges(
            content, change
        )
        range_sets = self._retarget_range_sets(line_ranges, anchor_ranges)

        for mode, ranges in range_sets:
            span_matches = self._literal_spans_in_ranges(content, target, ranges)
            if len(span_matches) == 1:
                start, end = span_matches[0]
                self.state.notes.append(
                    f"Retargeted replacement target via {mode} in {path}"
                )
                return start, end, content[start:end], mode
            if len(span_matches) > 1:
                self._record_patch_miss_event(
                    change,
                    "ambiguous_target",
                    matches_found=len(span_matches),
                    anchor_before_found=anchor_before_found,
                    anchor_after_found=anchor_after_found,
                )
                self.state.notes.append(f"Replacement target is ambiguous: {path}")
                return None

        if len(exact_spans) > 1:
            self._record_patch_miss_event(
                change,
                "ambiguous_target",
                matches_found=len(exact_spans),
                anchor_before_found=anchor_before_found,
                anchor_after_found=anchor_after_found,
            )
            self.state.notes.append(f"Replacement target is ambiguous: {path}")
            return None

        for mode, ranges in range_sets:
            stripped_matches = self._stripped_line_spans_in_ranges(content, target, ranges)
            if len(stripped_matches) == 1:
                start, end = stripped_matches[0]
                self.state.notes.append(
                    f"Retargeted replacement target via {mode} whitespace in {path}"
                )
                return start, end, content[start:end], f"{mode}_whitespace"
            if len(stripped_matches) > 1:
                self._record_patch_miss_event(
                    change,
                    "ambiguous_target",
                    matches_found=len(stripped_matches),
                    anchor_before_found=anchor_before_found,
                    anchor_after_found=anchor_after_found,
                )
                self.state.notes.append(f"Replacement target is ambiguous: {path}")
                return None

        fallback = self._stripped_line_spans_in_ranges(content, target, [(0, len(content))])
        if len(fallback) == 1:
            start, end = fallback[0]
            self.state.notes.append(
                "Retargeted replacement target to exact current source whitespace "
                f"in {path}"
            )
            return start, end, content[start:end], "stripped_whitespace"
        if len(fallback) > 1:
            self._record_patch_miss_event(
                change,
                "ambiguous_target",
                matches_found=len(fallback),
                anchor_before_found=anchor_before_found,
                anchor_after_found=anchor_after_found,
            )
            self.state.notes.append(f"Replacement target is ambiguous: {path}")
            return None

        miss_kind = (
            "line_anchor_mismatch"
            if line_ranges or change.anchor_before or change.anchor_after
            else "target_not_found"
        )
        self._record_patch_miss_event(
            change,
            miss_kind,
            matches_found=0,
            anchor_before_found=anchor_before_found,
            anchor_after_found=anchor_after_found,
        )
        self.state.notes.append(f"Replacement target not found: {path}")
        return None

    def _retarget_range_sets(
        self,
        line_ranges: list[tuple[int, int]],
        anchor_ranges: list[tuple[int, int]],
    ) -> list[tuple[str, list[tuple[int, int]]]]:
        ranges: list[tuple[str, list[tuple[int, int]]]] = []
        if line_ranges and anchor_ranges:
            intersections = self._intersect_ranges(line_ranges, anchor_ranges)
            if intersections:
                ranges.append(("line_anchor", intersections))
            ranges.append(("anchor", anchor_ranges))
            return ranges
        if anchor_ranges:
            ranges.append(("anchor", anchor_ranges))
        if line_ranges:
            ranges.append(("line_window", line_ranges))
        return ranges

    @staticmethod
    def _literal_spans(content: str, target: str) -> list[tuple[int, int]]:
        if not target:
            return []
        spans: list[tuple[int, int]] = []
        start = 0
        while True:
            index = content.find(target, start)
            if index == -1:
                break
            spans.append((index, index + len(target)))
            start = index + 1
        return spans

    def _literal_spans_in_ranges(
        self,
        content: str,
        target: str,
        ranges: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        spans: set[tuple[int, int]] = set()
        for range_start, range_end in ranges:
            for start, end in self._literal_spans(content[range_start:range_end], target):
                spans.add((range_start + start, range_start + end))
        return sorted(spans)

    def _stripped_line_spans_in_ranges(
        self,
        content: str,
        target: str,
        ranges: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        target_lines = target.splitlines()
        if not target_lines:
            return []
        target_key = [line.strip() for line in target_lines]
        lines = content.splitlines(keepends=True)
        if not lines:
            return []
        starts: list[int] = []
        offset = 0
        for line in lines:
            starts.append(offset)
            offset += len(line)
        width = len(target_lines)
        spans: set[tuple[int, int]] = set()
        for index in range(0, len(lines) - width + 1):
            window = lines[index : index + width]
            if [line.strip() for line in window] != target_key:
                continue
            start = starts[index]
            end = starts[index + width - 1] + len(lines[index + width - 1])
            if any(start >= range_start and end <= range_end for range_start, range_end in ranges):
                spans.add((start, end))
        return sorted(spans)

    def _line_hint_ranges(
        self, content: str, change: CodeChange
    ) -> list[tuple[int, int]]:
        if change.start_line is None and change.end_line is None:
            return []
        lines = content.splitlines(keepends=True)
        if not lines:
            return []
        start_line = change.start_line or change.end_line or 1
        end_line = change.end_line or change.start_line or start_line
        if start_line > end_line:
            start_line, end_line = end_line, start_line
        if end_line < 1 or start_line > len(lines):
            return []
        context_lines = int(
            self.config.get("workflow", {}).get("patch_line_anchor_context_lines", 8)
        )
        start_index = max(0, start_line - 1 - context_lines)
        end_index = min(len(lines), end_line + context_lines)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        return [(offsets[start_index], offsets[end_index])]

    def _anchor_hint_ranges(
        self, content: str, change: CodeChange
    ) -> tuple[list[tuple[int, int]], bool, bool]:
        before = change.anchor_before or ""
        after = change.anchor_after or ""
        before_spans = self._literal_spans(content, before) if before else []
        after_spans = self._literal_spans(content, after) if after else []
        ranges: list[tuple[int, int]] = []
        context_chars = int(
            self.config.get("workflow", {}).get("patch_anchor_context_chars", 4000)
        )
        if before and after and before_spans and after_spans:
            for before_start, before_end in before_spans:
                next_after = next(
                    (
                        (after_start, after_end)
                        for after_start, after_end in after_spans
                        if after_start >= before_end
                    ),
                    None,
                )
                if next_after is not None:
                    ranges.append((before_start, next_after[1]))
        elif before and before_spans:
            for before_start, before_end in before_spans:
                ranges.append((before_start, min(len(content), before_end + context_chars)))
        elif after and after_spans:
            for after_start, after_end in after_spans:
                ranges.append((max(0, after_start - context_chars), after_end))
        return ranges, (not before or bool(before_spans)), (not after or bool(after_spans))

    @staticmethod
    def _intersect_ranges(
        left: list[tuple[int, int]], right: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        intersections: set[tuple[int, int]] = set()
        for left_start, left_end in left:
            for right_start, right_end in right:
                start = max(left_start, right_start)
                end = min(left_end, right_end)
                if start < end:
                    intersections.add((start, end))
        return sorted(intersections)

    def _record_patch_miss_event(
        self,
        change: CodeChange,
        patch_miss_kind: str,
        *,
        matches_found: int,
        anchor_before_found: bool | None = None,
        anchor_after_found: bool | None = None,
        patch_miss_path: str | None = None,
        patch_touched_files: list[str] | None = None,
        patch_rejected_files: list[str] | None = None,
        failure_detail: str = "",
    ) -> dict[str, Any]:
        target = change.target or ""
        event: dict[str, Any] = {
            "patch_miss_path": patch_miss_path or change.path,
            "patch_miss_kind": patch_miss_kind,
            "target_line_count": len(target.splitlines()),
            "target_hash": change.target_hash
            or hashlib.sha256(target.encode("utf-8")).hexdigest()[:16],
            "matches_found": matches_found,
            "repair_attempted": False,
            "repair_status": "not_attempted",
        }
        if patch_touched_files:
            event["patch_touched_files"] = patch_touched_files
        if patch_rejected_files:
            event["patch_rejected_files"] = patch_rejected_files
        if failure_detail:
            event["patch_failure_detail"] = self._truncate_text(failure_detail, 500)
        if change.start_line is not None or change.end_line is not None:
            event["line_range"] = {
                "start_line": change.start_line,
                "end_line": change.end_line,
            }
        if change.target_region:
            event["fresh_context_region"] = change.target_region
            event["target_region"] = change.target_region
        if change.anchor_before is not None:
            event["anchor_before_found"] = bool(anchor_before_found)
        if change.anchor_after is not None:
            event["anchor_after_found"] = bool(anchor_after_found)
        events = self.state.scratch.setdefault("patch_miss_events", [])
        if isinstance(events, list):
            events.append(event)
        return event

    @staticmethod
    def _without_comment_lines(text: str) -> str:
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(line)
        return "\n".join(lines)

    async def _apply_patch(self, patch: str, allowed: set[str]) -> PatchApplyResult:
        touched_files = self._patch_touched_files(patch)
        if not touched_files:
            self.state.notes.append("Patch rejected: no changed files detected")
            return PatchApplyResult(failure_detail="no changed files detected")
        rejected_files = sorted(path for path in touched_files if path not in allowed)
        if rejected_files:
            detail = "touches out-of-plan files: " + ", ".join(rejected_files[:8])
            self.state.notes.append(
                "Patch rejected: " + detail
            )
            return PatchApplyResult(
                touched_files=touched_files,
                rejected_files=rejected_files,
                failure_detail=detail,
            )
        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as handle:
            handle.write(patch)
            patch_path = handle.name
        patch_arg = shlex.quote(patch_path)
        try:
            result = await self.mcp.run_command(
                f"git apply --check {patch_arg}", cwd=str(self.state.repo_root)
            )
            if result["exit_code"] != 0:
                detail = str(result["stderr"][-1000:])
                self.state.notes.append(f"Patch rejected: {detail}")
                return PatchApplyResult(
                    touched_files=touched_files,
                    failure_detail=detail,
                )
            result = await self.mcp.run_command(
                f"git apply {patch_arg}", cwd=str(self.state.repo_root)
            )
            if result["exit_code"] != 0:
                detail = str(result["stderr"][-1000:])
                self.state.notes.append(f"Patch apply failed: {detail}")
                return PatchApplyResult(
                    touched_files=touched_files,
                    failure_detail=detail,
                )
            return PatchApplyResult(applied=True, touched_files=touched_files)
        finally:
            Path(patch_path).unlink(missing_ok=True)

    @classmethod
    def _patch_touched_files(cls, patch: str) -> set[str]:
        paths: set[str] = set()
        for line in patch.splitlines():
            if line.startswith("diff --git "):
                match = re.match(r"^diff --git a/(.*) b/(.*)$", line)
                if not match:
                    continue
                for raw_path in match.groups():
                    normalized = cls._normalize_patch_path(raw_path)
                    if normalized:
                        paths.add(normalized)
                continue
            if line.startswith("--- ") or line.startswith("+++ "):
                normalized = cls._normalize_patch_path(line[4:])
                if normalized:
                    paths.add(normalized)
        return paths

    @staticmethod
    def _normalize_patch_path(raw_path: str) -> str:
        path = raw_path.strip()
        if not path or path == "/dev/null":
            return ""
        if path.startswith('"'):
            try:
                parsed = shlex.split(path)
            except ValueError:
                parsed = []
            if parsed:
                path = parsed[0]
        path = path.split("\t", 1)[0]
        if path.startswith("a/") or path.startswith("b/"):
            path = path[2:]
        return path

def load_config(path: Path) -> dict[str, Any]:
    # Expand presets here so CLI-side values derived from the raw config,
    # such as AgentState.max_loops, see the preset-provided defaults.
    return apply_workflow_preset(json.loads(path.read_text()))


def dump_prompts() -> str:
    blocks = []
    for name, prompt in PROMPT_MARKDOWN.items():
        blocks.append(f"## {name}\n\n```markdown\n{prompt}\n```")
    return "\n\n".join(blocks)


async def async_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--dump-prompts", action="store_true")
    args = parser.parse_args()

    if args.dump_prompts:
        print(dump_prompts())
        return

    config = load_config(args.config)
    state = AgentState(
        repo_root=args.repo.resolve(),
        user_request=args.request,
        max_loops=config.get("workflow", {}).get("max_code_test_loops", 3),
    )
    result = await MicroAgent(config, state).run()
    print(json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
