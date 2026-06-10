from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import re
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any

from .decisions import CodeCandidate, CodeDecision, ReadDecision, TestDecision
from .mcp_client import McpServerSpec, McpToolClient
from .models import ModelManager
from .prompts import (
    PROMPT_MARKDOWN,
    code_prompt,
    plan_prompt,
    read_prompt,
    reflect_prompt,
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
        raw_workflow = config.get("workflow", {})
        self.config = apply_workflow_preset(config)
        preset_loops = self.config.get("workflow", {}).get("max_code_test_loops")
        if (
            isinstance(raw_workflow, dict)
            and "max_code_test_loops" not in raw_workflow
            and isinstance(preset_loops, int)
        ):
            # The preset is the only source for the loop budget here; a state
            # built from the raw config could not have seen it.
            state.max_loops = preset_loops
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
                if self.state.current == AgentStateName.PLAN:
                    self._log("PLAN")
                    await self._profiled_phase("PLAN", self.plan)
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
        await self._maybe_refresh_run_spec()
        self.state.current = AgentStateName.CODE

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
                if self.config.get("workflow", {}).get("candidate_queue"):
                    messages = [*messages, self._candidate_queue_message(output_format)]
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
                failure_memory = self._format_failure_memory()
                if failure_memory:
                    add_runtime_context(
                        "Failure memory follows. Treat it as durable negative "
                        "evidence from prior candidates. Do not treat improved "
                        "diagnostics as valid unless tests also passed. Avoid "
                        "candidate variants whose failure_signature, failure_class, "
                        "or next_rule says avoid; when next_rule is repair_with_constraint, "
                        "address why_invalid before optimizing further.\n"
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
                active_todo = self._format_active_todo()
                if active_todo:
                    add_runtime_context(
                        "Active durable todo follows. Implement only this todo. "
                        "Candidate strategy_axis must exactly match the todo "
                        "strategy_axis. Candidate reason and change reasons must "
                        "preserve the todo context, stay on its family_key when one "
                        "is present, and should mention the todo_id. Todo drift is "
                        "rejected before edits or tests.\n"
                        f"{active_todo}"
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
                history = self._format_candidate_history()
                if history:
                    add_runtime_context(
                        "Recent candidate history follows. Avoid repeating rejected changes. "
                        "Preserve ideas that were accepted unless the current plan says otherwise.\n"
                        f"{history}"
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
        await self._apply_changes(decision.changes, allowed)
        self.state.current = AgentStateName.TEST

    async def _apply_changes(self, changes: list[CodeChange], allowed: set[str]) -> int:
        applied = 0
        self.state.proposed_changes = changes
        for change in changes:
            if change.path not in allowed:
                self.state.notes.append(f"Rejected out-of-plan change: {change.path}")
                continue
            if change.target is not None and change.replacement is not None:
                if await self._apply_replacement(change.path, change.target, change.replacement):
                    applied += 1
                continue
            if change.patch:
                if await self._apply_patch(change.patch, allowed):
                    applied += 1
                continue
            if change.content is not None:
                await self.mcp.write_file(str(self.state.repo_root / change.path), change.content)
                applied += 1
                continue
            self.state.notes.append(f"Skipped empty change: {change.path}")
        self.state.scratch["applied_changes"] = applied
        return applied

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
            applied = await self._apply_changes(candidate.changes, allowed)
            current_snapshot = await self._snapshot_files(sorted(allowed))
            patch_text = self._snapshot_patch(baseline_snapshot, current_snapshot)
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
                    note_start = len(self.state.notes)
                    applied = await self._apply_changes(repaired_candidate.changes, allowed)
                    current_snapshot = await self._snapshot_files(sorted(allowed))
                    patch_text = self._snapshot_patch(baseline_snapshot, current_snapshot)
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
                    continue
                candidate = candidate_for_record
            else:
                candidate_for_record = candidate

            candidate = candidate_for_record
            if repair_parent_id:
                self.state.notes.append(
                    f"Candidate {candidate.candidate_id} is repair of {repair_parent_id}"
                )

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
        commands = self.config.get("workflow", {}).get("test_commands", [])
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

    @staticmethod
    def _candidate_queue_message(output_format: str) -> dict[str, str]:
        if output_format == "xml":
            return {
                "role": "system",
                "content": (
                    "Candidate queue mode is enabled. Output one or more <candidate> "
                    "blocks inside a single <candidates> root. Each candidate must be "
                    "independent and safe to apply from the same baseline. If a strategy "
                    "axis contract is present, include <strategy_axis>axis</strategy_axis> "
                    "inside each <candidate>."
                ),
            }
        return {
            "role": "system",
            "content": (
                "Candidate queue mode is enabled. Output strict JSON with a top-level "
                '"candidates" array, not a top-level "changes" array. Example: '
                '{"candidates":[{"id":"1","strategy_axis":"general_edit","reason":"short","changes":[{"path":"file.py",'
                '"target":"exact text","replacement":"new text","reason":"short"}]}]}. '
                "Each candidate must be independent and safe to apply from the same baseline."
            ),
        }

    async def test(self) -> None:
        self.state.test_results = await self._run_test_commands()
        failed = False
        for result in self.state.test_results:
            failed = failed or result.exit_code != 0
        if self.state.scratch.get("applied_changes", 0) == 0:
            failed = True
            self.state.notes.append("No code changes were applied")
        metric_failed = self._evaluate_metric_acceptance()
        failed = failed or metric_failed
        if failed:
            await self._restore_pre_code_snapshot()
        else:
            await self._persist_current_best_state()
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

    def _writable_files(self) -> set[str]:
        workflow = self.config.get("workflow", {})
        candidates = workflow.get("writable_files") or self.state.planned_files
        writable = {str(path) for path in candidates}
        if workflow.get("allow_external_context_writes"):
            return writable
        external_paths = self._external_context_path_keys()
        return {
            path
            for path in writable
            if self._repo_path_key(path) not in external_paths
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
        if not workflow.get("metric_regex"):
            return False
        metric = self._metric_from_results(self.state.test_results)
        if metric is None:
            self.state.notes.append(f"Metric not found with regex: {workflow.get('metric_regex')}")
            return bool(workflow.get("require_metric"))
        self.state.scratch["last_metric"] = metric

        baseline = self.state.scratch.get("best_metric", workflow.get("baseline_metric"))
        if baseline is None:
            self.state.scratch["best_metric"] = metric
            self.state.notes.append(f"Recorded initial metric: {metric}")
            return False

        baseline_int = int(baseline)
        improved = self._metric_improved(metric, baseline_int)
        self.state.notes.append(f"Metric candidate={metric} baseline={baseline_int} improved={improved}")
        if improved:
            self.state.scratch["best_metric"] = metric
            self.state.scratch["metric_improved"] = True
            return False
        return bool(workflow.get("accept_if_improved"))

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

    async def _apply_replacement(self, path: str, target: str, replacement: str) -> bool:
        abs_path = self.state.repo_root / path
        if target == replacement:
            self.state.notes.append(f"Replacement is a no-op: {path}")
            return False
        if self._without_comment_lines(target) == self._without_comment_lines(replacement):
            self.state.notes.append(f"Replacement only changes comments or blank lines: {path}")
            return False
        original = await self.mcp.read_file(str(abs_path))
        if target not in original:
            self.state.notes.append(f"Replacement target not found: {path}")
            return False
        if original.count(target) != 1:
            self.state.notes.append(f"Replacement target is ambiguous: {path}")
            return False
        await self.mcp.write_file(str(abs_path), original.replace(target, replacement, 1))
        return True

    @staticmethod
    def _without_comment_lines(text: str) -> str:
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(line)
        return "\n".join(lines)

    async def _apply_patch(self, patch: str, allowed: set[str]) -> bool:
        touched_files = self._patch_touched_files(patch)
        if not touched_files:
            self.state.notes.append("Patch rejected: no changed files detected")
            return False
        rejected_files = sorted(path for path in touched_files if path not in allowed)
        if rejected_files:
            self.state.notes.append(
                "Patch rejected: touches out-of-plan files: "
                + ", ".join(rejected_files[:8])
            )
            return False
        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as handle:
            handle.write(patch)
            patch_path = handle.name
        patch_arg = shlex.quote(patch_path)
        try:
            result = await self.mcp.run_command(
                f"git apply --check {patch_arg}", cwd=str(self.state.repo_root)
            )
            if result["exit_code"] != 0:
                self.state.notes.append(f"Patch rejected: {result['stderr'][-1000:]}")
                return False
            result = await self.mcp.run_command(
                f"git apply {patch_arg}", cwd=str(self.state.repo_root)
            )
            if result["exit_code"] != 0:
                self.state.notes.append(f"Patch apply failed: {result['stderr'][-1000:]}")
                return False
            return True
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
