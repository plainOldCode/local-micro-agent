from __future__ import annotations

import argparse
import ast
import asyncio
import difflib
import hashlib
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from .mcp_client import McpServerSpec, McpToolClient
from .models import ModelManager
from .prompts import (
    PROMPT_MARKDOWN,
    brainstorm_prompt,
    code_prompt,
    plan_prompt,
    read_prompt,
    reflect_prompt,
    test_prompt,
)
from .state import AgentState, AgentStateName, CodeChange, FileSnapshot, TestResult
from .validators import (
    JsonValidationError,
    parse_json_object,
    parse_xml_candidates,
    require_keys,
    retry_repair_prompt,
)


class ReadDecision:
    def __init__(self, files: list[str], reason: str = ""):
        self.files = files
        self.reason = reason


class CodeDecision:
    def __init__(self, changes: list[CodeChange], candidates: list["CodeCandidate"] | None = None):
        self.changes = changes
        self.candidates = candidates or [CodeCandidate("1", changes, "single candidate")]


class CodeCandidate:
    def __init__(
        self,
        candidate_id: str,
        changes: list[CodeChange],
        reason: str = "",
        strategy_axis: str = "",
    ):
        self.candidate_id = candidate_id
        self.changes = changes
        self.reason = reason
        self.strategy_axis = strategy_axis


class TestDecision:
    def __init__(self, status: str, reason: str = "", next_focus: str = ""):
        self.status = status
        self.reason = reason
        self.next_focus = next_focus


class MicroAgent:
    def __init__(self, config: dict[str, Any], state: AgentState):
        self.config = config
        self.state = state
        self.models = ModelManager(config)
        self.mcp = McpToolClient(
            {
                name: McpServerSpec(command=spec["command"], args=spec.get("args", []))
                for name, spec in config.get("mcp_servers", {}).items()
            }
        )

    async def run(self) -> AgentState:
        await self.mcp.start()
        try:
            while self.state.current not in {AgentStateName.DONE, AgentStateName.FAILED}:
                if self.state.current == AgentStateName.PLAN:
                    self._log("PLAN")
                    await self.plan()
                elif self.state.current == AgentStateName.READ:
                    self._log("READ")
                    await self.read()
                elif self.state.current == AgentStateName.REFLECT:
                    self._log(f"REFLECT loop={self.state.loop_count}")
                    await self.reflect()
                elif self.state.current == AgentStateName.CODE:
                    self._log(f"CODE loop={self.state.loop_count}")
                    await self.code()
                elif self.state.current == AgentStateName.TEST:
                    self._log(f"TEST loop={self.state.loop_count}")
                    await self.test()
                else:
                    self.state.current = AgentStateName.FAILED
        finally:
            await self.mcp.close()
        return self.state

    async def plan(self) -> None:
        seeded_plan = self.config.get("workflow", {}).get("plan_markdown")
        if seeded_plan:
            self.state.plan_markdown = seeded_plan.strip()
            self.state.current = AgentStateName.READ
            return

        project_context = await self._load_project_context()
        workflow_context = self._workflow_plan_context()
        if workflow_context:
            project_context = "\n\n".join(part for part in [project_context, workflow_context] if part)
        output = await self.models.get("planner").chat(plan_prompt(self.state, project_context))
        self.state.plan_markdown = output.strip()
        self.state.current = AgentStateName.READ

    async def read(self) -> None:
        seeded_files = self.config.get("workflow", {}).get("seed_files")
        if seeded_files is not None:
            decision = ReadDecision(files=seeded_files, reason="seeded by workflow config")
        else:
            decision = await self._json_call("planner", read_prompt(self.state), ReadDecision)
        self.state.planned_files = decision.files
        self.state.file_context = []
        for rel_path in decision.files:
            abs_path = self.state.repo_root / rel_path
            content = await self.mcp.read_file(str(abs_path))
            content = self._context_for_file(rel_path, content)
            self.state.file_context.append(FileSnapshot(path=rel_path, content=content))
        self.state.current = AgentStateName.CODE

    async def reflect(self) -> None:
        if self._should_brainstorm():
            await self._brainstorm()
            self.state.current = AgentStateName.CODE
            return
        feedback_notes_limit = int(
            self.config.get("workflow", {}).get("feedback_notes_limit", 12)
        )
        try:
            output = await self.models.get("reflector").chat(
                reflect_prompt(self.state, feedback_notes_limit)
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

    async def _brainstorm(self) -> None:
        reject_summary = self._format_recent_reject_summary()
        feedback_notes_limit = int(
            self.config.get("workflow", {}).get("brainstorm_feedback_notes_limit", 8)
        )
        try:
            output = await self.models.get("brainstorm").chat(
                brainstorm_prompt(
                    self.state,
                    reject_summary=reject_summary,
                    cooled_axes=self._current_cooled_axes(),
                    feedback_notes_limit=feedback_notes_limit,
                )
            )
        except Exception as exc:
            self.state.notes.append(
                f"Brainstorm model call failed: {type(exc).__name__}: {exc}"
            )
            return
        brainstorm = output.strip()
        if brainstorm:
            self.state.scratch["tactic_library"] = brainstorm
            self.state.scratch["last_brainstorm_loop"] = self.state.loop_count
            self.state.notes.append("Brainstorm tactics added for next CODE attempt")

    async def code(self) -> None:
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
                messages = code_prompt(self.state, feedback_notes_limit, output_format)
                if self.config.get("workflow", {}).get("candidate_queue"):
                    messages = [*messages, self._candidate_queue_message(output_format)]
                axis_contract = self._format_axis_contract()
                if axis_contract:
                    messages = [
                        *messages,
                        {
                            "role": "system",
                            "content": (
                                "Strategy axis contract follows. Candidate output must obey it. "
                                "A candidate with a missing, unknown, cooled, or wrong strategy_axis "
                                "will be rejected before edits or tests.\n"
                                f"{axis_contract}"
                            ),
                        },
                    ]
                search_memory = self._format_adaptive_search_memory()
                if search_memory:
                    messages = [
                        *messages,
                        {
                            "role": "system",
                            "content": (
                                "Adaptive search memory follows. Use it to allocate search budget. "
                                "Do not repeat cooled-down strategy axes unless the user request "
                                "explicitly requires them; prefer under-explored axes and explain "
                                "the chosen axis in the candidate reason.\n"
                                f"{search_memory}"
                            ),
                        },
                    ]
                tactic_library = self._format_tactic_library()
                if tactic_library:
                    messages = [
                        *messages,
                        {
                            "role": "system",
                            "content": (
                                "Stagnation brainstorm tactics follow. Prefer one tactic that "
                                "matches the required strategy axis and has not been rejected.\n"
                                f"{tactic_library}"
                            ),
                        },
                    ]
                history = self._format_candidate_history()
                if history:
                    messages = [
                        *messages,
                        {
                            "role": "system",
                            "content": (
                                "Recent candidate history follows. Avoid repeating rejected changes. "
                                "Preserve ideas that were accepted unless the current plan says otherwise.\n"
                                f"{history}"
                            ),
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
                if await self._apply_patch(change.patch):
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
            axis_rejection = self._candidate_axis_contract_rejection(candidate)
            if axis_rejection is not None:
                status, note = axis_rejection
                self.state.notes.append(f"Candidate {candidate.candidate_id} rejected: {note}")
                self._append_candidate_history(
                    candidate,
                    status=status,
                    metric=None,
                    applied=0,
                    failed=True,
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
                self._append_candidate_history(
                    candidate,
                    status="rejected_repeated_pattern",
                    metric=None,
                    applied=0,
                    failed=True,
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
                self._append_candidate_history(
                    candidate,
                    status="rejected_cooled_axis",
                    metric=None,
                    applied=0,
                    failed=True,
                )
                self._record_strategy_attempt(
                    candidate,
                    status="rejected_cooled_axis",
                    metric=None,
                    applied=0,
                    failed=True,
                )
                continue

            await self._restore_snapshot(baseline_snapshot)
            applied = await self._apply_changes(candidate.changes, allowed)
            if applied == 0:
                self.state.notes.append(f"Candidate {candidate.candidate_id} rejected: no changes applied")
                self._remember_rejected_candidate(candidate)
                self._append_candidate_history(
                    candidate,
                    status="rejected_no_changes",
                    metric=None,
                    applied=0,
                    failed=True,
                )
                self._record_strategy_attempt(
                    candidate,
                    status="rejected_no_changes",
                    metric=None,
                    applied=0,
                    failed=True,
                )
                continue

            results = await self._run_test_commands()
            failed = any(result.exit_code != 0 for result in results)
            metric = self._metric_from_results(results)
            if metric is None:
                failed = failed or bool(workflow.get("require_metric"))
                self.state.notes.append(
                    f"Candidate {candidate.candidate_id} metric not found"
                )
            improved = metric is not None and self._metric_improved(metric, iteration_best_metric)
            self.state.notes.append(
                f"Candidate {candidate.candidate_id} applied={applied} "
                f"metric={metric} failed={failed} improved={improved}"
            )
            self._append_candidate_history(
                candidate,
                status="improved" if improved and not failed else "rejected",
                metric=metric,
                applied=applied,
                failed=failed,
            )
            self._record_strategy_attempt(
                candidate,
                status="improved" if improved and not failed else "rejected",
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

    def _candidate_novelty_gate_enabled(self) -> bool:
        return bool(self.config.get("workflow", {}).get("candidate_novelty_gate"))

    def _adaptive_search_memory_enabled(self) -> bool:
        return bool(self.config.get("workflow", {}).get("adaptive_search_memory"))

    def _adaptive_search_reject_cooled_axes_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("adaptive_search_reject_cooled_axes")) and (
            self._adaptive_search_memory_enabled()
        )

    def _rejected_candidate_fingerprint(self, candidate: CodeCandidate) -> str | None:
        if not self._candidate_novelty_gate_enabled():
            return None
        fingerprint = self._candidate_fingerprint(candidate)
        seen = self.state.scratch.setdefault("rejected_candidate_fingerprints", [])
        if fingerprint in seen:
            return fingerprint
        return None

    def _remember_rejected_candidate(self, candidate: CodeCandidate) -> None:
        if not self._candidate_novelty_gate_enabled():
            return
        fingerprint = self._candidate_fingerprint(candidate)
        seen = self.state.scratch.setdefault("rejected_candidate_fingerprints", [])
        if fingerprint not in seen:
            seen.append(fingerprint)

    def _candidate_fingerprint(self, candidate: CodeCandidate) -> str:
        payload = {
            "reason": self._normalize_fingerprint_text(candidate.reason),
            "strategy_axis": self._normalize_fingerprint_text(candidate.strategy_axis),
            "changes": [
                {
                    "path": change.path,
                    "reason": self._normalize_fingerprint_text(change.reason),
                    "target": self._normalize_fingerprint_text(change.target or ""),
                    "replacement": self._normalize_fingerprint_text(change.replacement or ""),
                    "patch": self._normalize_fingerprint_text(change.patch or ""),
                    "content": self._normalize_fingerprint_text(change.content or ""),
                }
                for change in candidate.changes
            ],
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _candidate_strategy_axes(self, candidate: CodeCandidate) -> list[str]:
        declared = self._normalize_strategy_axis(candidate.strategy_axis)
        reason_axes = self._candidate_reason_strategy_axes(candidate)
        axes = [] if reason_axes == ["general_edit"] else list(reason_axes)
        code_text = self._normalize_fingerprint_text(
            "\n".join(
                part
                for change in candidate.changes
                for part in (
                    change.path,
                    change.target or "",
                    change.replacement or "",
                    change.patch or "",
                    change.content or "",
                )
            )
        )
        if not axes:
            axes = self._strategy_axes_for_text(code_text, self._strategy_axis_keywords())
        if declared and declared in self._strategy_axis_pool() and declared not in axes:
            axes.append(declared)
        if not axes:
            axes = ["general_edit"]
        return sorted(set(axes))

    def _candidate_reason_strategy_axes(self, candidate: CodeCandidate) -> list[str]:
        reason_parts = [candidate.reason, *(change.reason for change in candidate.changes)]
        reason_text = self._normalize_fingerprint_text("\n".join(reason_parts))
        axes = self._strategy_axes_for_text(reason_text, self._strategy_axis_keywords())
        return axes or ["general_edit"]

    @staticmethod
    def _strategy_axis_keywords() -> dict[str, tuple[str, ...]]:
        keyword_axes = {
            "hash_build": ("hash", "checksum", "digest", "build_hash"),
            "phase_interleave": ("phase", "stage", "interleave", "pipeline", "round"),
            "vector_unroll_lane": ("unroll", "vector", "simd", "lane", "parallel"),
            "memory_store_layout": (
                "address",
                "register",
                "spill",
                "store",
                "write",
                "buffer",
                "cache",
                "layout",
                "memory",
            ),
            "precompute_constants": ("precompute", "lookup", "table", "constant", "fold"),
            "branch_control": ("branch", "condition", "guard", "switch", "flow", "select", "bounds"),
            "instruction_scheduling": ("bundle", "slot", "hazard", "dependency", "dependent", "raw"),
            "parsing": ("parse", "parser", "regex", "xml", "json"),
            "api_contract": ("api", "interface", "signature", "schema", "contract"),
            "test_contract": ("test", "assert", "fixture", "threshold"),
            "runtime_control": ("timeout", "async", "process", "subprocess", "retry"),
        }
        return keyword_axes

    @staticmethod
    def _strategy_axes_for_text(
        text: str, keyword_axes: dict[str, tuple[str, ...]]
    ) -> list[str]:
        return [
            axis
            for axis, keywords in keyword_axes.items()
            if any(keyword in text for keyword in keywords)
        ]

    def _format_axis_contract(self) -> str:
        if not self._axis_contract_enabled():
            self.state.scratch.pop("required_strategy_axis", None)
            return ""
        required_axis = self._select_required_strategy_axis()
        self.state.scratch["required_strategy_axis"] = required_axis
        cooled_axes = self._current_cooled_axes()
        payload = {
            "required_strategy_axis": required_axis,
            "allowed_strategy_axes": self._allowed_strategy_axes(),
            "cooled_strategy_axes": cooled_axes,
            "known_strategy_axes": self._strategy_axis_pool(),
            "required_axis_guidance": self._strategy_axis_guidance(required_axis),
            "output_requirement": (
                "Set candidate strategy_axis exactly to required_strategy_axis. "
                "In XML mode include <strategy_axis>axis</strategy_axis> inside each "
                "<candidate>. Candidate reason and change reasons must substantively "
                "target required_strategy_axis, or the candidate is rejected as axis drift."
            ),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _strategy_axis_guidance(axis: str) -> dict[str, Any]:
        guidance = {
            "hash_build": {
                "focus": "Change how hash-stage instructions are generated or packed.",
                "try": [
                    "separate independent hash operands from dependent combine ops",
                    "reorder hash-stage temporaries to reduce RAW dependency stalls",
                    "change hash emission shape without changing surrounding phase structure",
                ],
                "avoid_drift": [
                    "store address reuse",
                    "loop unroll changes",
                    "branch/select rewrites",
                ],
            },
            "precompute_constants": {
                "focus": "Move repeated constant/scratch lookup work into reusable locals or tables.",
                "try": [
                    "cache repeated scratch_const results used in tight emission loops",
                    "prebuild per-round or per-stage constants before the hot loop",
                    "replace repeated literal lookup calls with indexed local arrays",
                ],
                "avoid_drift": [
                    "hash operation reorder",
                    "phase interleaving",
                    "store layout rewrites",
                ],
            },
            "branch_control": {
                "focus": "Reduce flow/select/control instructions or make control cheaper.",
                "try": [
                    "replace flow select with ALU mask/arithmetic when correctness is identical",
                    "combine guard computation with existing ALU work",
                    "remove redundant bounds or condition checks",
                ],
                "avoid_drift": [
                    "hash-stage scheduling",
                    "store address movement",
                    "unroll factor changes",
                ],
            },
            "phase_interleave": {
                "focus": "Change ordering between existing phases without changing the algorithm.",
                "try": [
                    "move independent work from adjacent phases together",
                    "split a phase into smaller chunks to improve engine-slot mixing",
                    "interleave only one narrow phase boundary at a time",
                ],
                "avoid_drift": [
                    "new hash algorithm",
                    "constant precompute only",
                    "branch/select-only edits",
                ],
            },
            "vector_unroll_lane": {
                "focus": "Change per-lane or unroll-lane structure.",
                "try": [
                    "change lane-local temporary reuse",
                    "alter lane order or grouping inside the current unroll factor",
                    "specialize first or last lane handling if it removes work",
                ],
                "avoid_drift": [
                    "global phase rewrite",
                    "hash-stage-only scheduling",
                    "branch/select-only edits",
                ],
            },
            "memory_store_layout": {
                "focus": "Change address, store, scratch, or memory layout work.",
                "try": [
                    "reuse already-computed store addresses when lifetime is valid",
                    "move address computation away from store bottlenecks",
                    "reduce scratch/register pressure around memory writes",
                ],
                "avoid_drift": [
                    "hash operation reorder",
                    "branch/select-only rewrites",
                    "generic loop restructuring",
                ],
            },
            "instruction_scheduling": {
                "focus": "Change instruction order to reduce bundle, slot, or dependency stalls.",
                "try": [
                    "separate producer and consumer instructions with independent work",
                    "mix engine types while preserving data dependencies",
                    "schedule one dependency chain locally instead of rewriting the whole loop",
                ],
                "avoid_drift": [
                    "new algorithm",
                    "constant caching only",
                    "memory layout only",
                ],
            },
            "general_edit": {
                "focus": "Make a small novel edit that does not fit a cooled specialist axis.",
                "try": [
                    "remove dead or duplicate work",
                    "simplify a local invariant",
                    "make one correctness-preserving local cleanup with measurable effect",
                ],
                "avoid_drift": [
                    "hidden hash/phase/vector/memory rewrite",
                    "repeating any cooled axis under a generic label",
                ],
            },
        }
        return guidance.get(
            axis,
            {
                "focus": f"Make a candidate centered on {axis}.",
                "try": ["choose one small concrete tactic for this axis"],
                "avoid_drift": ["renaming another strategy as this axis"],
            },
        )

    def _axis_contract_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("adaptive_search_force_strategy_axis")) and (
            self._adaptive_search_memory_enabled()
        )

    def _strategy_axis_pool(self) -> list[str]:
        workflow = self.config.get("workflow", {})
        configured = workflow.get("adaptive_search_axis_pool")
        if isinstance(configured, list) and configured:
            return [self._normalize_strategy_axis(str(axis)) for axis in configured if str(axis)]
        return [
            "hash_build",
            "phase_interleave",
            "vector_unroll_lane",
            "memory_store_layout",
            "precompute_constants",
            "branch_control",
            "instruction_scheduling",
            "parsing",
            "api_contract",
            "test_contract",
            "runtime_control",
            "general_edit",
        ]

    def _allowed_strategy_axes(self) -> list[str]:
        cooled = set(self._current_cooled_axes())
        return [axis for axis in self._strategy_axis_pool() if axis not in cooled]

    def _select_required_strategy_axis(self) -> str:
        allowed = self._allowed_strategy_axes()
        if not allowed:
            return "general_edit"
        memory = self.state.scratch.get("adaptive_search_memory")
        if not isinstance(memory, dict):
            memory = self._adaptive_search_memory_from_history()
            if memory:
                self.state.scratch["adaptive_search_memory"] = memory
        axes_state = memory.get("axes", {}) if isinstance(memory, dict) else {}
        axis_order = {axis: index for index, axis in enumerate(allowed)}

        def score(axis: str) -> tuple[int, int, int]:
            raw = axes_state.get(axis) if isinstance(axes_state, dict) else None
            if not isinstance(raw, dict):
                return (0, 0, axis_order.get(axis, 0))
            return (
                int(raw.get("attempts", 0)),
                int(raw.get("failures", 0)),
                axis_order.get(axis, 0),
            )

        return sorted(allowed, key=score)[0]

    def _current_cooled_axes(self) -> list[str]:
        memory = self.state.scratch.get("adaptive_search_memory")
        if not isinstance(memory, dict):
            memory = self._adaptive_search_memory_from_history()
            if memory:
                self.state.scratch["adaptive_search_memory"] = memory
        if not isinstance(memory, dict):
            return []
        axes_state = memory.get("axes")
        if not isinstance(axes_state, dict):
            return []
        current_loop = self.state.loop_count
        cooled = []
        for axis, raw_state in axes_state.items():
            if not isinstance(raw_state, dict):
                continue
            cooldown_until = raw_state.get("cooldown_until_loop")
            if isinstance(cooldown_until, int) and cooldown_until > current_loop:
                cooled.append(str(axis))
        return sorted(cooled)

    def _candidate_axis_contract_rejection(
        self, candidate: CodeCandidate
    ) -> tuple[str, str] | None:
        if not self._axis_contract_enabled():
            return None
        declared = self._normalize_strategy_axis(candidate.strategy_axis)
        if not declared:
            return ("rejected_missing_axis", "missing strategy_axis")
        if declared not in self._strategy_axis_pool():
            return ("rejected_unknown_axis", f"unknown strategy_axis {declared}")
        required = self.state.scratch.get("required_strategy_axis")
        if isinstance(required, str) and required and declared != required:
            return (
                "rejected_wrong_axis",
                f"strategy_axis {declared} does not match required {required}",
            )
        if isinstance(required, str) and required:
            reason_axes = self._candidate_reason_strategy_axes(candidate)
            if required == "general_edit":
                if reason_axes != ["general_edit"]:
                    return (
                        "rejected_axis_drift",
                        "candidate reason targets "
                        f"{', '.join(reason_axes)} instead of required general_edit",
                    )
            elif required not in reason_axes:
                return (
                    "rejected_axis_drift",
                    "candidate reason does not substantively target required "
                    f"strategy_axis {required}",
                )
        if declared in self._current_cooled_axes():
            return ("rejected_cooled_axis", f"cooled strategy_axis {declared}")
        return None

    @staticmethod
    def _normalize_strategy_axis(axis: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", axis.strip().lower()).strip("_")

    def _cooled_candidate_axes(self, candidate: CodeCandidate) -> list[str]:
        if not self._adaptive_search_reject_cooled_axes_enabled():
            return []
        memory = self.state.scratch.get("adaptive_search_memory")
        if not isinstance(memory, dict):
            memory = self._adaptive_search_memory_from_history()
            if memory:
                self.state.scratch["adaptive_search_memory"] = memory
        if not isinstance(memory, dict):
            return []
        axes_state = memory.get("axes")
        if not isinstance(axes_state, dict):
            return []
        current_loop = self.state.loop_count
        cooled = []
        for axis in self._candidate_strategy_axes(candidate):
            axis_state = axes_state.get(axis)
            if not isinstance(axis_state, dict):
                continue
            cooldown_until = axis_state.get("cooldown_until_loop")
            if isinstance(cooldown_until, int) and cooldown_until > current_loop:
                cooled.append(axis)
        return cooled

    def _record_strategy_attempt(
        self,
        candidate: CodeCandidate,
        status: str,
        metric: int | None,
        applied: int,
        failed: bool,
    ) -> None:
        if not self._adaptive_search_memory_enabled():
            return
        axes = self._candidate_strategy_axes(candidate)
        memory = self.state.scratch.setdefault(
            "adaptive_search_memory",
            {"axes": {}, "recent": []},
        )
        if not isinstance(memory, dict):
            memory = {"axes": {}, "recent": []}
            self.state.scratch["adaptive_search_memory"] = memory
        axes_state = memory.setdefault("axes", {})
        recent = memory.setdefault("recent", [])
        improved = status in {"improved", "accepted"} and not failed
        for axis in axes:
            axis_state = axes_state.setdefault(
                axis,
                {
                    "attempts": 0,
                    "failures": 0,
                    "successes": 0,
                    "cooldown_until_loop": None,
                    "last_status": None,
                    "last_metric": None,
                    "best_metric": None,
                },
            )
            axis_state["attempts"] = int(axis_state.get("attempts", 0)) + 1
            axis_state["last_status"] = status
            axis_state["last_metric"] = metric
            if improved:
                axis_state["successes"] = int(axis_state.get("successes", 0)) + 1
                axis_state["cooldown_until_loop"] = None
                best_metric = axis_state.get("best_metric")
                if metric is not None and (
                    best_metric is None or self._metric_improved(metric, int(best_metric))
                ):
                    axis_state["best_metric"] = metric
            else:
                axis_state["failures"] = int(axis_state.get("failures", 0)) + 1
                if self._axis_should_cool_down(axis, status):
                    cooldown = int(
                        self.config.get("workflow", {}).get(
                            "adaptive_search_axis_cooldown_loops", 3
                        )
                    )
                    axis_state["cooldown_until_loop"] = self.state.loop_count + cooldown
        recent.append(
            {
                "loop": self.state.loop_count,
                "candidate_id": candidate.candidate_id,
                "axes": axes,
                "status": status,
                "metric": metric,
                "applied": applied,
                "failed": failed,
            }
        )
        limit = int(self.config.get("workflow", {}).get("adaptive_search_recent_limit", 20))
        if len(recent) > limit:
            del recent[:-limit]

    def _axis_should_cool_down(self, axis: str, status: str) -> bool:
        memory = self.state.scratch.get("adaptive_search_memory")
        if not isinstance(memory, dict):
            return False
        recent = memory.get("recent")
        if not isinstance(recent, list):
            return False
        window = int(self.config.get("workflow", {}).get("adaptive_search_axis_window", 8))
        threshold = int(
            self.config.get("workflow", {}).get("adaptive_search_axis_failure_threshold", 3)
        )
        failure_statuses = {
            "rejected",
            "rejected_cooled_axis",
            "rejected_missing_axis",
            "rejected_no_changes",
            "rejected_repeated_pattern",
            "rejected_unknown_axis",
            "rejected_wrong_axis",
            "rejected_no_metric",
        }
        if status not in failure_statuses:
            return False
        recent_failures = 1
        for record in reversed(recent[-window:]):
            if axis not in record.get("axes", []):
                continue
            if record.get("status") in failure_statuses or record.get("failed") is True:
                recent_failures += 1
        return recent_failures >= threshold

    def _format_adaptive_search_memory(self) -> str:
        if not self._adaptive_search_memory_enabled():
            return ""
        memory = self.state.scratch.get("adaptive_search_memory")
        if not isinstance(memory, dict):
            memory = self._adaptive_search_memory_from_history()
            if memory:
                self.state.scratch["adaptive_search_memory"] = memory
        if not isinstance(memory, dict):
            return ""
        axes_state = memory.get("axes")
        if not isinstance(axes_state, dict) or not axes_state:
            return ""
        current_loop = self.state.loop_count
        axes = []
        cooled_down = []
        for axis, raw_state in sorted(axes_state.items()):
            if not isinstance(raw_state, dict):
                continue
            cooldown_until = raw_state.get("cooldown_until_loop")
            is_cooled = isinstance(cooldown_until, int) and cooldown_until > current_loop
            item = {
                "axis": axis,
                "attempts": raw_state.get("attempts", 0),
                "failures": raw_state.get("failures", 0),
                "successes": raw_state.get("successes", 0),
                "last_status": raw_state.get("last_status"),
                "last_metric": raw_state.get("last_metric"),
                "best_metric": raw_state.get("best_metric"),
            }
            if is_cooled:
                item["cooldown_until_loop"] = cooldown_until
                cooled_down.append(axis)
            axes.append(item)
        recent = memory.get("recent") if isinstance(memory.get("recent"), list) else []
        payload = {
            "current_loop": current_loop,
            "cooled_down_axes": cooled_down,
            "axes": axes,
            "recent": recent[-5:],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _adaptive_search_memory_from_history(self) -> dict[str, Any] | None:
        path = self._candidate_history_path()
        if path is None or not path.exists():
            return None
        limit = int(self.config.get("workflow", {}).get("candidate_history_limit", 20))
        lines = path.read_text(errors="replace").splitlines()[-limit:]
        memory: dict[str, Any] = {"axes": {}, "recent": []}
        failure_statuses = {
            "rejected",
            "rejected_cooled_axis",
            "rejected_missing_axis",
            "rejected_no_changes",
            "rejected_repeated_pattern",
            "rejected_unknown_axis",
            "rejected_wrong_axis",
            "rejected_no_metric",
        }
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            axes = record.get("strategy_axes")
            if not isinstance(axes, list) or not axes:
                continue
            status = str(record.get("status", ""))
            failed = bool(record.get("failed"))
            metric = record.get("metric")
            recent_record = {
                "loop": record.get("loop"),
                "candidate_id": record.get("candidate_id"),
                "axes": [str(axis) for axis in axes],
                "status": status,
                "metric": metric,
                "applied": record.get("applied", 0),
                "failed": failed,
            }
            memory["recent"].append(recent_record)
            for axis in recent_record["axes"]:
                axis_state = memory["axes"].setdefault(
                    axis,
                    {
                        "attempts": 0,
                        "failures": 0,
                        "successes": 0,
                        "cooldown_until_loop": None,
                        "last_status": None,
                        "last_metric": None,
                        "best_metric": None,
                    },
                )
                axis_state["attempts"] += 1
                axis_state["last_status"] = status
                axis_state["last_metric"] = metric
                if status in {"improved", "accepted"} and not failed:
                    axis_state["successes"] += 1
                    best_metric = axis_state.get("best_metric")
                    if isinstance(metric, int) and (
                        best_metric is None or self._metric_improved(metric, int(best_metric))
                    ):
                        axis_state["best_metric"] = metric
                elif status in failure_statuses or failed:
                    axis_state["failures"] += 1
        self._apply_history_cooldowns(memory, failure_statuses)
        return memory if memory["axes"] else None

    def _apply_history_cooldowns(
        self, memory: dict[str, Any], failure_statuses: set[str]
    ) -> None:
        recent = memory.get("recent")
        axes_state = memory.get("axes")
        if not isinstance(recent, list) or not isinstance(axes_state, dict):
            return
        window = int(self.config.get("workflow", {}).get("adaptive_search_axis_window", 8))
        threshold = int(
            self.config.get("workflow", {}).get("adaptive_search_axis_failure_threshold", 3)
        )
        cooldown = int(
            self.config.get("workflow", {}).get("adaptive_search_axis_cooldown_loops", 3)
        )
        for axis, axis_state in axes_state.items():
            recent_failures = 0
            for record in reversed(recent[-window:]):
                if axis not in record.get("axes", []):
                    continue
                if record.get("status") in failure_statuses or record.get("failed") is True:
                    recent_failures += 1
            if recent_failures >= threshold:
                axis_state["cooldown_until_loop"] = self.state.loop_count + cooldown

    @staticmethod
    def _normalize_fingerprint_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip()).lower()

    async def _run_test_commands(self) -> list[TestResult]:
        commands = self.config.get("workflow", {}).get("test_commands", [])
        workflow = self.config.get("workflow", {})
        results = []
        for command in commands:
            result = await self.mcp.run_command(
                command,
                cwd=str(self.state.repo_root),
                timeout_seconds=workflow.get("command_timeout_seconds", 120),
                output_limit=workflow.get("command_output_limit", 200_000),
            )
            results.append(TestResult(**result))
        return results

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
                self.state.current = (
                    AgentStateName.REFLECT
                    if self.config.get("workflow", {}).get("reflect_before_retry")
                    else AgentStateName.CODE
                )
                return
            if not failed and self._should_continue_after_improvement():
                self.state.loop_count += 1
                self.state.current = AgentStateName.CODE
                self.state.notes.append(
                    f"Continuing after improvement with baseline={self.state.scratch.get('best_metric')}"
                )
                return
            self.state.current = AgentStateName.FAILED if failed else AgentStateName.DONE
            return

        decision = await self._json_call("tester", test_prompt(self.state), TestDecision)
        if not failed and decision.status == "pass":
            if self._should_continue_after_improvement():
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
        self.state.current = (
            AgentStateName.REFLECT
            if self.config.get("workflow", {}).get("reflect_before_retry")
            else AgentStateName.CODE
        )

    def _writable_files(self) -> set[str]:
        workflow = self.config.get("workflow", {})
        return set(workflow.get("writable_files") or self.state.planned_files)

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

    def _should_continue_after_improvement(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("continue_after_improvement")) and (
            self.state.scratch.get("metric_improved") is True
            and self.state.loop_count + 1 < self.state.max_loops
        )

    def _should_brainstorm(self) -> bool:
        workflow = self.config.get("workflow", {})
        threshold = int(workflow.get("brainstorm_after_rejections", 0) or 0)
        if threshold <= 0:
            return False
        if self.state.scratch.get("last_brainstorm_loop") == self.state.loop_count:
            return False
        records = self._candidate_history_records(limit=max(threshold, 1))
        if len(records) < threshold:
            return False
        return all(str(record.get("status", "")).startswith("rejected") for record in records)

    def _format_tactic_library(self) -> str:
        tactic_library = self.state.scratch.get("tactic_library")
        if not isinstance(tactic_library, str) or not tactic_library.strip():
            return ""
        return tactic_library.strip()

    def _format_recent_reject_summary(self) -> str:
        limit = int(self.config.get("workflow", {}).get("brainstorm_reject_summary_limit", 8))
        records = self._candidate_history_records(limit=limit)
        if not records:
            return "No candidate history yet."
        summary = []
        for record in records:
            status = record.get("status")
            axis = record.get("strategy_axis") or ""
            axes = record.get("strategy_axes") or []
            reason = str(record.get("reason") or "")[:240]
            metric = record.get("metric")
            summary.append(
                {
                    "status": status,
                    "metric": metric,
                    "strategy_axis": axis,
                    "strategy_axes": axes,
                    "reason": reason,
                }
            )
        return json.dumps(summary, ensure_ascii=False, indent=2)

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
            "notes_tail": self.state.notes[-12:],
        }
        state_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        patch_path.write_text(patch_text)
        self.state.notes.append(f"Persisted best state: {state_path}")

    def _workflow_artifact_path(self, key: str, default: str) -> Path:
        raw = self.config.get("workflow", {}).get(key, default)
        path = Path(str(raw))
        if path.is_absolute():
            return path
        return self.state.repo_root / path

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

    async def _load_project_context(self) -> str:
        files = self._project_context_files()
        if not files:
            return ""
        limit = int(self.config.get("workflow", {}).get("project_context_char_limit", 12000))
        blocks = []
        for rel_path in files:
            try:
                content = await self.mcp.read_file(str(self.state.repo_root / rel_path))
            except FileNotFoundError:
                self.state.notes.append(f"Project context file not found: {rel_path}")
                continue
            blocks.append(f"### {rel_path}\n```text\n{self._slice_text(content, limit)}\n```")
        if blocks:
            self.state.notes.append(
                "Loaded project context: " + ", ".join(files)
            )
        return "\n\n".join(blocks)

    def _project_context_files(self) -> list[str]:
        workflow = self.config.get("workflow", {})
        configured = workflow.get("project_context_files")
        if isinstance(configured, list) and configured:
            return [str(path) for path in configured]
        files = []
        instruction_files = workflow.get("project_instruction_files")
        if isinstance(instruction_files, list) and instruction_files:
            files.extend(str(path) for path in instruction_files)
        else:
            files.extend(
                name
                for name in ("AGENTS.md", "CLAUDE.md", "INSTRUCTIONS.md")
                if (self.state.repo_root / name).exists()
            )
        if workflow.get("readme_first", True) is False:
            return self._unique_existing_paths(files)
        for name in ("README.md", "Readme.md", "readme.md", "README", "README.txt"):
            if (self.state.repo_root / name).exists():
                files.append(name)
                break
        return self._unique_existing_paths(files)

    def _unique_existing_paths(self, paths: list[str]) -> list[str]:
        unique = []
        seen = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            if (self.state.repo_root / path).exists():
                unique.append(path)
        return unique

    def _workflow_plan_context(self) -> str:
        workflow = self.config.get("workflow", {})
        keys = (
            "writable_files",
            "test_commands",
            "metric_regex",
            "metric_goal",
            "baseline_metric",
            "accept_if_improved",
            "require_metric",
        )
        summary = {key: workflow[key] for key in keys if key in workflow and workflow[key] not in (None, [], "")}
        if not summary:
            return ""
        return "### Workflow constraints\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```"

    @staticmethod
    def _slice_text(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        head = limit // 2
        tail = limit - head
        return text[:head] + "\n[...truncated...]\n" + text[-tail:]

    def _context_for_file(self, rel_path: str, content: str) -> str:
        symbols_by_path = self.config.get("workflow", {}).get("context_symbols")
        if not isinstance(symbols_by_path, dict):
            return content
        symbols = symbols_by_path.get(rel_path)
        if not symbols:
            return content
        if not isinstance(symbols, list):
            self.state.notes.append(f"Ignored non-list context_symbols for {rel_path}")
            return content
        excerpt = self._extract_python_symbols(content, [str(symbol) for symbol in symbols])
        if not excerpt:
            self.state.notes.append(f"No requested context symbols found in {rel_path}")
            return content
        self.state.notes.append(
            f"Using symbol context for {rel_path}: {', '.join(str(symbol) for symbol in symbols)}"
        )
        return excerpt

    @staticmethod
    def _extract_python_symbols(content: str, symbols: list[str]) -> str:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return ""
        lines = content.splitlines(keepends=True)
        selected_ranges: list[tuple[int, int]] = []
        for symbol in symbols:
            selected = MicroAgent._find_symbol_node(tree, symbol)
            if selected is None or not hasattr(selected, "lineno") or not hasattr(selected, "end_lineno"):
                continue
            selected_ranges.append((int(selected.lineno), int(selected.end_lineno)))
        if not selected_ranges:
            return ""
        merged: list[tuple[int, int]] = []
        for start, end in sorted(selected_ranges):
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                continue
            merged.append((start, end))
        return "\n\n".join("".join(lines[start - 1 : end]).rstrip() for start, end in merged)

    @staticmethod
    def _find_symbol_node(tree: ast.AST, symbol: str) -> ast.AST | None:
        if "." in symbol:
            class_name, member_name = symbol.split(".", 1)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            if child.name == member_name:
                                return child
            return None
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == symbol:
                    return node
        return None

    def _candidate_history_path(self) -> Path | None:
        path = self.config.get("workflow", {}).get("candidate_history_path")
        if not path:
            return None
        candidate_path = Path(str(path))
        if candidate_path.is_absolute():
            return candidate_path
        return self.state.repo_root / candidate_path

    def _format_candidate_history(self) -> str:
        limit = int(self.config.get("workflow", {}).get("candidate_history_limit", 20))
        records = self._candidate_history_records(limit=limit)
        if not records:
            return ""
        return json.dumps(
            [
                {
                    "status": record.get("status"),
                    "metric": record.get("metric"),
                    "failed": record.get("failed"),
                    "strategy_axis": record.get("strategy_axis", ""),
                    "strategy_axes": record.get("strategy_axes", []),
                    "changes": record.get("changes", []),
                }
                for record in records
            ],
            ensure_ascii=False,
            indent=2,
        )

    def _candidate_history_records(self, limit: int) -> list[dict[str, Any]]:
        path = self._candidate_history_path()
        if path is None or not path.exists():
            return []
        lines = path.read_text(errors="replace").splitlines()[-limit:]
        records = []
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(record)
        return records

    def _append_candidate_history(
        self,
        candidate: CodeCandidate,
        status: str,
        metric: int | None,
        applied: int,
        failed: bool,
    ) -> None:
        path = self._candidate_history_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "loop": self.state.loop_count,
            "candidate_id": candidate.candidate_id,
            "status": status,
            "metric": metric,
            "applied": applied,
            "failed": failed,
            "reason": candidate.reason,
            "fingerprint": self._candidate_fingerprint(candidate),
            "strategy_axis": candidate.strategy_axis,
            "strategy_axes": self._candidate_strategy_axes(candidate),
            "changes": self._summarize_changes(candidate.changes),
        }
        with path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _summarize_changes(changes: list[CodeChange]) -> list[dict[str, str]]:
        summary = []
        for change in changes:
            mode = "empty"
            if change.target is not None and change.replacement is not None:
                mode = "replacement"
            elif change.patch:
                mode = "patch"
            elif change.content is not None:
                mode = "content"
            summary.append(
                {
                    "path": change.path,
                    "reason": change.reason,
                    "mode": mode,
                }
            )
        return summary

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

    async def _json_call(self, role: str, messages: list[dict[str, str]], schema: type):
        try:
            output = await self.models.get(role).chat(messages)
        except Exception as exc:
            raise JsonValidationError(
                f"{role} model call failed: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            return self._parse_decision(output, schema)
        except JsonValidationError as exc:
            self._record_raw_model_output(role, "initial", output, exc)
            try:
                repaired = await self.models.get(role).chat(retry_repair_prompt(output, exc))
            except Exception as repair_exc:
                raise JsonValidationError(
                    f"{role} repair model call failed: {type(repair_exc).__name__}: {repair_exc}"
                ) from repair_exc
            try:
                return self._parse_decision(repaired, schema)
            except JsonValidationError as repair_parse_exc:
                self._record_raw_model_output(role, "repair", repaired, repair_parse_exc)
                raise

    def _record_raw_model_output(
        self, role: str, phase: str, output: str, error: Exception
    ) -> None:
        if not self.config.get("workflow", {}).get("log_raw_model_outputs"):
            return
        root = self.state.repo_root / self.config.get("workflow", {}).get(
            "raw_model_output_dir", ".local_micro_agent/raw_model_outputs"
        )
        root.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        path = root / f"{stamp}-{role}-{phase}.txt"
        path.write_text(
            f"error: {error}\n\n--- output ---\n{output}",
            encoding="utf-8",
        )
        self.state.notes.append(f"Raw model output logged: {path.relative_to(self.state.repo_root)}")

    @staticmethod
    def _parse_decision(output: str, schema: type):
        if schema is CodeDecision and "<candidates" in output:
            data = parse_xml_candidates(output)
        else:
            data = parse_json_object(output)
        if schema is ReadDecision:
            require_keys(data, ["files"])
            return ReadDecision(files=[str(path) for path in data["files"]], reason=str(data.get("reason", "")))
        if schema is CodeDecision:
            if "candidates" in data:
                candidates = []
                for index, item in enumerate(data["candidates"], start=1):
                    if not isinstance(item, dict):
                        raise JsonValidationError("Candidate must be an object")
                    require_keys(item, ["changes"])
                    candidates.append(
                        CodeCandidate(
                            candidate_id=str(item.get("id", index)),
                            changes=[CodeChange.from_dict(change) for change in item["changes"]],
                            reason=str(item.get("reason", "")),
                            strategy_axis=str(item.get("strategy_axis", "")),
                        )
                    )
                changes = candidates[0].changes if candidates else []
                return CodeDecision(changes=changes, candidates=candidates)
            require_keys(data, ["changes"])
            return CodeDecision(changes=[CodeChange.from_dict(change) for change in data["changes"]])
        if schema is TestDecision:
            require_keys(data, ["status"])
            return TestDecision(
                status=str(data["status"]),
                reason=str(data.get("reason", "")),
                next_focus=str(data.get("next_focus", "")),
            )
        raise JsonValidationError(f"Unsupported decision schema: {schema}")

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

    async def _apply_patch(self, patch: str) -> bool:
        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as handle:
            handle.write(patch)
            patch_path = handle.name
        result = await self.mcp.run_command(f"git apply --check {patch_path}", cwd=str(self.state.repo_root))
        if result["exit_code"] != 0:
            self.state.notes.append(f"Patch rejected: {result['stderr'][-1000:]}")
            return False
        result = await self.mcp.run_command(f"git apply {patch_path}", cwd=str(self.state.repo_root))
        if result["exit_code"] != 0:
            self.state.notes.append(f"Patch apply failed: {result['stderr'][-1000:]}")
            return False
        return True

    @staticmethod
    def _log(message: str) -> None:
        print(f"[local-micro-agent] {message}", flush=True)


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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
