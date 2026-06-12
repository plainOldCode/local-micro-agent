"""Durable todo lifecycle, run-spec task graph, and structural checkpoints.

Extracted from orchestrator.py; mixed into MicroAgent.
"""
from __future__ import annotations

import ast
import copy
import fnmatch
import hashlib
import json
import re
import shlex
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from ..decisions import CodeCandidate
from ..prompts import acceptance_synth_prompt, spec_idea_prompt, spec_prompt
from ..state import AgentStateName, CodeChange, FileSnapshot, TestResult
from ..validators import parse_json_object


_SPEC_JSONL_READ_LIMIT = 1000


class TodoLifecycleMixin:
    async def _maybe_refresh_run_spec(self, force: bool = False) -> None:
        workflow = self.config.get("workflow", {})
        path = self._workflow_artifact_path(
            "run_spec_path", ".local_micro_agent/run_spec.json"
        )
        if not self._run_spec_enabled():
            self.state.scratch.pop("run_spec", None)
            return
        if workflow.get("spec_resume", True) and path.exists() and not force:
            spec = self._load_run_spec(path)
            if isinstance(spec, dict) and int(spec.get("version", 1) or 1) >= 2:
                spec = self._normalize_run_spec(spec)
                if spec:
                    self.state.scratch["run_spec"] = spec
                    self.state.notes.append(f"Resumed run spec: {path}")
                    return
        if workflow.get("run_spec_after_read") or force:
            previous_spec = self._current_run_spec_snapshot(path) if force else {}
            rewrite_target_task_id = str(
                self.state.scratch.get("spec_rewrite_target_task_id") or ""
            ).strip()
            targeted_rewrite = self._targeted_spec_rewrite_enabled(
                force,
                previous_spec,
                rewrite_target_task_id,
            )
            graph_generation_origin = str(
                self.state.scratch.get("spec_graph_generation_origin") or ""
            ).strip()
            graph_parent_id = str(
                self.state.scratch.get("spec_graph_parent_graph_id") or ""
            ).strip()
            graph_origin = (
                "targeted_design_rewrite"
                if targeted_rewrite
                else (graph_generation_origin or "spec_synth")
            )
            if not targeted_rewrite:
                self.state.scratch.pop("run_spec", None)
            focus = "\n\n".join(
                part
                for part in (
                    self._focused_read_model_context(str(workflow.get("run_spec_focus", ""))),
                    self._spec_acceptance_policy_context(),
                    self._spec_grounding_facts_context(),
                    self._spec_candidate_failure_scope_context(),
                    self._spec_design_failure_memory_context(),
                    self._spec_rewrite_portfolio_context(
                        previous_spec,
                        rewrite_target_task_id,
                    ),
                    self._spec_rewrite_focus_context(),
                    self._correct_survivor_spec_context(),
                )
                if part.strip()
            )
            idea_context = await self._maybe_build_spec_idea_context(
                focus,
                force=force,
                targeted_rewrite=targeted_rewrite,
            )
            if idea_context:
                focus = "\n\n".join(part for part in (focus, idea_context) if part.strip())
            two_call = self._spec_two_call_synthesis_enabled(force)
            role_default = (
                "spec_synth"
                if two_call and self._spec_mode_enabled() and force
                else ("reasoner" if self._spec_mode_enabled() and force else "planner")
            )
            role_key = "spec_finalize_model_role" if two_call else "run_spec_model_role"
            role = str(workflow.get(role_key) or workflow.get("run_spec_model_role", role_default))
            call_site = "spec_synth" if self._spec_mode_enabled() and force else "run_spec"
            fallback_role = str(
                workflow.get(
                    "spec_synth_fallback_model_role",
                    "coder" if self._spec_mode_enabled() and force else "",
                )
                or ""
            ).strip()
            quality_attempts = self._spec_quality_rewrite_attempts()
            quality_feedback = ""
            for quality_attempt in range(quality_attempts + 1):
                prompt_focus = "\n\n".join(
                    part for part in (focus, quality_feedback) if part.strip()
                )
                prompt = spec_prompt(self.state, focus=prompt_focus)
                if not self._consume_spec_synth_call_budget(call_site):
                    return
                spec = await self._request_run_spec_from_model(
                    role=role,
                    prompt=prompt,
                    call_site=call_site,
                    fallback_role=fallback_role,
                )
                if not spec:
                    return
                preserved_task_ids: list[str] = []
                self._apply_spec_graph_generation_metadata(
                    spec,
                    previous_spec=previous_spec,
                    origin=graph_origin,
                    parent_graph_id=graph_parent_id,
                )
                if targeted_rewrite:
                    spec, preserved_task_ids = self._merge_targeted_spec_rewrite(
                        previous_spec,
                        spec,
                        rewrite_target_task_id,
                    )
                    graph_issues = self._spec_rewrite_graph_contract_issues(
                        previous_spec,
                        spec,
                        rewrite_target_task_id,
                    )
                    if graph_issues:
                        self._append_spec_graph_candidate_event(
                            spec,
                            event="candidate_rejected",
                            status="rejected_graph_contract",
                            origin=graph_origin,
                            issues=graph_issues,
                            parent_graph_id=self._spec_graph_id(previous_spec),
                        )
                        self._reject_spec_graph_rewrite(
                            previous_spec,
                            rewrite_target_task_id,
                            graph_issues,
                        )
                        self.state.scratch.pop("spec_rewrite_focus", None)
                        self.state.scratch.pop("spec_rewrite_target_task_id", None)
                        self.state.scratch.pop("spec_graph_generation_origin", None)
                        self.state.scratch.pop("spec_graph_parent_graph_id", None)
                        return
                    spec["last_rewrite_mode"] = "targeted_design_rewrite"
                    spec["last_rewrite_target_task_id"] = rewrite_target_task_id
                    if preserved_task_ids:
                        spec["last_rewrite_preserved_task_ids"] = preserved_task_ids
                quality_report = self._spec_quality_report(
                    spec,
                    attempt=quality_attempt,
                )
                self._persist_spec_quality_report(quality_report)
                if not self._spec_quality_report_failed(quality_report):
                    self._append_spec_graph_candidate_event(
                        spec,
                        event="candidate_selected",
                        status="selected",
                        origin=graph_origin,
                        quality_report=quality_report,
                        parent_graph_id=(
                            self._spec_graph_id(previous_spec)
                            if targeted_rewrite
                            else graph_parent_id
                        ),
                    )
                    self._persist_run_spec(spec)
                    if not targeted_rewrite:
                        await self._maybe_generate_backtrackable_spec_graphs(
                            selected_spec=spec,
                            base_focus=focus,
                            role=role,
                            call_site=call_site,
                            fallback_role=fallback_role,
                            graph_origin=graph_origin,
                            parent_graph_id=graph_parent_id,
                        )
                    self.state.scratch.pop("spec_rewrite_focus", None)
                    self.state.scratch.pop("spec_rewrite_target_task_id", None)
                    self.state.scratch.pop("spec_graph_generation_origin", None)
                    self.state.scratch.pop("spec_graph_parent_graph_id", None)
                    if targeted_rewrite:
                        self._append_spec_progress_event(
                            "rewrite_merged",
                            spec,
                            extra={
                                "rewrite_mode": "targeted_design_rewrite",
                                "target_task_id": rewrite_target_task_id,
                                "preserved_task_ids": preserved_task_ids,
                                "runnable_tasks_after_merge": len(
                                    self._schedulable_spec_tasks(
                                        spec.get("task_graph", [])
                                    )
                                ),
                            },
                        )
                    self.state.notes.append(f"Persisted run spec: {path}")
                    return
                issue_codes = [
                    str(issue.get("code") or "")
                    for issue in quality_report.get("issues", [])
                    if isinstance(issue, dict)
                ]
                self.state.notes.append(
                    "Run spec quality gate rejected finalizer output: "
                    + ", ".join(code for code in issue_codes if code)
                )
                self._append_spec_progress_event(
                    "quality_rejected",
                    spec,
                    extra={
                        "quality_attempt": quality_attempt,
                        "quality_issue_codes": issue_codes,
                    },
                )
                self._append_spec_graph_candidate_event(
                    spec,
                    event="candidate_rejected",
                    status="rejected_quality",
                    origin=graph_origin,
                    quality_report=quality_report,
                    parent_graph_id=(
                        self._spec_graph_id(previous_spec)
                        if targeted_rewrite
                        else graph_parent_id
                    ),
                )
                if quality_attempt >= quality_attempts:
                    if self._maybe_persist_soft_fallback_spec(spec, quality_report):
                        self.state.scratch.pop("spec_rewrite_focus", None)
                        self.state.scratch.pop("spec_rewrite_target_task_id", None)
                        self.state.scratch.pop("spec_graph_generation_origin", None)
                        self.state.scratch.pop("spec_graph_parent_graph_id", None)
                        return
                    self.state.scratch.pop("spec_rewrite_focus", None)
                    self.state.scratch.pop("spec_rewrite_target_task_id", None)
                    self.state.scratch.pop("spec_graph_generation_origin", None)
                    self.state.scratch.pop("spec_graph_parent_graph_id", None)
                    self.state.notes.append(
                        "Run spec discarded: quality gate issues remain"
                    )
                    return
                quality_feedback = self._spec_quality_feedback_context(quality_report)
        if path.exists():
            spec = self._load_run_spec(path)
            if spec:
                self.state.scratch["run_spec"] = spec
                self.state.notes.append(f"Loaded run spec: {path}")

    def _maybe_persist_soft_fallback_spec(
        self,
        spec: dict[str, Any],
        quality_report: dict[str, Any],
    ) -> bool:
        if not self._spec_gate_soft_fallback_enabled():
            return False
        if self.state.loop_count > 0:
            return False
        if not isinstance(spec.get("task_graph"), list) or not spec.get("task_graph"):
            return False
        spec["quality_gate_advisory"] = {
            "status": "soft_fallback",
            "reason": "quality_gate_exhausted_before_code",
            "report": quality_report,
        }
        spec["last_quality_gate_issues"] = quality_report.get("issues", [])
        self._append_spec_graph_candidate_event(
            spec,
            event="candidate_selected",
            status="selected_soft_fallback",
            origin="quality_soft_fallback",
            quality_report=quality_report,
        )
        self._persist_run_spec(spec)
        self._append_spec_progress_event(
            "quality_soft_fallback",
            spec,
            extra={
                "reason": "quality_gate_exhausted_before_code",
                "quality_issue_codes": quality_report.get("issue_codes", []),
            },
        )
        self.state.notes.append(
            "Run spec quality gate exhausted before CODE; persisted last spec as "
            "soft fallback advisory"
        )
        return True

    async def _maybe_generate_backtrackable_spec_graphs(
        self,
        *,
        selected_spec: dict[str, Any],
        base_focus: str,
        role: str,
        call_site: str,
        fallback_role: str,
        graph_origin: str,
        parent_graph_id: str,
    ) -> None:
        if not str(graph_origin or "").startswith("reseed"):
            return
        candidate_count = self._spec_graph_candidate_count()
        if candidate_count <= 1:
            return
        selected_search = (
            selected_spec.get("search")
            if isinstance(selected_spec.get("search"), dict)
            else {}
        )
        parent_graph_id = parent_graph_id or str(
            selected_search.get("parent_graph_id") or ""
        )
        seen_signatures = {
            tuple(self._spec_graph_signature(selected_spec)),
            *self._existing_spec_graph_signature_set(),
        }
        for index in range(2, candidate_count + 1):
            diversity_context = self._spec_graph_candidate_diversity_context(
                seen_signatures,
                candidate_index=index,
                candidate_count=candidate_count,
            )
            prompt_focus = "\n\n".join(
                part for part in (base_focus, diversity_context) if part.strip()
            )
            prompt = spec_prompt(self.state, focus=prompt_focus)
            if not self._consume_spec_synth_call_budget(f"{call_site}_candidate"):
                return
            candidate = await self._request_run_spec_from_model(
                role=role,
                prompt=prompt,
                call_site=f"{call_site}_candidate",
                fallback_role=fallback_role,
            )
            if not candidate:
                return
            self._apply_spec_graph_generation_metadata(
                candidate,
                previous_spec=selected_spec,
                origin=graph_origin,
                parent_graph_id=parent_graph_id,
            )
            signature = tuple(self._spec_graph_signature(candidate))
            if signature in seen_signatures:
                self._append_spec_graph_candidate_event(
                    candidate,
                    event="candidate_rejected",
                    status="duplicate_variant",
                    origin=graph_origin,
                    issues=["duplicate graph variant"],
                    parent_graph_id=parent_graph_id,
                )
                continue
            seen_signatures.add(signature)
            quality_report = self._spec_quality_report(candidate, attempt=0)
            if self._spec_quality_report_failed(quality_report):
                self._append_spec_graph_candidate_event(
                    candidate,
                    event="candidate_rejected",
                    status="rejected_quality",
                    origin=graph_origin,
                    quality_report=quality_report,
                    parent_graph_id=parent_graph_id,
                )
                continue
            tasks = (
                candidate.get("task_graph")
                if isinstance(candidate.get("task_graph"), list)
                else []
            )
            if not self._schedulable_spec_tasks(tasks):
                self._append_spec_graph_candidate_event(
                    candidate,
                    event="candidate_rejected",
                    status="rejected_quality",
                    origin=graph_origin,
                    issues=["candidate graph has no schedulable task"],
                    parent_graph_id=parent_graph_id,
                )
                continue
            self._append_spec_graph_candidate_event(
                candidate,
                event="candidate_created",
                status="backtrackable",
                origin=graph_origin,
                quality_report=quality_report,
                parent_graph_id=parent_graph_id,
            )
            self._append_spec_progress_event(
                "graph_candidate_backtrackable",
                candidate,
                extra={
                    "candidate_index": index,
                    "candidate_count": candidate_count,
                    "parent_graph_id": parent_graph_id,
                    "graph_signature": list(signature),
                },
            )

    def _spec_graph_candidate_count(self) -> int:
        workflow = self.config.get("workflow", {})
        return int(workflow.get("spec_graph_candidate_count", 1) or 1)

    def _existing_spec_graph_signature_set(self) -> set[tuple[str, ...]]:
        signatures: set[tuple[str, ...]] = set()
        for record in self._read_spec_jsonl(
            self._spec_graph_candidates_path(),
            limit=_SPEC_JSONL_READ_LIMIT,
        ):
            signature = record.get("graph_signature")
            if isinstance(signature, list):
                normalized = tuple(str(item) for item in signature)
                if normalized:
                    signatures.add(normalized)
        return signatures

    @staticmethod
    def _spec_graph_candidate_diversity_context(
        seen_signatures: set[tuple[str, ...]],
        *,
        candidate_index: int,
        candidate_count: int,
    ) -> str:
        compact = [list(signature) for signature in sorted(seen_signatures)]
        return (
            f"Generate graph candidate {candidate_index}/{candidate_count}. "
            "Do not repeat any already proposed graph signature. Choose a different "
            "target region or tactic stage, while staying grounded in writable source "
            "regions and recent failure cooldowns.\n"
            + json.dumps(compact, ensure_ascii=False, indent=2)
        )

    def _run_spec_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(
            workflow.get("run_spec_after_read")
            or workflow.get("run_spec_enabled")
            or workflow.get("spec_mode")
        )

    def _spec_mode_enabled(self) -> bool:
        return bool(self.config.get("workflow", {}).get("spec_mode"))

    def _spec_tactic_portfolio_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(
            workflow.get("spec_tactic_portfolio")
            or workflow.get("spec_metric_tactic_portfolio")
        )

    def _spec_force_metric_acceptance_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        if workflow.get("spec_force_metric_acceptance"):
            return True
        return bool(
            self._spec_tactic_portfolio_enabled()
            and workflow.get("metric_regex")
            and workflow.get("spec_metric_requires_improvement", True) is not False
        )

    async def _request_run_spec_from_model(
        self,
        *,
        role: str,
        prompt: list[dict[str, str]],
        call_site: str,
        fallback_role: str,
    ) -> dict[str, Any]:
        used_role = role
        try:
            output = await self._model_chat(
                role,
                prompt,
                call_site=call_site,
            )
        except Exception as exc:
            self.state.notes.append(
                f"Run spec model call failed: {type(exc).__name__}: {exc}"
            )
            if not fallback_role or fallback_role == role:
                return {}
            if not self._consume_spec_synth_call_budget(f"{call_site}_fallback"):
                return {}
            try:
                output = await self._model_chat(
                    fallback_role,
                    prompt,
                    call_site=f"{call_site}_fallback",
                )
                self.state.notes.append(
                    f"Run spec model fallback succeeded: {fallback_role}"
                )
                used_role = fallback_role
            except Exception as fallback_exc:
                self.state.notes.append(
                    "Run spec fallback model call failed: "
                    f"{type(fallback_exc).__name__}: {fallback_exc}"
                )
                return {}
        try:
            spec = parse_json_object(output)
        except Exception as exc:
            self.state.notes.append(
                f"Run spec JSON parse failed: {type(exc).__name__}: {exc}"
            )
            if not fallback_role or fallback_role == used_role:
                return {}
            if not self._consume_spec_synth_call_budget(f"{call_site}_fallback"):
                return {}
            try:
                output = await self._model_chat(
                    fallback_role,
                    prompt,
                    call_site=f"{call_site}_fallback",
                )
                self.state.notes.append(
                    f"Run spec model fallback succeeded: {fallback_role}"
                )
                spec = parse_json_object(output)
            except Exception as fallback_exc:
                self.state.notes.append(
                    "Run spec fallback parse failed: "
                    f"{type(fallback_exc).__name__}: {fallback_exc}"
                )
                return {}
        spec = self._normalize_run_spec(spec)
        if not spec:
            self.state.notes.append("Run spec discarded: no task_graph")
            return {}
        spec.pop("search", None)
        return spec

    def _spec_two_call_synthesis_enabled(self, force: bool) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(
            force
            and self._spec_mode_enabled()
            and workflow.get("spec_two_call_synthesis")
        )

    def _spec_idea_path(self) -> Path:
        return self._workflow_artifact_path(
            "spec_idea_path",
            ".local_micro_agent/spec_idea.md",
        )

    async def _maybe_build_spec_idea_context(
        self,
        focus: str,
        *,
        force: bool,
        targeted_rewrite: bool,
    ) -> str:
        if not self._spec_two_call_synthesis_enabled(force):
            return ""
        workflow = self.config.get("workflow", {})
        role = str(workflow.get("spec_idea_model_role") or "reasoner")
        call_site = "spec_idea_rewrite" if targeted_rewrite else "spec_idea"
        remaining = self._spec_synth_call_budget_remaining()
        if remaining == 0:
            self.state.notes.append("Spec idea skipped: spec synthesis call budget exhausted")
            return ""
        if remaining == 1:
            self.state.notes.append(
                "Spec idea skipped: preserving final spec synthesis call budget"
            )
            return ""
        if not self._consume_spec_synth_call_budget(call_site):
            return ""
        try:
            output = await self._model_chat(
                role,
                spec_idea_prompt(self.state, focus=focus),
                call_site=call_site,
            )
        except Exception as exc:
            self.state.notes.append(
                f"Spec idea model call failed; continuing with finalizer only: "
                f"{type(exc).__name__}: {exc}"
            )
            self.state.scratch.pop("spec_idea_brief", None)
            return ""
        brief = output.strip()
        if not brief:
            self.state.notes.append("Spec idea model returned empty brief; finalizer will use facts only")
            self.state.scratch.pop("spec_idea_brief", None)
            return ""
        path = self._spec_idea_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(brief.rstrip() + "\n")
        self.state.scratch["spec_idea_brief"] = brief
        self.state.notes.append(f"Persisted spec idea brief: {path}")
        return (
            "Spec idea brief from a non-authoritative thinking/analysis pass "
            "follows. Use it only as advisory design input. The final JSON spec "
            "must still obey deterministic grounding facts and workflow constraints.\n"
            + brief
        )

    def _spec_grounding_gate_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("spec_grounding_gate"))

    def _spec_grounding_facts_path(self) -> Path:
        return self._workflow_artifact_path(
            "spec_grounding_facts_path",
            ".local_micro_agent/spec_grounding_facts.json",
        )

    def _spec_grounding_facts_context(self) -> str:
        if not (
            self._spec_grounding_gate_enabled()
            or self.config.get("workflow", {}).get("spec_grounding_facts")
        ):
            return ""
        facts = self._spec_grounding_facts()
        if not facts:
            return ""
        path = self._spec_grounding_facts_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n")
        self.state.scratch["spec_grounding_facts"] = facts
        self.state.notes.append(f"Persisted spec grounding facts: {path}")
        compact = {
            "writable_files": facts.get("writable_files", []),
            "read_only_files": facts.get("read_only_files", []),
            "allowed_target_regions": facts.get("allowed_target_regions", [])[:80],
            "imported_symbols": facts.get("imported_symbols", [])[:80],
            "test_commands": facts.get("test_commands", []),
            "metric_regex": facts.get("metric_regex"),
            "baseline_metric": facts.get("baseline_metric"),
        }
        return (
            "Spec grounding facts from the deterministic controller follow. "
            "Every implementation task must choose target_regions and "
            "probe_diff_contract expected_changed_regions only from "
            "allowed_target_regions unless it is a file-level non-Python edit. "
            "Imported/read-only symbols may be mentioned as context or invariants, "
            "but they must not be deliverables or changed targets.\n"
            + json.dumps(compact, ensure_ascii=False, indent=2)
        )

    def _spec_grounding_facts(self) -> dict[str, Any]:
        workflow = self.config.get("workflow", {})
        writable = self._spec_grounding_writable_files()
        context_paths = {snapshot.path for snapshot in self.state.file_context}
        paths = sorted(context_paths | writable)
        files: dict[str, dict[str, Any]] = {}
        symbol_regions: dict[str, dict[str, Any]] = {}
        imports: list[dict[str, Any]] = []
        for rel_path in paths:
            if not rel_path or self._repo_path_key(rel_path) in self._external_context_path_keys():
                continue
            abs_path = self.state.repo_root / rel_path
            content = self._spec_grounding_file_content(rel_path)
            file_record: dict[str, Any] = {
                "path": rel_path,
                "writable": self._spec_path_is_writable(rel_path, writable),
                "exists": abs_path.exists(),
            }
            if rel_path.endswith(".py") and content is not None:
                py_facts = self._python_spec_grounding_facts(rel_path, content)
                file_record.update(
                    {
                        "language": "python",
                        "defined_regions": py_facts["defined_regions"],
                        "imports": py_facts["imports"],
                    }
                )
                for region in py_facts["defined_regions"]:
                    symbol_regions[str(region["region"])] = region
                imports.extend(py_facts["imports"])
            files[rel_path] = file_record
        imported_symbols = [
            self._resolve_imported_symbol(record)
            for record in imports
            if self._spec_path_is_writable(str(record.get("path", "")), writable)
        ]
        imported_symbols = [
            record for record in imported_symbols if record.get("origin_path")
        ]
        read_only_symbols = [
            region
            for region, meta in sorted(symbol_regions.items())
            if not self._spec_path_is_writable(str(meta.get("path", "")), writable)
        ]
        allowed_target_regions = [
            region
            for region, meta in sorted(symbol_regions.items())
            if self._spec_path_is_writable(str(meta.get("path", "")), writable)
        ]
        facts = {
            "version": 1,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "writable_files": sorted(writable),
            "read_only_files": sorted(path for path in files if path not in writable),
            "files": files,
            "allowed_target_regions": allowed_target_regions,
            "read_only_symbols": read_only_symbols,
            "imported_symbols": imported_symbols,
            "test_commands": workflow.get("test_commands", []),
            "metric_regex": workflow.get("metric_regex"),
            "baseline_metric": workflow.get("baseline_metric"),
        }
        return facts

    def _spec_grounding_writable_files(self) -> set[str]:
        workflow = self.config.get("workflow", {})
        candidates = workflow.get("writable_files") or self.state.planned_files
        return {str(path).strip() for path in candidates if str(path).strip()}

    def _spec_grounding_file_content(self, rel_path: str) -> str | None:
        for snapshot in self.state.file_context:
            if snapshot.path == rel_path:
                return snapshot.content
        try:
            return (self.state.repo_root / rel_path).read_text(errors="replace")
        except FileNotFoundError:
            return None

    def _python_spec_grounding_facts(
        self, rel_path: str, content: str
    ) -> dict[str, list[dict[str, Any]]]:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {"defined_regions": [], "imports": []}
        regions: list[dict[str, Any]] = []
        imports: list[dict[str, Any]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                regions.append(self._python_region_record(rel_path, node.name, node))
            elif isinstance(node, ast.ClassDef):
                regions.append(self._python_region_record(rel_path, node.name, node))
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        regions.append(
                            self._python_region_record(
                                rel_path,
                                f"{node.name}.{child.name}",
                                child,
                                parent=node.name,
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                module = "." * int(node.level or 0) + (node.module or "")
                for alias in node.names:
                    imports.append(
                        {
                            "path": rel_path,
                            "kind": "from",
                            "module": module,
                            "name": alias.name,
                            "asname": alias.asname or alias.name,
                            "line": getattr(node, "lineno", None),
                        }
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        {
                            "path": rel_path,
                            "kind": "import",
                            "module": alias.name,
                            "name": alias.name,
                            "asname": alias.asname or alias.name.split(".")[0],
                            "line": getattr(node, "lineno", None),
                        }
                    )
        return {"defined_regions": regions, "imports": imports}

    @staticmethod
    def _python_region_record(
        rel_path: str,
        symbol: str,
        node: ast.AST,
        parent: str = "",
    ) -> dict[str, Any]:
        return {
            "path": rel_path,
            "symbol": symbol,
            "region": f"{rel_path}::{symbol}",
            "parent": parent,
            "start_line": getattr(node, "lineno", None),
            "end_line": getattr(node, "end_lineno", getattr(node, "lineno", None)),
            "kind": "method" if parent else node.__class__.__name__.replace("Def", "").lower(),
        }

    def _resolve_imported_symbol(self, record: dict[str, Any]) -> dict[str, Any]:
        module = str(record.get("module", "") or "")
        name = str(record.get("name", "") or "")
        origin_path = ""
        origin_region = ""
        if module and not module.startswith("."):
            candidate = module.replace(".", "/") + ".py"
            if (self.state.repo_root / candidate).exists():
                origin_path = candidate
                if str(record.get("kind")) == "from" and name != "*":
                    origin_region = f"{candidate}::{name}"
        return {
            "path": record.get("path"),
            "symbol": record.get("asname") or name,
            "imported_name": name,
            "module": module,
            "origin_path": origin_path,
            "origin_region": origin_region,
            "line": record.get("line"),
        }

    @staticmethod
    def _spec_path_is_writable(rel_path: str, writable: set[str]) -> bool:
        return rel_path in writable or any(fnmatch.fnmatch(rel_path, pattern) for pattern in writable)

    def _current_spec_grounding_facts(self) -> dict[str, Any]:
        facts = self.state.scratch.get("spec_grounding_facts")
        if isinstance(facts, dict) and facts:
            return facts
        path = self._spec_grounding_facts_path()
        if path.exists():
            try:
                loaded = json.loads(path.read_text(errors="replace"))
            except json.JSONDecodeError:
                loaded = {}
            if isinstance(loaded, dict):
                self.state.scratch["spec_grounding_facts"] = loaded
                return loaded
        facts = self._spec_grounding_facts()
        self.state.scratch["spec_grounding_facts"] = facts
        path = self._spec_grounding_facts_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n")
        return facts

    def _spec_quality_gate_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("spec_quality_gate"))

    def _spec_quality_rewrite_attempts(self) -> int:
        if not self._spec_quality_gate_enabled():
            return 0
        workflow = self.config.get("workflow", {})
        return max(0, int(workflow.get("spec_quality_rewrite_attempts", 1) or 0))

    def _spec_synth_call_budget(self) -> int | None:
        raw = self.config.get("workflow", {}).get("spec_synth_call_budget")
        if raw in (None, "", False):
            return None
        budget = int(raw)
        return budget if budget >= 0 else None

    def _spec_synth_call_count(self) -> int:
        return int(self.state.scratch.get("spec_synth_call_count", 0) or 0)

    def _spec_synth_call_budget_remaining(self) -> int | None:
        budget = self._spec_synth_call_budget()
        if budget is None:
            return None
        return max(0, budget - self._spec_synth_call_count())

    def _consume_spec_synth_call_budget(self, call_site: str) -> bool:
        budget = self._spec_synth_call_budget()
        used = self._spec_synth_call_count()
        if budget is not None and used >= budget:
            self.state.scratch["spec_synth_budget_exhausted"] = True
            self.state.scratch["spec_synth_budget_exhausted_at"] = {
                "call_site": call_site,
                "used": used,
                "budget": budget,
            }
            self.state.notes.append(
                f"Spec synthesis call budget exhausted before {call_site}: "
                f"{used}/{budget}"
            )
            return False
        self.state.scratch["spec_synth_call_count"] = used + 1
        return True

    def _spec_gate_soft_fallback_enabled(self) -> bool:
        return bool(self.config.get("workflow", {}).get("spec_gate_soft_fallback"))

    def _spec_quality_report_path(self) -> Path:
        return self._workflow_artifact_path(
            "spec_quality_report_path",
            ".local_micro_agent/spec_quality_report.json",
        )

    def _persist_spec_quality_report(self, report: dict[str, Any]) -> None:
        if not self._spec_quality_gate_enabled():
            return
        path = self._spec_quality_report_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        self.state.scratch["spec_quality_report"] = report

    @staticmethod
    def _spec_quality_report_failed(report: dict[str, Any]) -> bool:
        return str(report.get("status") or "") == "fail"

    def _spec_quality_report(
        self,
        spec: dict[str, Any],
        *,
        attempt: int = 0,
    ) -> dict[str, Any]:
        issues = self._spec_quality_issues(spec)
        report = {
            "version": 1,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "fail" if issues else "pass",
            "attempt": attempt,
            "spec_id": spec.get("spec_id"),
            "issues": issues,
            "issue_codes": [
                issue.get("code")
                for issue in issues
                if isinstance(issue, dict) and issue.get("code")
            ],
        }
        return report

    def _spec_quality_issues(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        if not self._spec_quality_gate_enabled():
            return []
        tasks = [
            task
            for task in self._schedulable_spec_tasks(spec.get("task_graph", []))
            if isinstance(task, dict) and not self._spec_task_is_context_only(task)
        ]
        issues: list[dict[str, Any]] = []
        for task in tasks:
            issues.extend(self._spec_task_quality_issues(spec, task))
        preferred = self._spec_idea_preferred_target_region()
        if preferred:
            runnable = self._schedulable_spec_tasks(spec.get("task_graph", []))
            first = next(
                (
                    task
                    for task in runnable
                    if isinstance(task, dict)
                    and not self._spec_task_is_context_only(task)
                ),
                None,
            )
            first_targets = (
                self._normalize_string_list(first.get("target_regions"))
                if isinstance(first, dict)
                else []
            )
            if (
                preferred not in first_targets
                and not self._spec_has_supported_idea_rejection(spec, preferred)
            ):
                issues.append(
                    {
                        "code": "idea_alignment_failed",
                        "severity": "error",
                        "task_id": first.get("task_id") if isinstance(first, dict) else "",
                        "preferred_target_region": preferred,
                        "detail": (
                            "SPEC_FINALIZE silently ignored the first feasible "
                            "SPEC_IDEA target instead of making it the first "
                            "runnable task or citing a deterministic rejection reason."
                        ),
                        "rewrite_hint": (
                            "Either make the preferred SPEC_IDEA target the first "
                            "runnable task, or add an idea_rejection_reason in "
                            "known_facts/decision_rules citing grounding facts or "
                            "failure memory."
                        ),
                    }
                )
        return issues

    def _spec_task_quality_issues(
        self, spec: dict[str, Any], task: dict[str, Any]
    ) -> list[dict[str, Any]]:
        workflow = self.config.get("workflow", {})
        task_id = str(task.get("task_id") or "")
        issues: list[dict[str, Any]] = []
        max_deliverables = int(workflow.get("spec_quality_max_deliverables", 1) or 1)
        deliverables = self._normalize_string_list(task.get("deliverables"))
        if max_deliverables > 0 and len(deliverables) > max_deliverables:
            issues.append(
                self._spec_quality_issue(
                    "too_many_deliverables",
                    task_id,
                    f"deliverables has {len(deliverables)} entries",
                    "Use one writable deliverable per implementation task.",
                )
            )
        max_read_hints = int(workflow.get("spec_quality_max_read_hints", 3) or 3)
        read_hints = self._normalize_string_list(task.get("read_hints"))
        if max_read_hints >= 0 and len(read_hints) > max_read_hints:
            issues.append(
                self._spec_quality_issue(
                    "too_many_read_hints",
                    task_id,
                    f"read_hints has {len(read_hints)} entries",
                    "Keep read_hints focused on the source needed by this task.",
                )
            )
        target_regions = self._normalize_string_list(task.get("target_regions"))
        stage = str(task.get("tactic_stage") or "").strip().lower()
        if len(target_regions) != 1:
            issues.append(
                self._spec_quality_issue(
                    "target_region_count",
                    task_id,
                    f"target_regions has {len(target_regions)} entries",
                    "Use exactly one primary target region per runnable task.",
                )
            )
        max_target_lines = int(workflow.get("spec_quality_max_target_lines", 160) or 160)
        if max_target_lines > 0:
            for region in target_regions:
                line_count = self._spec_region_line_count(region)
                if line_count is not None and line_count > max_target_lines:
                    issues.append(
                        self._spec_quality_issue(
                            "target_span_too_large",
                            task_id,
                            f"{region} spans {line_count} lines",
                            "Split the task or choose a smaller nested target region.",
                        )
                    )
        edit_scope = str(task.get("edit_scope") or "").strip()
        if self._spec_quality_edit_scope_too_vague(edit_scope):
            issues.append(
                self._spec_quality_issue(
                    "vague_edit_scope",
                    task_id,
                    edit_scope or "missing edit_scope",
                    self._spec_quality_vague_edit_scope_hint(task, edit_scope),
                )
            )
        acceptance = task.get("acceptance")
        kind = (
            str(acceptance.get("kind") or "").strip()
            if isinstance(acceptance, dict)
            else ""
        )
        if (
            (workflow.get("metric_regex") or workflow.get("test_commands"))
            and kind == "synthesized"
        ):
            issues.append(
                self._spec_quality_issue(
                    "acceptance_not_configured_command_or_metric",
                    task_id,
                    "acceptance.kind=synthesized despite configured test/metric",
                    "Use metric or command acceptance when workflow supplies deterministic validation.",
                )
            )
        fallback_text = " ".join(
            str(task.get(key) or "")
            for key in ("fallback_plan", "rollback_or_shrink_plan")
        )
        if not self._spec_quality_fallback_is_safe(fallback_text):
            issues.append(
                self._spec_quality_issue(
                    "unsafe_or_missing_fallback",
                    task_id,
                    fallback_text.strip() or "missing fallback plan",
                    "Fallback must say revert, restore, shrink, guard, or probe.",
                )
            )
        contract = task.get("probe_diff_contract")
        if stage == "structural_probe" and isinstance(contract, dict):
            expected = self._normalize_string_list(contract.get("expected_changed_regions"))
            if len(expected) != 1:
                issues.append(
                    self._spec_quality_issue(
                        "structural_probe_expected_region_count",
                        task_id,
                        f"expected_changed_regions has {len(expected)} entries",
                        "A structural probe must name one expected changed region.",
                    )
                )
        design_issues = self._spec_task_design_contract_issues(spec, task)
        issues.extend(
            self._spec_quality_issue_from_design_issue(task_id, issue)
            for issue in design_issues
        )
        return issues

    @staticmethod
    def _spec_quality_issue(
        code: str,
        task_id: str,
        detail: str,
        rewrite_hint: str,
    ) -> dict[str, Any]:
        return {
            "code": code,
            "severity": "error",
            "task_id": task_id,
            "detail": detail,
            "rewrite_hint": rewrite_hint,
        }

    def _spec_quality_issue_from_design_issue(
        self, task_id: str, design_issue: str
    ) -> dict[str, Any]:
        raw = str(design_issue or "").strip()
        base = raw.split(":", 1)[0] or "issue"
        code = "design_contract_" + re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")
        return self._spec_quality_issue(
            code,
            task_id,
            raw,
            self._spec_design_issue_rewrite_hint(raw),
        )

    @staticmethod
    def _spec_design_issue_rewrite_hint(design_issue: str) -> str:
        lowered = design_issue.lower()
        if "probe_contract_region_mismatch" in lowered:
            return (
                "Make probe_diff_contract.expected_changed_regions exactly match "
                "the task target region, or split cross-region work into a "
                "separate structural task. Bad: target=parse_item with expected "
                "[parse_item, format_item]. Good: target=parse_item with expected "
                "[parse_item]."
            )
        if "rollback_or_shrink_plan" in lowered:
            return (
                "Describe a smaller guarded probe, not only a revert. Bad: "
                "'Revert the patch.' Good: 'Shrink to one guarded branch in the "
                "target region; keep the old path as fallback; revert if the "
                "guarded probe fails.'"
            )
        if "structural edit_scope too broad" in lowered or "edit_scope too broad" in lowered:
            return (
                "Narrow the task to one reversible operation in one target "
                "region. Split multi-step rewrites into independent probe tasks."
            )
        if "local risk_level contradicts structural action" in lowered:
            return (
                "Reclassify the task as risk_level=structural with "
                "tactic_stage=structural_probe, or reduce the edit_scope to a "
                "single local edit that does not change signatures, callsites, "
                "data flow, ordering, or side effects."
            )
        if "missing probe_diff_contract" in lowered:
            return (
                "Add a probe_diff_contract with allowed_files, allowed_regions, "
                "exactly one expected_changed_regions entry, forbidden regions, "
                "and small max_files/max_hunks/max_changed_lines limits."
            )
        if "missing probe_plan" in lowered:
            return "Add the smallest reversible structural probe plan before expanding."
        return (
            "Rewrite the runnable task so it satisfies the deterministic design "
            "contract before run_spec persistence."
        )

    def _spec_region_line_count(self, region: str) -> int | None:
        facts = self._current_spec_grounding_facts()
        region_path = self._region_path(region)
        file_record = facts.get("files", {}).get(region_path)
        if not isinstance(file_record, dict):
            return None
        for record in file_record.get("defined_regions", []) or []:
            if not isinstance(record, dict) or str(record.get("region") or "") != region:
                continue
            start = record.get("start_line")
            end = record.get("end_line")
            if isinstance(start, int) and isinstance(end, int) and end >= start:
                return end - start + 1
        return None

    @staticmethod
    def _spec_quality_edit_scope_too_vague(edit_scope: str) -> bool:
        text = edit_scope.strip().lower()
        if not text:
            return True
        vague_patterns = (
            r"^optimi[sz]e\b",
            r"^improve\b",
            r"^refactor\b",
            r"^rewrite\b",
            r"^clean up\b",
            r"^make .* faster\b",
        )
        concrete_markers = (
            "add ",
            "remove ",
            "replace ",
            "guard",
            "branch",
            "call",
            "loop",
            "constant",
            "cache",
            "check",
            "return",
            "condition",
            "assignment",
            "one ",
            "single ",
        )
        return any(re.search(pattern, text) for pattern in vague_patterns) and not any(
            marker in text for marker in concrete_markers
        )

    def _spec_quality_vague_edit_scope_hint(
        self,
        task: dict[str, Any],
        edit_scope: str,
    ) -> str:
        target_regions = self._normalize_string_list(task.get("target_regions"))
        target = target_regions[0] if len(target_regions) == 1 else "the target region"
        hint = (
            "State one exact operation boundary in one target region; do not use "
            "only optimize/refactor/improve. Bad: 'Refactor target() to group "
            "and pack work.' Good: 'In "
            + target
            + ", add one guarded branch for a single operation category; keep "
            "the old path as fallback and preserve existing ordering invariants.'"
        )
        lowered = edit_scope.lower()
        if "pack" in lowered or "group" in lowered:
            hint += (
                " For group/pack ideas, name the one category being packed and "
                "the categories/orderings intentionally left unchanged."
            )
        return hint

    @staticmethod
    def _spec_quality_fallback_is_safe(text: str) -> bool:
        lowered = text.lower()
        if not lowered.strip():
            return False
        unsafe_patterns = (
            r"\bpass\b",
            r"\breplace\s+whole\b",
            r"\breplace\s+entire\b",
            r"\brewrite\s+whole\b",
            r"\brewrite\s+entire\b",
            r"\bfull\s+rewrite\b",
            r"\bdelete\s+the\s+function\b",
        )
        if any(re.search(pattern, lowered) for pattern in unsafe_patterns):
            return False
        return any(
            marker in lowered
            for marker in (
                "revert",
                "restore",
                "rollback",
                "shrink",
                "smaller",
                "guard",
                "probe",
                "abandon",
            )
        )

    def _spec_idea_preferred_target_region(self) -> str:
        if not self._spec_two_call_synthesis_enabled(True):
            return ""
        brief = str(self.state.scratch.get("spec_idea_brief") or "")
        if not brief:
            try:
                brief = self._spec_idea_path().read_text(errors="replace")
            except FileNotFoundError:
                return ""
        facts = self._current_spec_grounding_facts()
        allowed = {
            str(region)
            for region in facts.get("allowed_target_regions", []) or []
            if str(region)
        }
        if not allowed:
            return ""
        pattern = r"[\w./-]+\.py::[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*"
        for match in re.finditer(pattern, brief):
            candidate = match.group(0)
            if candidate in allowed:
                return candidate
        return ""

    @staticmethod
    def _spec_has_supported_idea_rejection(
        spec: dict[str, Any],
        preferred_region: str,
    ) -> bool:
        preferred_symbol = preferred_region.split("::", 1)[-1]
        text = "\n".join(
            str(item)
            for field in ("known_facts", "decision_rules")
            for item in (spec.get(field, []) if isinstance(spec.get(field), list) else [])
        ).lower()
        if "idea_rejection" not in text and "rejected idea" not in text:
            return False
        if preferred_region.lower() not in text and preferred_symbol.lower() not in text:
            return False
        supported_markers = (
            "non_writable",
            "unresolvable",
            "read_only",
            "imported",
            "failed",
            "no_improvement",
            "patch_miss",
            "too broad",
            "ambiguous",
            "scope",
            "grounding",
        )
        return any(marker in text for marker in supported_markers)

    @staticmethod
    def _spec_quality_feedback_context(report: dict[str, Any]) -> str:
        issues = [
            issue
            for issue in report.get("issues", [])
            if isinstance(issue, dict)
        ]
        compact = [
            {
                "code": issue.get("code"),
                "task_id": issue.get("task_id"),
                "detail": issue.get("detail"),
                "preferred_target_region": issue.get("preferred_target_region"),
                "rewrite_hint": issue.get("rewrite_hint"),
            }
            for issue in issues[:12]
        ]
        return (
            "SPEC quality gate rejected the previous finalizer output. Rewrite "
            "the JSON spec to fix only these domain-neutral experiment-design "
            "issues. Do not invent benchmark-specific tactics. If you do not "
            "use the first feasible SPEC_IDEA target, add an "
            "idea_rejection_reason in known_facts or decision_rules that cites "
            "grounding facts or failure memory.\n"
            + json.dumps(compact, ensure_ascii=False, indent=2)
        )

    @staticmethod
    def _load_run_spec(path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(errors="replace"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _current_run_spec_snapshot(self, path: Path | None = None) -> dict[str, Any]:
        spec = self.state.scratch.get("run_spec")
        if isinstance(spec, dict) and spec:
            snapshot = self._normalize_run_spec(copy.deepcopy(spec))
            if snapshot:
                return snapshot
        if path is None:
            path = self._run_spec_path()
        loaded = self._load_run_spec(path)
        if loaded:
            return self._normalize_run_spec(loaded)
        return {}

    def _targeted_spec_rewrite_enabled(
        self,
        force: bool,
        previous_spec: dict[str, Any],
        target_task_id: str,
    ) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(
            force
            and self._spec_mode_enabled()
            and workflow.get("spec_preserve_rewrite_portfolio", True) is not False
            and isinstance(previous_spec, dict)
            and previous_spec.get("task_graph")
            and target_task_id
        )

    def _spec_rewrite_portfolio_context(
        self,
        previous_spec: dict[str, Any],
        target_task_id: str,
    ) -> str:
        if not self._targeted_spec_rewrite_enabled(True, previous_spec, target_task_id):
            return ""
        tasks = previous_spec.get("task_graph")
        if not isinstance(tasks, list):
            return ""
        compact_tasks = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            compact = {
                "task_id": task.get("task_id"),
                "status": task.get("status"),
                "title": task.get("title"),
                "strategy_axis": task.get("strategy_axis"),
                "family_key": task.get("family_key"),
                "risk_level": task.get("risk_level"),
                "tactic_stage": task.get("tactic_stage"),
                "target_symbols": task.get("target_symbols"),
                "target_regions": task.get("target_regions"),
                "edit_scope": task.get("edit_scope"),
                "depends_on": task.get("depends_on"),
            }
            compact_tasks.append(
                {key: value for key, value in compact.items() if value not in (None, "", [], {})}
            )
        if not compact_tasks:
            return ""
        return (
            "Existing task graph before this targeted SPEC rewrite follows. The "
            f"rewrite target is {target_task_id}. Preserve all sibling tasks unless "
            "the current source evidence proves they are obsolete. Do not collapse "
            "the portfolio to one broad task. If you replace the target, keep the "
            "same task_id or set replaces_task_id to the rejected task id. New "
            "implementation tasks must still be bounded, verifiable, independently "
            "executable units.\n"
            + json.dumps(compact_tasks, ensure_ascii=False, indent=2)
        )

    def _normalize_run_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(spec, dict):
            return {}
        tasks = spec.get("task_graph")
        if not isinstance(tasks, list) or not tasks:
            return {}
        normalized_tasks = []
        known_axes = set(self._known_strategy_axes())
        version = int(spec.get("version", 1) or 1)
        max_tasks = int(self.config.get("workflow", {}).get("spec_max_tasks", 24) or 24)
        for index, task in enumerate(tasks, start=1):
            if max_tasks > 0 and len(normalized_tasks) >= max_tasks:
                break
            if not isinstance(task, dict):
                continue
            axis = self._normalize_strategy_axis(str(task.get("strategy_axis", "")))
            if not axis:
                axis = "general_edit"
            if self._strict_strategy_axis_pool_enabled() and axis not in known_axes:
                axis = "general_edit"
            task_id = str(task.get("task_id") or f"task-{index:03d}").strip()
            normalized = {
                "task_id": task_id,
                "replaces_task_id": str(task.get("replaces_task_id") or "").strip(),
                "title": str(task.get("title") or task_id).strip(),
                "strategy_axis": axis,
                "family_key": self._normalize_strategy_axis(str(task.get("family_key", ""))),
                "expected_signal": str(task.get("expected_signal", "")).strip(),
                "status": str(task.get("status") or "open").strip(),
                "attempts": int(task.get("attempts", 0) or 0),
                "last_observation": task.get("last_observation", ""),
                "decision_hint": task.get("decision_hint", ""),
            }
            if isinstance(task.get("design_contract"), dict):
                normalized["design_contract"] = copy.deepcopy(task["design_contract"])
            if version >= 2:
                normalized.update(self._normalize_run_spec_v2_task(task))
            normalized_tasks.append(normalized)
        if not normalized_tasks:
            return {}
        normalized_spec = {
            "version": 2 if version >= 2 else 1,
            "spec_id": str(spec.get("spec_id") or "run-spec").strip(),
            "objective": str(spec.get("objective", "")).strip(),
            "invariants": [
                str(item).strip()
                for item in spec.get("invariants", [])
                if str(item).strip()
            ]
            if isinstance(spec.get("invariants"), list)
            else [],
            "known_facts": [
                str(item).strip()
                for item in spec.get("known_facts", [])
                if str(item).strip()
            ]
            if isinstance(spec.get("known_facts"), list)
            else [],
            "task_graph": normalized_tasks,
            "decision_rules": [
                str(item).strip()
                for item in spec.get("decision_rules", [])
                if str(item).strip()
            ]
            if isinstance(spec.get("decision_rules"), list)
            else [],
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if version >= 2:
            normalized_spec["progress"] = self._run_spec_progress(normalized_spec)
        if isinstance(spec.get("search"), dict):
            normalized_spec["search"] = copy.deepcopy(spec["search"])
        return normalized_spec

    def _normalize_run_spec_v2_task(self, task: dict[str, Any]) -> dict[str, Any]:
        workflow = self.config.get("workflow", {})
        depends_on = [
            str(item).strip()
            for item in task.get("depends_on", [])
            if str(item).strip()
        ] if isinstance(task.get("depends_on"), list) else []
        deliverables = [
            str(item).strip()
            for item in task.get("deliverables", [])
            if str(item).strip()
        ] if isinstance(task.get("deliverables"), list) else []
        read_hints = [
            str(item).strip()
            for item in task.get("read_hints", [])
            if str(item).strip()
        ] if isinstance(task.get("read_hints"), list) else []
        acceptance = task.get("acceptance")
        if not isinstance(acceptance, dict):
            acceptance = {}
        default_kind = workflow.get("spec_default_acceptance_kind", "command")
        if self._spec_force_metric_acceptance_enabled():
            kind = "metric"
        elif workflow.get("spec_force_default_acceptance_kind"):
            kind = str(default_kind).strip() or "command"
        else:
            kind = str(acceptance.get("kind") or default_kind).strip() or "command"
        commands = acceptance.get("commands")
        if kind == "metric" and self._spec_force_metric_acceptance_enabled():
            commands = commands if isinstance(commands, list) else []
        elif not isinstance(commands, list):
            commands = workflow.get("test_commands", [])
        if kind == "metric" and self._spec_tactic_portfolio_enabled():
            depends_on = []
        normalized_acceptance = {
            "kind": kind,
            "commands": [str(command) for command in commands if str(command).strip()],
        }
        for key in ("test_paths", "frozen_sha256", "synthesized_at", "red_first"):
            value = acceptance.get(key)
            if value not in (None, "", [], {}):
                normalized_acceptance[key] = value
        budget = task.get("budget")
        if not isinstance(budget, dict):
            budget = {}
        attempts_max = int(
            budget.get(
                "attempts_max",
                workflow.get("spec_task_attempt_budget", workflow.get("todo_attempt_budget", 3)),
            )
            or 1
        )
        attempts_used = int(budget.get("attempts_used", task.get("attempts", 0)) or 0)
        target_symbols = self._normalize_string_list(task.get("target_symbols"))
        target_regions = self._normalize_string_list(task.get("target_regions"))
        return {
            "depends_on": depends_on,
            "deliverables": deliverables,
            "read_hints": read_hints,
            "target_symbols": target_symbols,
            "target_regions": target_regions,
            "preserved_invariants": self._normalize_string_list(
                task.get("preserved_invariants")
            ),
            "edit_scope": self._normalize_task_text_field(task.get("edit_scope")),
            "risk_level": self._normalize_risk_level(task.get("risk_level")),
            "tactic_stage": self._normalize_tactic_stage(task.get("tactic_stage")),
            "risk_evidence": self._normalize_task_risk_evidence(
                task.get("risk_evidence")
            ),
            "probe_plan": self._normalize_task_text_field(task.get("probe_plan")),
            "probe_diff_contract": self._normalize_probe_diff_contract(
                task.get("probe_diff_contract"),
                deliverables=deliverables,
                target_symbols=target_symbols,
                target_regions=target_regions,
            ),
            "invariant_evidence": self._normalize_string_list(
                task.get("invariant_evidence")
            ),
            "validator": self._normalize_task_validator(task.get("validator")),
            "correctness_rationale": self._normalize_task_text_field(
                task.get("correctness_rationale")
            ),
            "fallback_plan": self._normalize_task_text_field(task.get("fallback_plan")),
            "rollback_or_shrink_plan": self._normalize_task_text_field(
                task.get("rollback_or_shrink_plan")
            ),
            "acceptance": normalized_acceptance,
            "budget": {"attempts_max": attempts_max, "attempts_used": attempts_used},
            "closed_at": task.get("closed_at"),
            "recovery_rounds": int(task.get("recovery_rounds", 0) or 0),
            "attempts_total": int(task.get("attempts_total", 0) or 0),
        }

    @staticmethod
    def _normalize_string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, list):
            items = value
        else:
            return []
        return [str(item).strip() for item in items if str(item).strip()]

    @staticmethod
    def _normalize_task_text_field(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _normalize_risk_level(value: Any) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in {"local", "structural"}:
            return normalized
        return ""

    @staticmethod
    def _normalize_tactic_stage(value: Any) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in {"local_edit", "structural_probe", "structural_expand"}:
            return normalized
        return ""

    @staticmethod
    def _risk_evidence_fields() -> tuple[str, ...]:
        return (
            "title",
            "edit_scope",
            "strategy_axis",
            "family_key",
            "expected_signal",
        )

    def _normalize_task_risk_evidence(self, value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        field = str(value.get("field") or "").strip()
        if field not in self._risk_evidence_fields():
            field = ""
        normalized = {
            "field": field,
            "quote": str(value.get("quote") or "").strip(),
            "explanation": str(value.get("explanation") or "").strip(),
        }
        return {key: item for key, item in normalized.items() if item}

    def _normalize_probe_diff_contract(
        self,
        value: Any,
        *,
        deliverables: list[str],
        target_symbols: list[str],
        target_regions: list[str],
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        normalized: dict[str, Any] = {}
        for field in (
            "allowed_files",
            "allowed_regions",
            "expected_changed_regions",
            "target_symbols",
            "forbidden_symbols",
            "forbidden_regions",
            "required_unchanged_regions",
            "allowed_change_kinds",
        ):
            normalized[field] = self._normalize_string_list(value.get(field))
        if not normalized["allowed_files"]:
            normalized["allowed_files"] = list(deliverables)
        if not normalized["allowed_regions"]:
            normalized["allowed_regions"] = [*target_regions, *target_symbols]
        if not normalized["expected_changed_regions"]:
            normalized["expected_changed_regions"] = normalized["allowed_regions"]
        if not normalized["target_symbols"]:
            normalized["target_symbols"] = list(target_symbols)
        for field in (
            "max_files_changed",
            "max_hunks",
            "max_changed_lines",
            "max_changed_functions",
        ):
            try:
                parsed = int(value.get(field))
            except (TypeError, ValueError):
                continue
            if parsed >= 0:
                normalized[field] = parsed
        observation = self._normalize_task_text_field(value.get("observation"))
        if observation:
            normalized["observation"] = observation
        return {
            key: item
            for key, item in normalized.items()
            if item not in (None, "", [], {})
        }

    def _normalize_task_validator(self, value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        normalized = {
            "kind": str(value.get("kind") or "").strip(),
            "failure_condition": str(value.get("failure_condition") or "").strip(),
        }
        command = str(value.get("command") or "").strip()
        if command:
            normalized["command"] = command
        return {key: item for key, item in normalized.items() if item}

    def _todo_soft_until_first_improvement_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("todo_soft_until_first_improvement", True))

    def _spec_hard_active_todo_contract_now(self) -> bool:
        workflow = self.config.get("workflow", {})
        if not self._spec_mode_enabled():
            return False
        hard_enabled = workflow.get(
            "spec_hard_active_todo_contract",
            workflow.get("spec_design_contract_gate", False),
        )
        if not hard_enabled:
            return False
        active_todo = self.state.scratch.get("active_todo")
        if not isinstance(active_todo, dict):
            active_todo = self._load_active_todo()
            if active_todo:
                self.state.scratch["active_todo"] = active_todo
        if not isinstance(active_todo, dict):
            return False
        return bool(
            active_todo.get("spec_task_id")
            or active_todo.get("source") == "spec_scheduler"
        )

    def _run_spec_path(self) -> Path:
        return self._workflow_artifact_path(
            "run_spec_path", ".local_micro_agent/run_spec.json"
        )

    def _persist_run_spec(self, spec: dict[str, Any]) -> None:
        self._ensure_spec_search_metadata(spec, origin="persisted")
        spec["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if int(spec.get("version", 1) or 1) >= 2:
            spec["progress"] = self._run_spec_progress(spec)
        self.state.scratch["run_spec"] = spec
        path = self._run_spec_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")

    def _spec_graph_candidates_path(self) -> Path:
        return self._workflow_artifact_path(
            "spec_graph_candidates_path",
            ".local_micro_agent/spec_graph_candidates.jsonl",
        )

    def _spec_graph_candidate_sidecar_dir(self) -> Path:
        return self._workflow_artifact_path(
            "spec_graph_candidate_dir",
            ".local_micro_agent/spec_graph_candidates",
        )

    def _spec_graph_candidate_sidecar_path(self, graph_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", graph_id).strip("_")
        if not safe_id:
            safe_id = "graph-unknown"
        return self._spec_graph_candidate_sidecar_dir() / f"{safe_id}.json"

    def _apply_spec_graph_generation_metadata(
        self,
        spec: dict[str, Any],
        *,
        previous_spec: dict[str, Any],
        origin: str,
        parent_graph_id: str = "",
    ) -> None:
        if not isinstance(spec, dict) or int(spec.get("version", 1) or 1) < 2:
            return
        if not str(origin or "").startswith("reseed"):
            return
        search = spec.get("search") if isinstance(spec.get("search"), dict) else {}
        if not search:
            search = {}
            spec["search"] = search
        previous_search = (
            previous_spec.get("search")
            if isinstance(previous_spec.get("search"), dict)
            else {}
        )
        parent_graph_id = parent_graph_id or self._spec_graph_id(previous_spec)
        if parent_graph_id:
            search.setdefault("parent_graph_id", parent_graph_id)
        attempts = int(previous_search.get("reseed_attempts", 0) or 0)
        attempts_max = int(
            previous_search.get(
                "reseed_attempts_max",
                self._spec_graph_reseed_attempts_max(),
            )
            or 0
        )
        if attempts:
            search["reseed_attempts"] = attempts
        if attempts_max:
            search["reseed_attempts_max"] = attempts_max
        cooldown_keys = self._current_failure_cooldown_keys()
        if cooldown_keys:
            search["cooldown_keys"] = cooldown_keys

    def _ensure_spec_search_metadata(
        self,
        spec: dict[str, Any],
        *,
        origin: str = "unknown",
        parent_graph_id: str = "",
    ) -> dict[str, Any]:
        if not isinstance(spec, dict) or int(spec.get("version", 1) or 1) < 2:
            return {}
        search = spec.get("search") if isinstance(spec.get("search"), dict) else {}
        if not search:
            search = {}
            spec["search"] = search
        if not str(search.get("graph_id") or "").strip():
            search["graph_id"] = self._next_spec_graph_id()
            search["created_loop"] = self.state.loop_count
            search["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if parent_graph_id and not str(search.get("parent_graph_id") or "").strip():
            search["parent_graph_id"] = parent_graph_id
        search.setdefault("origin", origin)
        search["last_seen_loop"] = self.state.loop_count
        search.setdefault(
            "spec_synth_calls_used_at_selection",
            self._spec_synth_call_count(),
        )
        search["spec_synth_calls_used"] = self._spec_synth_call_count()
        return search

    def _next_spec_graph_id(self) -> str:
        max_id = 0
        pattern = re.compile(r"^graph-(\d+)$")
        current_spec = self._load_run_spec(self._run_spec_path())
        if isinstance(current_spec, dict):
            match = pattern.match(self._spec_graph_id(current_spec))
            if match:
                max_id = max(max_id, int(match.group(1)))
        for record in self._read_spec_jsonl(self._spec_graph_candidates_path()):
            match = pattern.match(str(record.get("graph_id") or ""))
            if match:
                max_id = max(max_id, int(match.group(1)))
        sidecar_dir = self._spec_graph_candidate_sidecar_dir()
        if sidecar_dir.exists():
            for path in sidecar_dir.glob("graph-*.json"):
                match = pattern.match(path.stem)
                if match:
                    max_id = max(max_id, int(match.group(1)))
        return f"graph-{max_id + 1:04d}"

    def _append_spec_graph_candidate_event(
        self,
        spec: dict[str, Any],
        *,
        event: str,
        status: str,
        origin: str,
        quality_report: dict[str, Any] | None = None,
        issues: list[str] | None = None,
        parent_graph_id: str = "",
    ) -> dict[str, Any] | None:
        if not self._spec_mode_enabled() or not isinstance(spec, dict):
            return None
        search = self._ensure_spec_search_metadata(
            spec,
            origin=origin,
            parent_graph_id=parent_graph_id,
        )
        graph_id = str(search.get("graph_id") or "")
        if not graph_id:
            return None
        sidecar_path = self._spec_graph_candidate_sidecar_path(graph_id)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
        if self._spec_graph_candidate_event_seen(graph_id, event, status):
            return None
        issue_codes = self._spec_graph_candidate_issue_codes(
            quality_report=quality_report,
            issues=issues,
        )
        record = {
            "schema": "spec_graph_candidate.v1",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "event": event,
            "status": status,
            "origin": origin,
            "graph_id": graph_id,
            "parent_graph_id": search.get("parent_graph_id", ""),
            "spec_id": spec.get("spec_id"),
            "loop": self.state.loop_count,
            "fsm_step": self.state.fsm_step_count,
            "score": self._spec_graph_candidate_score(
                spec,
                quality_report,
                exclude_graph_id=graph_id,
            ),
            "graph_signature": self._spec_graph_signature(spec),
            "issue_codes": issue_codes,
            "spec_sidecar_path": str(
                sidecar_path.relative_to(self.state.repo_root)
                if sidecar_path.is_relative_to(self.state.repo_root)
                else sidecar_path
            ),
            "spec_synth_calls_used": self._spec_synth_call_count(),
            "spec_synth_call_budget": self._spec_synth_call_budget(),
        }
        path = self._spec_graph_candidates_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record

    def _spec_graph_candidate_event_seen(
        self,
        graph_id: str,
        event: str,
        status: str,
    ) -> bool:
        for record in self._read_spec_jsonl(
            self._spec_graph_candidates_path(),
            limit=_SPEC_JSONL_READ_LIMIT,
        ):
            if (
                str(record.get("graph_id") or "") == graph_id
                and str(record.get("event") or "") == event
                and str(record.get("status") or "") == status
            ):
                return True
        return False

    @staticmethod
    def _spec_graph_candidate_issue_codes(
        *,
        quality_report: dict[str, Any] | None = None,
        issues: list[str] | None = None,
    ) -> list[str]:
        codes: list[str] = []
        if isinstance(quality_report, dict):
            raw_codes = quality_report.get("issue_codes")
            if isinstance(raw_codes, list):
                codes.extend(str(code) for code in raw_codes if str(code).strip())
            for issue in quality_report.get("issues", []):
                if isinstance(issue, dict) and str(issue.get("code") or "").strip():
                    codes.append(str(issue["code"]))
        if issues:
            codes.extend(TodoLifecycleMixin._failure_issue_code(issue) for issue in issues)
        return list(dict.fromkeys(codes))

    def _spec_graph_candidate_score(
        self,
        spec: dict[str, Any],
        quality_report: dict[str, Any] | None = None,
        *,
        exclude_graph_id: str = "",
    ) -> dict[str, int]:
        tasks = spec.get("task_graph") if isinstance(spec.get("task_graph"), list) else []
        issues = (
            quality_report.get("issues", [])
            if isinstance(quality_report, dict)
            and isinstance(quality_report.get("issues"), list)
            else []
        )
        issue_codes = self._spec_graph_candidate_issue_codes(
            quality_report=quality_report
        )
        return {
            "runnable_tasks": len(self._schedulable_spec_tasks(tasks)),
            "quality_issues": len(issues),
            "design_issues": sum(
                1 for code in issue_codes if str(code).startswith("design_contract_")
            ),
            "cooldown_hits": self._spec_graph_cooldown_hits(spec),
            "duplicate_hits": self._spec_graph_duplicate_hits(
                spec,
                exclude_graph_id=exclude_graph_id,
            ),
        }

    def _spec_graph_cooldown_hits(self, spec: dict[str, Any]) -> int:
        cooldown_keys = self._current_failure_cooldown_keys()
        if not cooldown_keys:
            return 0
        cooldown_prefixes = {
            ":".join(key.split(":")[:2]) + ":"
            for key in cooldown_keys
            if len(key.split(":")) >= 2
        }
        if not cooldown_prefixes:
            return 0
        tasks = spec.get("task_graph") if isinstance(spec.get("task_graph"), list) else []
        hits = 0
        for task in tasks:
            if not isinstance(task, dict):
                continue
            region_hash = self._failure_signature_target_region_hash(
                self._failure_signature_list(task.get("target_regions")),
                target_symbols=self._failure_signature_list(task.get("target_symbols")),
            )
            tactic = str(task.get("tactic_stage") or "tactic-unknown")
            if f"{region_hash}:{tactic}:" in cooldown_prefixes:
                hits += 1
        return hits

    def _spec_graph_duplicate_hits(
        self,
        spec: dict[str, Any],
        *,
        exclude_graph_id: str = "",
    ) -> int:
        candidate_signature = set(self._spec_graph_signature(spec))
        if not candidate_signature:
            return 0
        existing_items: set[str] = set()
        for record in self._read_spec_jsonl(
            self._spec_graph_candidates_path(),
            limit=_SPEC_JSONL_READ_LIMIT,
        ):
            if exclude_graph_id and str(record.get("graph_id") or "") == exclude_graph_id:
                continue
            signature = record.get("graph_signature")
            if isinstance(signature, list):
                existing_items.update(str(item) for item in signature if str(item).strip())
        return len(candidate_signature & existing_items)

    @staticmethod
    def _spec_graph_candidate_sort_key(
        record: dict[str, Any],
        score: dict[str, Any] | None = None,
    ) -> tuple[int, int, int, int, int, int, str]:
        score = score if isinstance(score, dict) else record.get("score", {})
        if not isinstance(score, dict):
            score = {}
        return (
            int(score.get("quality_issues", 0) or 0),
            int(score.get("design_issues", 0) or 0),
            int(score.get("cooldown_hits", 0) or 0),
            int(score.get("duplicate_hits", 0) or 0),
            -int(score.get("runnable_tasks", 0) or 0),
            int(record.get("loop", 0) or 0),
            str(record.get("graph_id") or ""),
        )

    def _latest_spec_graph_candidate_events(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for record in self._read_spec_jsonl(
            self._spec_graph_candidates_path(),
            limit=_SPEC_JSONL_READ_LIMIT,
        ):
            graph_id = str(record.get("graph_id") or "")
            if graph_id:
                latest[graph_id] = record
        return latest

    def _load_spec_graph_candidate_sidecar(
        self,
        record: dict[str, Any],
    ) -> dict[str, Any] | None:
        raw_path = str(record.get("spec_sidecar_path") or "").strip()
        if not raw_path:
            return None
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.state.repo_root / path
        try:
            spec = json.loads(path.read_text(errors="replace"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return spec if isinstance(spec, dict) else None

    def _append_stale_spec_graph_candidate_record(
        self,
        record: dict[str, Any],
        *,
        parent_graph_id: str,
        issue_code: str,
    ) -> None:
        graph_id = str(record.get("graph_id") or "")
        if not graph_id:
            return
        if self._spec_graph_candidate_event_seen(
            graph_id,
            "candidate_rejected",
            "rejected_stale",
        ):
            return
        stale_record = {
            "schema": "spec_graph_candidate.v1",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "event": "candidate_rejected",
            "status": "rejected_stale",
            "origin": "graph_backtrack",
            "graph_id": graph_id,
            "parent_graph_id": parent_graph_id,
            "spec_id": record.get("spec_id"),
            "loop": self.state.loop_count,
            "fsm_step": self.state.fsm_step_count,
            "score": record.get("score", {}),
            "graph_signature": record.get("graph_signature", []),
            "issue_codes": [issue_code],
            "spec_sidecar_path": record.get("spec_sidecar_path", ""),
            "spec_synth_calls_used": self._spec_synth_call_count(),
            "spec_synth_call_budget": self._spec_synth_call_budget(),
        }
        path = self._spec_graph_candidates_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stale_record, ensure_ascii=False, sort_keys=True) + "\n")

    def _spec_graph_reseed_attempts_max(self) -> int:
        workflow = self.config.get("workflow", {})
        return int(workflow.get("spec_graph_reseed_attempts", 0) or 0)

    def _current_failure_cooldown_keys(self, limit: int = 8) -> list[str]:
        keys: list[str] = []
        for record in reversed(
            self._read_spec_jsonl(
                self._failure_signature_path(),
                limit=_SPEC_JSONL_READ_LIMIT,
            )
        ):
            key = str(record.get("cooldown_key") or "").strip()
            if key and key not in keys:
                keys.append(key)
            if len(keys) >= limit:
                break
        keys.reverse()
        return keys

    def _maybe_select_backtrackable_spec_graph(
        self,
        current_spec: dict[str, Any],
    ) -> bool:
        current_graph_id = self._spec_graph_id(current_spec)
        latest = self._latest_spec_graph_candidate_events()
        valid_candidates: list[
            tuple[
                tuple[int, int, int, int, int, int, str],
                dict[str, Any],
                dict[str, Any],
                dict[str, Any],
                dict[str, int],
            ]
        ] = []
        for graph_id, record in latest.items():
            if graph_id == current_graph_id:
                continue
            if str(record.get("status") or "") != "backtrackable":
                continue
            candidate = self._load_spec_graph_candidate_sidecar(record)
            if not isinstance(candidate, dict):
                self._append_stale_spec_graph_candidate_record(
                    record,
                    parent_graph_id=current_graph_id,
                    issue_code="missing_or_invalid_sidecar",
                )
                continue
            quality_report = self._spec_quality_report(candidate, attempt=0)
            if self._spec_quality_report_failed(quality_report):
                self._append_spec_graph_candidate_event(
                    candidate,
                    event="candidate_rejected",
                    status="rejected_stale",
                    origin="graph_backtrack",
                    quality_report=quality_report,
                    parent_graph_id=current_graph_id,
                )
                continue
            candidate_tasks = (
                candidate.get("task_graph")
                if isinstance(candidate.get("task_graph"), list)
                else []
            )
            if not self._schedulable_spec_tasks(candidate_tasks):
                self._append_spec_graph_candidate_event(
                    candidate,
                    event="candidate_rejected",
                    status="rejected_stale",
                    origin="graph_backtrack",
                    issues=["backtrackable graph has no schedulable task"],
                    parent_graph_id=current_graph_id,
                )
                continue
            score = self._spec_graph_candidate_score(
                candidate,
                quality_report,
                exclude_graph_id=graph_id,
            )
            valid_candidates.append(
                (
                    self._spec_graph_candidate_sort_key(record, score),
                    record,
                    candidate,
                    quality_report,
                    score,
                )
            )
        if not valid_candidates:
            return False
        _, record, candidate, quality_report, score = sorted(
            valid_candidates,
            key=lambda item: item[0],
        )[0]
        self._append_spec_graph_candidate_event(
            candidate,
            event="candidate_selected",
            status="selected_backtrack",
            origin="graph_backtrack",
            quality_report=quality_report,
            parent_graph_id=current_graph_id,
        )
        self._persist_run_spec(candidate)
        self._append_spec_progress_event(
            "graph_backtracked",
            candidate,
            extra={
                "from_graph_id": current_graph_id,
                "selected_graph_id": self._spec_graph_id(candidate),
                "selection_score": score,
                "source_graph_candidate_event": record.get("event", ""),
                "remaining_loops": self._spec_remaining_loop_budget(),
            },
        )
        self.state.notes.append(
            "Selected backtrackable spec graph: " + self._spec_graph_id(candidate)
        )
        self.state.current = AgentStateName.SCHEDULE
        return True

    def _maybe_request_spec_graph_reseed(
        self,
        spec: dict[str, Any],
        tasks: list[Any],
    ) -> bool:
        attempts_max = self._spec_graph_reseed_attempts_max()
        if attempts_max <= 0:
            return False
        if self._spec_global_loop_cap_reached():
            return False
        search = spec.get("search") if isinstance(spec.get("search"), dict) else {}
        if not search:
            search = {}
            spec["search"] = search
        attempts = int(search.get("reseed_attempts", 0) or 0)
        if attempts >= attempts_max:
            return False
        next_attempt = attempts + 1
        search["reseed_attempts"] = next_attempt
        search["reseed_attempts_max"] = attempts_max
        cooldown_keys = self._current_failure_cooldown_keys()
        if cooldown_keys:
            search["cooldown_keys"] = cooldown_keys
        self.state.scratch["spec_rewrite_focus"] = self._spec_graph_reseed_focus(
            spec,
            tasks,
            reseed_attempt=next_attempt,
            reseed_attempts_max=attempts_max,
            cooldown_keys=cooldown_keys,
        )
        self.state.scratch.pop("spec_rewrite_target_task_id", None)
        self.state.scratch["spec_graph_generation_origin"] = (
            "reseed_after_graph_frontier_exhausted"
        )
        self.state.scratch["spec_graph_parent_graph_id"] = self._spec_graph_id(spec)
        self._persist_run_spec(spec)
        self._append_spec_progress_event(
            "graph_reseed_requested",
            spec,
            extra={
                "graph_id": self._spec_graph_id(spec),
                "reseed_attempt": next_attempt,
                "reseed_attempts_max": attempts_max,
                "cooldown_keys": cooldown_keys,
                "remaining_loops": self._spec_remaining_loop_budget(),
            },
        )
        self.state.notes.append(
            f"Spec graph frontier exhausted; requesting reseed "
            f"{next_attempt}/{attempts_max}"
        )
        self.state.current = AgentStateName.SPEC_SYNTH
        return True

    def _maybe_recover_spec_search_frontier(
        self,
        spec: dict[str, Any],
        tasks: list[Any],
    ) -> bool:
        if self._maybe_select_backtrackable_spec_graph(spec):
            return True
        return self._maybe_request_spec_graph_reseed(spec, tasks)

    def _spec_graph_reseed_focus(
        self,
        spec: dict[str, Any],
        tasks: list[Any],
        *,
        reseed_attempt: int,
        reseed_attempts_max: int,
        cooldown_keys: list[str],
    ) -> str:
        task_summaries: list[dict[str, Any]] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            compact = {
                "task_id": task.get("task_id"),
                "status": task.get("status"),
                "title": task.get("title"),
                "target_regions": task.get("target_regions"),
                "target_symbols": task.get("target_symbols"),
                "tactic_stage": task.get("tactic_stage"),
                "decision_hint": task.get("decision_hint"),
                "last_observation": task.get("last_observation"),
            }
            task_summaries.append(
                {
                    key: value
                    for key, value in compact.items()
                    if value not in (None, "", [], {})
                }
            )
        summary_limit = int(
            self.config.get("workflow", {}).get(
                "spec_graph_reseed_task_summary_limit",
                8,
            )
            or 8
        )
        if summary_limit > 0:
            task_summaries = task_summaries[-summary_limit:]
        signatures = self._read_spec_jsonl(
            self._failure_signature_path(),
            limit=_SPEC_JSONL_READ_LIMIT,
        )[-6:]
        compact_signatures = [
            {
                key: record.get(key)
                for key in (
                    "failure_class",
                    "issue_code",
                    "issue_scope",
                    "target_regions",
                    "target_symbols",
                    "tactic_stage",
                    "cooldown_key",
                    "summary",
                )
                if record.get(key) not in (None, "", [], {})
            }
            for record in signatures
        ]
        return "\n\n".join(
            [
                "The selected spec graph has no runnable frontier. Generate a new "
                "candidate graph instead of repairing the same task graph.",
                f"Reseed attempt {reseed_attempt}/{reseed_attempts_max}.",
                "Do not repeat any failed shape identified by cooldown_keys. The new "
                "graph must include at least one runnable local_edit task or a "
                "materially narrower structural_probe with a concrete diff contract.",
                "Preserve closed/survivor evidence as facts only; do not copy closed "
                "tasks as already-completed graph nodes.",
                "Current exhausted graph tasks:",
                json.dumps(task_summaries, ensure_ascii=False, indent=2),
                "Recent failure signatures:",
                json.dumps(compact_signatures, ensure_ascii=False, indent=2),
                "Cooldown keys banned for this reseed:",
                json.dumps(cooldown_keys, ensure_ascii=False, indent=2),
            ]
        )

    @staticmethod
    def _spec_graph_signature(spec: dict[str, Any]) -> list[str]:
        tasks = spec.get("task_graph") if isinstance(spec.get("task_graph"), list) else []
        signature: list[str] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            raw_regions = task.get("target_regions")
            if isinstance(raw_regions, list) and raw_regions:
                regions = [
                    str(region) for region in raw_regions if str(region).strip()
                ]
            else:
                deliverables = task.get("deliverables")
                regions = (
                    [
                        str(deliverable)
                        for deliverable in deliverables
                        if str(deliverable).strip()
                    ]
                    if isinstance(deliverables, list)
                    else []
                )
            tactic = str(task.get("tactic_stage") or task.get("risk_level") or "")
            if not regions and tactic:
                regions = [""]
            for region in regions:
                signature.append(f"{region}:{tactic}")
        return sorted(dict.fromkeys(signature))

    def _merge_targeted_spec_rewrite(
        self,
        previous_spec: dict[str, Any],
        rewrite_spec: dict[str, Any],
        target_task_id: str,
    ) -> tuple[dict[str, Any], list[str]]:
        previous_tasks = [
            task
            for task in previous_spec.get("task_graph", [])
            if isinstance(task, dict)
        ]
        rewrite_tasks = [
            task
            for task in rewrite_spec.get("task_graph", [])
            if isinstance(task, dict)
        ]
        previous_target = self._spec_task_by_id(previous_tasks, target_task_id)
        target_origin = self._spec_rewrite_origin_task_id(previous_target, target_task_id)
        replacement_tasks, additional_tasks = self._split_rewrite_replacement_tasks(
            rewrite_tasks,
            target_task_id,
            target_origin,
            {str(task.get("task_id") or "") for task in previous_tasks},
        )
        if not replacement_tasks and additional_tasks:
            replacement_tasks = additional_tasks
            additional_tasks = []
        inherited_attempts = 0
        if isinstance(previous_target, dict) and isinstance(
            previous_target.get("design_contract"), dict
        ):
            inherited_attempts = int(
                previous_target["design_contract"].get("rewrite_attempts", 0) or 0
            )
        inherited_contract_drift_attempts = 0
        if isinstance(previous_target, dict) and isinstance(
            previous_target.get("contract_rewrite"), dict
        ):
            inherited_contract_drift_attempts = int(
                previous_target["contract_rewrite"].get("rewrite_attempts", 0) or 0
            )
        for task in replacement_tasks:
            if str(task.get("task_id") or "") != target_task_id:
                task["replaces_task_id"] = target_origin
            if inherited_attempts:
                task["design_contract"] = {
                    "status": "inherited",
                    "rewrite_attempt_key": target_origin,
                    "rewrite_attempts": inherited_attempts,
                }
            if inherited_contract_drift_attempts:
                task["contract_rewrite"] = {
                    "status": "inherited",
                    "rewrite_attempt_key": target_origin,
                    "rewrite_attempts": inherited_contract_drift_attempts,
                }
        merged_tasks: list[dict[str, Any]] = []
        preserved_task_ids: list[str] = []
        inserted_replacements = False
        for previous_task in previous_tasks:
            previous_id = str(previous_task.get("task_id") or "")
            if previous_id == target_task_id:
                if replacement_tasks:
                    merged_tasks.extend(copy.deepcopy(replacement_tasks))
                    inserted_replacements = True
                else:
                    retired = copy.deepcopy(previous_task)
                    retired["status"] = "failed_design"
                    retired["design_contract"] = {
                        "status": "failed_design",
                        "issues": [
                            "targeted SPEC rewrite omitted the rejected task without a replacement"
                        ],
                        "rewrite_attempt_key": target_origin,
                        "rewrite_attempts": inherited_attempts,
                        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    }
                    retired["decision_hint"] = (
                        "targeted_rewrite_retired_without_replacement: keep this "
                        "design-invalid task out of scheduling and continue with "
                        "surviving sibling tasks."
                    )
                    merged_tasks.append(retired)
                continue
            preserved = copy.deepcopy(previous_task)
            preserved_task_ids.append(previous_id)
            if str(preserved.get("status", "open")) in {
                "open",
                "deferred",
                "in_progress",
                "needs_design",
                "needs_contract_rewrite",
            }:
                preserved["portfolio_preserved_after_rewrite"] = True
                preserve_hint = (
                    "portfolio_preserved_after_targeted_rewrite: sibling task was "
                    "kept by the controller because this rewrite targeted a different task."
                )
                existing_hint = str(preserved.get("decision_hint") or "").strip()
                preserved["decision_hint"] = (
                    f"{preserve_hint} {existing_hint}".strip()
                    if existing_hint
                    else preserve_hint
                )
            merged_tasks.append(preserved)
        if replacement_tasks and not inserted_replacements:
            merged_tasks = copy.deepcopy(replacement_tasks) + merged_tasks
        if additional_tasks:
            merged_tasks.extend(copy.deepcopy(additional_tasks))
        merged_spec = copy.deepcopy(rewrite_spec)
        merged_spec["task_graph"] = merged_tasks
        merged_spec["progress"] = self._run_spec_progress(merged_spec)
        return merged_spec, [task_id for task_id in preserved_task_ids if task_id]

    @staticmethod
    def _spec_task_by_id(tasks: list[dict[str, Any]], task_id: str) -> dict[str, Any] | None:
        for task in tasks:
            if str(task.get("task_id") or "") == task_id:
                return task
        return None

    @staticmethod
    def _spec_rewrite_origin_task_id(
        task: dict[str, Any] | None,
        fallback_task_id: str,
    ) -> str:
        if isinstance(task, dict):
            replaces = str(task.get("replaces_task_id") or "").strip()
            if replaces:
                return replaces
            task_id = str(task.get("task_id") or "").strip()
            if task_id:
                return task_id
        return fallback_task_id

    @staticmethod
    def _split_rewrite_replacement_tasks(
        rewrite_tasks: list[dict[str, Any]],
        target_task_id: str,
        target_origin: str,
        previous_task_ids: set[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        replacement_tasks: list[dict[str, Any]] = []
        additional_tasks: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for task in rewrite_tasks:
            task_id = str(task.get("task_id") or "")
            if task_id in seen_ids:
                continue
            seen_ids.add(task_id)
            replaces = str(task.get("replaces_task_id") or "").strip()
            if task_id == target_task_id or replaces in {target_task_id, target_origin}:
                replacement_tasks.append(copy.deepcopy(task))
                continue
            if task_id in previous_task_ids:
                continue
            additional_tasks.append(copy.deepcopy(task))
        return replacement_tasks, additional_tasks

    def _spec_rewrite_graph_contract_issues(
        self,
        previous_spec: dict[str, Any],
        rewrite_spec: dict[str, Any],
        target_task_id: str,
    ) -> list[str]:
        workflow = self.config.get("workflow", {})
        if workflow.get("spec_rewrite_graph_gate", True) is False:
            return []
        previous_tasks = [
            task
            for task in previous_spec.get("task_graph", [])
            if isinstance(task, dict)
        ]
        rewritten_tasks = [
            task
            for task in rewrite_spec.get("task_graph", [])
            if isinstance(task, dict)
        ]
        issues: list[str] = []
        rewritten_ids = {
            str(task.get("task_id") or "")
            for task in rewritten_tasks
            if str(task.get("task_id") or "")
        }
        previous_sibling_ids = [
            str(task.get("task_id") or "")
            for task in previous_tasks
            if str(task.get("task_id") or "")
            and str(task.get("task_id") or "") != target_task_id
            and str(task.get("status", "open"))
            in {
                "open",
                "deferred",
                "in_progress",
                "needs_design",
                "needs_contract_rewrite",
            }
        ]
        missing_siblings = [
            task_id for task_id in previous_sibling_ids if task_id not in rewritten_ids
        ]
        if missing_siblings:
            issues.append(
                "targeted SPEC rewrite dropped runnable sibling tasks: "
                + ", ".join(missing_siblings[:6])
            )
        runnable = self._schedulable_spec_tasks(rewritten_tasks)
        if not runnable:
            issues.append("targeted SPEC rewrite produced no schedulable task")
        if self._spec_tactic_portfolio_enabled():
            min_runnable = int(workflow.get("spec_rewrite_min_runnable_tasks", 2) or 2)
            prior_runnable_siblings = [
                task
                for task in self._schedulable_spec_tasks(previous_tasks)
                if str(task.get("task_id") or "") != target_task_id
            ]
            if prior_runnable_siblings and len(runnable) < min_runnable:
                issues.append(
                    "targeted SPEC rewrite collapsed portfolio below "
                    f"{min_runnable} runnable tasks"
                )
        if len(runnable) == 1:
            task = runnable[0]
            if (
                str(task.get("risk_level") or "") == "structural"
                and self._structural_probe_scope_too_broad(str(task.get("edit_scope") or ""))
            ):
                issues.append(
                    "targeted SPEC rewrite left only one broad structural task"
                )
        return issues

    def _reject_spec_graph_rewrite(
        self,
        previous_spec: dict[str, Any],
        target_task_id: str,
        issues: list[str],
    ) -> None:
        spec = copy.deepcopy(previous_spec)
        tasks = spec.get("task_graph")
        target_task = None
        if isinstance(tasks, list):
            for task in tasks:
                if isinstance(task, dict) and str(task.get("task_id") or "") == target_task_id:
                    target_task = task
                    break
        if isinstance(target_task, dict):
            target_task["status"] = "deferred_design_invalid"
            target_task["design_contract"] = {
                "status": "deferred_design_invalid",
                "issues": issues,
                "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            target_task["decision_hint"] = (
                "graph_rewrite_rejected: SPEC rewrite would collapse or delete "
                "the runnable portfolio. Defer this design-invalid task while the "
                "controller tries a sibling graph or reseed with a materially "
                "narrower replacement."
            )
            spec["active_task_id"] = None
            spec["last_design_contract_issues"] = {
                "task_id": target_task.get("task_id"),
                "issues": issues,
                "graph_rewrite_rejected": True,
            }
        self._persist_run_spec(spec)
        self._append_spec_progress_event(
            "graph_rewrite_rejected",
            spec,
            target_task,
            extra={"issues": issues, "target_task_id": target_task_id},
        )
        self._append_failure_signature(
            phase="graph_rewrite",
            spec=spec,
            task=target_task,
            status="graph_rewrite_rejected",
            failure_class="design_rewrite_invalid",
            issue_code=self._failure_issue_code(issues),
            issue_scope="spec_graph",
            summary="; ".join(issues),
            extra={"issues": issues, "target_task_id": target_task_id},
        )
        self.state.notes.append(
            "Rejected SPEC graph rewrite: " + "; ".join(issues)
        )

    def _append_spec_progress_event(
        self,
        event: str,
        spec: dict[str, Any],
        task: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self._spec_mode_enabled():
            return
        path = self._workflow_artifact_path(
            "spec_progress_path", ".local_micro_agent/spec_progress.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "event": event,
            "state": str(self.state.current),
            "loop": self.state.loop_count,
            "fsm_step": self.state.fsm_step_count,
            "spec_id": spec.get("spec_id"),
            "active_task_id": spec.get("active_task_id"),
            "progress": self._run_spec_progress(spec),
        }
        if isinstance(task, dict):
            record["task_id"] = task.get("task_id")
            record["task_status"] = task.get("status")
            record["task_attempts"] = task.get("attempts")
            if event in {
                "design_rejected",
                "drift_recovery",
                "failed_design",
                "needs_design",
            }:
                record["task_title"] = task.get("title")
                record["task_edit_scope"] = task.get("edit_scope")
                record["task_risk_level"] = task.get("risk_level")
                record["task_tactic_stage"] = task.get("tactic_stage")
                record["task_target_symbols"] = task.get("target_symbols")
                record["task_target_regions"] = task.get("target_regions")
        if extra:
            record.update(extra)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _failure_signature_path(self) -> Path:
        return self._workflow_artifact_path(
            "failure_signatures_path",
            ".local_micro_agent/failure_signatures.jsonl",
        )

    def _append_failure_signature(
        self,
        *,
        phase: str,
        spec: dict[str, Any] | None = None,
        task: dict[str, Any] | None = None,
        status: str,
        failure_class: str,
        issue_code: str,
        issue_scope: str,
        summary: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self._spec_mode_enabled():
            return None
        spec = spec if isinstance(spec, dict) else {}
        task = task if isinstance(task, dict) else {}
        target_regions = self._failure_signature_list(task.get("target_regions"))
        target_symbols = self._failure_signature_list(task.get("target_symbols"))
        task_id = str(task.get("task_id") or "")
        graph_id = self._spec_graph_id(spec)
        tactic_stage = str(task.get("tactic_stage") or "")
        region_hash = self._failure_signature_target_region_hash(
            target_regions,
            target_symbols=target_symbols,
        )
        issue_code = self._normalize_failure_issue_code(issue_code)
        record = {
            "schema": "failure_signature.v1",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "created_loop": self.state.loop_count,
            "fsm_step": self.state.fsm_step_count,
            "graph_id": graph_id,
            "spec_id": spec.get("spec_id"),
            "phase": phase,
            "task_id": task_id,
            "status": status,
            "failure_class": failure_class,
            "issue_code": issue_code,
            "issue_scope": issue_scope,
            "target_regions": target_regions,
            "target_symbols": target_symbols,
            "tactic_stage": tactic_stage,
            "target_region_hash": region_hash,
            "episode_fingerprint": ":".join(
                [
                    graph_id,
                    phase,
                    task_id or "task-unknown",
                    tactic_stage or "tactic-unknown",
                    issue_code,
                    region_hash,
                ]
            ),
            "cooldown_key": ":".join(
                [region_hash, tactic_stage or "tactic-unknown", issue_code]
            ),
            "summary": self._truncate_text(str(summary or ""), 800),
        }
        if extra:
            record.update(
                {
                    key: value
                    for key, value in extra.items()
                    if value not in (None, "", [], {})
                }
            )
        path = self._failure_signature_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record

    @staticmethod
    def _failure_signature_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if value not in (None, "", [], {}):
            return [str(value)]
        return []

    @staticmethod
    def _failure_signature_target_region_hash(
        target_regions: list[str],
        *,
        target_symbols: list[str] | None = None,
    ) -> str:
        if target_regions:
            source = "|".join(target_regions)
        elif target_symbols:
            source = "symbols:" + "|".join(target_symbols)
        else:
            source = "unscoped"
        return hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]

    @staticmethod
    def _spec_graph_id(spec: dict[str, Any]) -> str:
        search = spec.get("search") if isinstance(spec.get("search"), dict) else {}
        graph_id = str(search.get("graph_id") or "").strip()
        if graph_id:
            return graph_id
        spec_id = str(spec.get("spec_id") or "").strip()
        return spec_id or "graph-unknown"

    @staticmethod
    def _failure_issue_code(issues: list[str] | str) -> str:
        if isinstance(issues, str):
            texts = [issues]
        else:
            texts = [str(issue) for issue in issues if str(issue).strip()]
        joined = " ".join(texts).lower()
        if "only one broad structural task" in joined:
            return "single_broad_structural_task"
        if "collapsed portfolio below" in joined:
            return "portfolio_collapsed_below_min_runnable"
        if "dropped runnable sibling" in joined:
            return "dropped_runnable_sibling_tasks"
        if "no schedulable task" in joined:
            return "no_schedulable_task"
        if not texts:
            return "unspecified"
        return TodoLifecycleMixin._normalize_failure_issue_code(texts[0])

    @staticmethod
    def _normalize_failure_issue_code(issue_code: Any) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(issue_code or "").lower()).strip("_")
        return normalized[:80] or "unspecified"

    def _run_spec_progress(self, spec: dict[str, Any]) -> dict[str, int]:
        tasks = spec.get("task_graph")
        if not isinstance(tasks, list):
            return {"total": 0, "closed": 0, "deferred": 0, "failed": 0}
        statuses = [str(task.get("status", "")) for task in tasks if isinstance(task, dict)]
        return {
            "total": len(statuses),
            "closed": statuses.count("closed"),
            "deferred": sum(1 for status in statuses if self._spec_status_is_deferred(status)),
            "failed": sum(1 for status in statuses if self._spec_status_is_failed(status)),
        }

    def _spec_design_contract_gate_enabled(self) -> bool:
        return bool(self.config.get("workflow", {}).get("spec_design_contract_gate"))

    @staticmethod
    def _spec_status_is_deferred(status: str) -> bool:
        return status in {
            "deferred",
            "deferred_contract_drift",
            "deferred_design",
            "deferred_design_invalid",
            "deferred_portfolio_exhausted",
        }

    @staticmethod
    def _spec_status_is_failed(status: str) -> bool:
        return status in {"failed", "failed_design"}

    def _spec_structural_risk_gate_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("spec_structural_risk_gate"))

    def _spec_task_design_contract_issues(
        self, spec: dict[str, Any], task: dict[str, Any]
    ) -> list[str]:
        if not self._spec_design_contract_gate_enabled():
            return []
        if self._spec_task_is_context_only(task):
            return []
        issues: list[str] = []
        if (
            not self._normalize_string_list(task.get("target_symbols"))
            and not self._normalize_string_list(task.get("target_regions"))
        ):
            issues.append("missing target_symbols or target_regions")
        task_invariants = self._normalize_string_list(task.get("preserved_invariants"))
        spec_invariants = self._normalize_string_list(spec.get("invariants"))
        if not task_invariants and not spec_invariants:
            issues.append("missing preserved_invariants")
        edit_scope = str(task.get("edit_scope") or "").strip()
        if not edit_scope:
            issues.append("missing edit_scope")
        elif self._spec_edit_scope_too_broad(edit_scope):
            issues.append("edit_scope too broad")
        validator = task.get("validator")
        failure_condition = (
            str(validator.get("failure_condition") or "").strip()
            if isinstance(validator, dict)
            else ""
        )
        if not failure_condition:
            issues.append("missing validator.failure_condition")
        if not str(task.get("correctness_rationale") or "").strip():
            issues.append("missing correctness_rationale")
        if not str(task.get("fallback_plan") or "").strip():
            issues.append("missing fallback_plan")
        issues.extend(self._spec_task_grounding_issues(task))
        issues.extend(self._spec_task_structural_risk_issues(task))
        return issues

    def _spec_task_grounding_issues(self, task: dict[str, Any]) -> list[str]:
        if not self._spec_grounding_gate_enabled():
            return []
        facts = self._current_spec_grounding_facts()
        if not facts:
            return []
        writable = set(str(path) for path in facts.get("writable_files", []) or [])
        allowed_regions = set(
            str(region) for region in facts.get("allowed_target_regions", []) or []
        )
        read_only_symbols = set(
            str(region) for region in facts.get("read_only_symbols", []) or []
        )
        imported_symbols = [
            record
            for record in facts.get("imported_symbols", []) or []
            if isinstance(record, dict)
        ]
        imported_roots = {
            str(record.get("symbol") or "")
            for record in imported_symbols
            if str(record.get("symbol") or "")
        }
        issues: list[str] = []
        for path in self._normalize_string_list(task.get("deliverables")):
            if not self._spec_path_is_writable(path, writable):
                issues.append(f"read_only_deliverable:{path}")
        target_regions = self._normalize_string_list(task.get("target_regions"))
        for region in target_regions:
            region_path = self._region_path(region)
            if region_path and not self._spec_path_is_writable(region_path, writable):
                issues.append(f"non_writable_target_region:{region}")
                continue
            if region_path and region_path.endswith(".py") and region not in allowed_regions:
                issues.append(f"unresolvable_target_region:{region}")
            elif region in read_only_symbols:
                issues.append(f"non_writable_target_region:{region}")
        for symbol in self._normalize_string_list(task.get("target_symbols")):
            symbol_root = self._symbol_root(symbol)
            if symbol in read_only_symbols:
                issues.append(f"non_writable_symbol:{symbol}")
            elif symbol_root in imported_roots:
                issues.append(f"imported_symbol_targeted:{symbol}")
            elif "::" in symbol:
                region_path = self._region_path(symbol)
                if region_path and not self._spec_path_is_writable(region_path, writable):
                    issues.append(f"non_writable_symbol:{symbol}")
                elif region_path and region_path.endswith(".py") and symbol not in allowed_regions:
                    issues.append(f"unresolvable_target_region:{symbol}")
        contract = task.get("probe_diff_contract")
        if isinstance(contract, dict):
            writable_probe_regions: list[str] = []
            for key in ("allowed_regions", "expected_changed_regions"):
                writable_probe_regions.extend(self._normalize_string_list(contract.get(key)))
            for region in writable_probe_regions:
                region_path = self._region_path(region)
                if region_path and not self._spec_path_is_writable(region_path, writable):
                    issues.append(f"non_writable_probe_region:{region}")
                elif (
                    region_path
                    and region_path.endswith(".py")
                    and region not in allowed_regions
                    and region not in read_only_symbols
                ):
                    issues.append(f"unresolvable_probe_region:{region}")
            for region in self._normalize_string_list(contract.get("required_unchanged_regions")):
                region_path = self._region_path(region)
                if (
                    region_path
                    and region_path.endswith(".py")
                    and region not in allowed_regions
                    and region not in read_only_symbols
                ):
                    issues.append(f"unresolvable_probe_region:{region}")
            for region in self._normalize_string_list(
                contract.get("expected_changed_regions")
            ):
                if not self._region_matches_task_targets(region, target_regions):
                    issues.append(f"probe_contract_region_mismatch:{region}")
        return sorted(dict.fromkeys(issues))

    @staticmethod
    def _region_path(region: str) -> str:
        return region.split("::", 1)[0].strip() if "::" in region else region.strip()

    @staticmethod
    def _symbol_root(symbol: str) -> str:
        raw = symbol.split("::", 1)[-1]
        return raw.split(".", 1)[0].strip()

    @staticmethod
    def _region_matches_task_targets(region: str, target_regions: list[str]) -> bool:
        if not target_regions:
            return True
        for target in target_regions:
            if region == target:
                return True
            if "::" not in region or "::" not in target:
                continue
            region_path, region_symbol = region.split("::", 1)
            target_path, target_symbol = target.split("::", 1)
            if region_path == target_path and region_symbol.startswith(target_symbol + "."):
                return True
        return False

    @staticmethod
    def _spec_edit_scope_too_broad(edit_scope: str) -> bool:
        text = edit_scope.lower()
        broad_patterns = (
            r"\brewrite\b",
            r"\bentire\b",
            r"\bwhole\b",
            r"\ball\b",
            r"\boptimi[sz]e (the )?(algorithm|program|codebase|hot path)\b",
        )
        return any(re.search(pattern, text) for pattern in broad_patterns)

    def _spec_task_structural_risk_issues(self, task: dict[str, Any]) -> list[str]:
        if not self._spec_structural_risk_gate_enabled():
            return []

        issues: list[str] = []
        risk_level = str(task.get("risk_level", "")).strip().lower()
        stage = str(task.get("tactic_stage", "")).strip().lower()
        obvious_structural = self._spec_task_has_obvious_structural_action(task)

        if risk_level not in {"local", "structural"}:
            issues.append("missing risk_level local|structural")
        issues.extend(self._spec_task_risk_evidence_issues(task))

        if risk_level == "local":
            if stage != "local_edit":
                issues.append("local task must use tactic_stage=local_edit")
            if obvious_structural:
                issues.append(
                    "local risk_level contradicts structural action in task scope"
                )
            return issues

        is_structural = risk_level == "structural" or obvious_structural
        if not is_structural:
            return issues

        if risk_level != "structural":
            issues.append("structural task must declare risk_level=structural")
        if stage not in {"structural_probe", "structural_expand"}:
            issues.append("structural task must use tactic_stage=structural_probe")
        attempts = int(task.get("attempts", 0) or 0)
        budget = task.get("budget")
        if isinstance(budget, dict):
            attempts = max(attempts, int(budget.get("attempts_used", 0) or 0))
        if attempts <= 0 and stage == "structural_expand":
            issues.append("first structural attempt must use structural_probe")
        if not str(task.get("probe_plan", "")).strip():
            issues.append("missing probe_plan for structural task")
        workflow = self.config.get("workflow", {})
        if (
            stage == "structural_probe"
            and workflow.get("spec_probe_diff_contract_required") is True
            and not task.get("probe_diff_contract")
        ):
            issues.append("missing probe_diff_contract for structural probe")
        if not self._normalize_string_list(task.get("invariant_evidence")):
            issues.append("missing invariant_evidence for structural task")
        if self._structural_probe_scope_too_broad(str(task.get("edit_scope") or "")):
            issues.append("structural edit_scope too broad; start with one reversible probe")
        shrink_plan = str(task.get("rollback_or_shrink_plan") or "").strip()
        if not shrink_plan:
            issues.append("missing rollback_or_shrink_plan for structural task")
        elif not self._plan_mentions_shrink_or_probe(shrink_plan):
            issues.append("rollback_or_shrink_plan must describe a smaller/guarded probe")
        return issues

    def _spec_task_has_structural_risk(self, task: dict[str, Any]) -> bool:
        if str(task.get("risk_level", "")).strip().lower() == "structural":
            return True
        return self._spec_task_has_obvious_structural_action(task)

    def _spec_task_risk_evidence_issues(self, task: dict[str, Any]) -> list[str]:
        evidence = task.get("risk_evidence")
        if not isinstance(evidence, dict) or not evidence:
            return ["missing risk_evidence"]
        field = str(evidence.get("field") or "").strip()
        quote = str(evidence.get("quote") or "").strip()
        explanation = str(evidence.get("explanation") or "").strip()
        issues: list[str] = []
        if field not in self._risk_evidence_fields():
            issues.append("risk_evidence.field must name an actionable task field")
        if not quote:
            issues.append("missing risk_evidence.quote")
        elif field in self._risk_evidence_fields() and not self._risk_quote_in_field(
            task, field, quote
        ):
            issues.append("risk_evidence.quote must appear in the named field")
        if not explanation:
            issues.append("missing risk_evidence.explanation")
        return issues

    @staticmethod
    def _risk_quote_in_field(task: dict[str, Any], field: str, quote: str) -> bool:
        value = str(task.get(field) or "")
        return quote in value or quote.lower() in value.lower()

    def _spec_task_has_obvious_structural_action(self, task: dict[str, Any]) -> bool:
        text = " ".join(
            str(task.get(key, "") or "")
            for key in (
                "title",
                "strategy_axis",
                "family_key",
                "expected_signal",
                "edit_scope",
            )
        ).lower()
        patterns = (
            r"\b(rewrite|replace|refactor|restructure)\b.*\b(function|class|method|module|loop|pipeline|algorithm|hot path)\b",
            r"\b(entire|whole|all)\b.*\b(function|class|method|module|loop|pipeline|algorithm|hot path)\b",
            r"\breorder(?:ing)?\b",
            r"\breschedul(?:e|ing)\b",
            r"\bparallel(?:ize|ise|ization|isation)\b",
            r"\bconcurrent\b",
            r"\b(batch|bundle|group|merge|split)\b.*\b(operation|instruction|task|item|loop|work|state|request|event)s?\b",
            r"\b(schedule|reschedule)\b.*\b(operation|instruction|task|work|event|side[- ]?effect)s?\b",
            r"\bstate\s+(lifecycle|machine|transition|ordering)\b",
            r"\bmove\b.*\b(state|effect|write|read|operation|logic)\b",
            r"\b(change|alter|replace)\b.*\b(data\s*flow|control\s*flow|loop|order|ordering|state)\b",
            r"\b(change|alter|replace|update|remove|add)\b.*\b(signature|call\s*site|callsite|api|argument|parameter|schema|contract)\b",
            r"\b(signature|call\s*site|callsite|api|argument|parameter|schema|contract)\b.*\b(change|alter|replace|update|remove|add)\b",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @classmethod
    def _structural_probe_scope_too_broad(cls, edit_scope: str) -> bool:
        text = edit_scope.lower()
        patterns = (
            r"\b(rewrite|replace|refactor|restructure)\b.*\b(function|class|method|module|loop|pipeline|algorithm|hot path)\b",
            r"\b(entire|whole|all)\b.*\b(function|class|method|module|loop|pipeline|algorithm|hot path)\b",
            r"\b(across|throughout)\b.*\b(file|module|codebase|pipeline|system)\b",
            r"\bchange\b.*\b(data\s*flow|control\s*flow|state lifecycle|ordering)\b",
            r"\b(replace|remove|rewrite)\s+all\b",
        )
        return any(re.search(pattern, text) for pattern in patterns) or (
            cls._structural_probe_action_count(text) >= 3
        )

    @staticmethod
    def _structural_probe_action_count(text: str) -> int:
        actions = re.findall(
            r"\b(add|allocate|cache|change|decrement|group|hoist|increment|"
            r"inline|initiali[sz]e|merge|move|remove|replace|rewrite|split|"
            r"update)\b",
            text.lower(),
        )
        return len(actions)

    @staticmethod
    def _plan_mentions_shrink_or_probe(text: str) -> bool:
        lowered = text.lower()
        strong_markers = (
            "shrink",
            "smaller",
            "narrow",
            "probe",
            "fallback branch",
            "feature flag",
            "isolate",
        )
        if any(marker in lowered for marker in strong_markers):
            return True
        if re.search(r"\b(revert|restore|rollback|roll\s+back)\b", lowered):
            return False
        return bool(
            re.search(r"\bsingle\b.*\b(guard(?:ed)?|branch|path|probe)\b", lowered)
            or re.search(r"\b(guard(?:ed)?|branch|path|probe)\b.*\bsingle\b", lowered)
        )

    def _reject_spec_task_for_design_contract(
        self, spec: dict[str, Any], task: dict[str, Any], issues: list[str]
    ) -> None:
        workflow = self.config.get("workflow", {})
        max_rewrites = int(workflow.get("spec_design_contract_rewrite_attempts", 2) or 0)
        task_id = str(task.get("task_id") or "")
        attempt_key = self._spec_rewrite_origin_task_id(task, task_id)
        prior_contract = task.get("design_contract")
        persisted_attempts = (
            int(prior_contract.get("rewrite_attempts", 0) or 0)
            if isinstance(prior_contract, dict)
            else 0
        )
        attempts_by_task = self.state.scratch.setdefault(
            "spec_design_contract_rewrite_attempts_by_task",
            {},
        )
        if not isinstance(attempts_by_task, dict):
            attempts_by_task = {}
            self.state.scratch["spec_design_contract_rewrite_attempts_by_task"] = (
                attempts_by_task
            )
        scratch_attempts = int(attempts_by_task.get(attempt_key, 0) or 0)
        attempts = max(scratch_attempts, persisted_attempts)
        next_attempt = attempts + 1
        task["status"] = "needs_design"
        task["design_contract"] = {
            "status": "rejected",
            "issues": issues,
            "rewrite_attempt_key": attempt_key,
            "rewrite_attempts": next_attempt,
            "rewrite_attempts_max": max_rewrites,
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        task["decision_hint"] = (
            "spec_design_contract_incomplete: rewrite this task with target symbols/"
            "regions, preserved invariants, a small edit scope, validator failure "
            "condition, correctness rationale, fallback plan, and structural probe "
            "metadata when the task changes ordering/state/dataflow before CODE."
        )
        spec["active_task_id"] = None
        spec["last_design_contract_issues"] = {
            "task_id": task.get("task_id"),
            "issues": issues,
        }
        self._persist_run_spec(spec)
        self._append_spec_progress_event(
            "design_rejected",
            spec,
            task,
            extra={"issues": issues, "rewrite_attempt": next_attempt},
        )
        self.state.notes.append(
            f"Spec task {task.get('task_id')} failed design contract gate: "
            + "; ".join(issues)
        )
        if attempts < max_rewrites:
            attempts_by_task[attempt_key] = next_attempt
            self.state.scratch["spec_rewrite_focus"] = self._spec_design_rewrite_focus(
                task,
                issues,
            )
            self.state.scratch["spec_rewrite_target_task_id"] = task_id
            self.state.current = AgentStateName.SPEC_SYNTH
            return
        if self._spec_gate_soft_fallback_enabled() and self.state.loop_count == 0:
            task["status"] = "open"
            task["design_contract"]["status"] = "soft_fallback_advisory"
            task["design_contract"]["soft_fallback_reason"] = (
                "design_contract_exhausted_before_code"
            )
            task["design_contract_advisory_once"] = True
            task["decision_hint"] = (
                "design_contract_soft_fallback: this task exhausted design rewrites "
                "before any CODE attempt. Execute once under advisory warning; all "
                "CODE/apply/test gates still apply."
            )
            spec["last_design_contract_issues"] = {
                "task_id": task.get("task_id"),
                "issues": issues,
                "soft_fallback": True,
            }
            self._persist_run_spec(spec)
            self._append_spec_progress_event(
                "design_soft_fallback",
                spec,
                task,
                extra={"issues": issues, "rewrite_attempt": next_attempt},
            )
            self.state.notes.append(
                f"Spec task {task.get('task_id')} exhausted design rewrites before "
                "CODE; scheduling one advisory soft-fallback attempt."
            )
            self.state.scratch.pop("spec_rewrite_focus", None)
            self.state.scratch.pop("spec_rewrite_target_task_id", None)
            self.state.current = AgentStateName.SCHEDULE
            return
        task["status"] = "failed_design"
        task["design_contract"]["status"] = "failed_design"
        task["decision_hint"] = (
            "design_contract_exhausted: this task could not be rewritten into a "
            "bounded executable unit. Do not schedule it again unless a future "
            "SPEC rewrite supplies a materially narrower target, invariant, "
            "validator, and risk contract."
        )
        spec["last_design_contract_issues"] = {
            "task_id": task.get("task_id"),
            "issues": issues,
            "exhausted": True,
        }
        self._persist_run_spec(spec)
        self._append_spec_progress_event(
            "failed_design",
            spec,
            task,
            extra={"issues": issues, "rewrite_attempt": next_attempt},
        )
        self.state.notes.append(
            f"Spec task {task.get('task_id')} failed design contract after "
            f"{next_attempt} rewrite attempts; continuing with other schedulable tasks."
        )
        self.state.scratch.pop("spec_rewrite_focus", None)
        self.state.scratch.pop("spec_rewrite_target_task_id", None)
        self.state.current = AgentStateName.SCHEDULE

    def _spec_design_rewrite_focus(
        self, task: dict[str, Any], issues: list[str], failure_summary: str = ""
    ) -> str:
        compact_task = {
            "task_id": task.get("task_id"),
            "title": task.get("title"),
            "strategy_axis": task.get("strategy_axis"),
            "family_key": task.get("family_key"),
            "expected_signal": task.get("expected_signal"),
            "risk_level": task.get("risk_level"),
            "tactic_stage": task.get("tactic_stage"),
            "risk_evidence": task.get("risk_evidence"),
            "probe_plan": task.get("probe_plan"),
            "probe_diff_contract": task.get("probe_diff_contract"),
            "invariant_evidence": task.get("invariant_evidence"),
            "rollback_or_shrink_plan": task.get("rollback_or_shrink_plan"),
            "issues": issues,
            "last_observation": task.get("last_observation"),
        }
        parts = [
            "The previous run-local spec was rejected before CODE because one task "
            "was not an executable design contract.",
            "Rewrite the spec tasks so every implementation task names concrete "
            "target_symbols or target_regions, preserved_invariants, edit_scope, "
            "validator.failure_condition, correctness_rationale, and fallback_plan.",
            "If the task changes behavior ordering, data/control flow, state lifecycle, "
            "scheduling, batching, parallelism, loop structure, or side-effect placement, "
            "rewrite it as risk_level=structural with tactic_stage=structural_probe, "
            "risk_evidence, probe_plan, probe_diff_contract, invariant_evidence, "
            "and rollback_or_shrink_plan. "
            "The first structural task should be a small reversible probe, not a full "
            "rewrite. risk_evidence must quote an actionable field such as title or "
            "edit_scope, not a correctness rationale or invariant.",
            "Do not regenerate the same rejected design shape. A previously "
            "failed_design task may reappear only if it has a materially narrower "
            "target region, clearer validator, and a risk contract that directly "
            "addresses the rejection issues.",
            "If the rejected task's last_observation has issue_scope=candidate_delta "
            "or repair_task_eligible=false, do not convert that transient rejected "
            "candidate failure into a repair/syntax-fix task. Treat it as negative "
            "candidate evidence: avoid the failed shape, retarget from fresh source, "
            "or choose a bounded sibling hypothesis. Create repair tasks only for "
            "current_repo issue_scope observations.",
            "Rejected task:",
            json.dumps(compact_task, ensure_ascii=False, indent=2),
        ]
        if failure_summary.strip():
            parts.extend(["Latest failure summary:", failure_summary.strip()])
        return "\n\n".join(parts)

    def _spec_design_failure_memory_context(self) -> str:
        workflow = self.config.get("workflow", {})
        if workflow.get("spec_design_failure_memory", True) is False:
            return ""
        limit = int(workflow.get("spec_design_failure_memory_limit", 6) or 6)
        if limit <= 0:
            return ""
        path = self._workflow_artifact_path(
            "spec_progress_path", ".local_micro_agent/spec_progress.jsonl"
        )
        records: list[dict[str, Any]] = []
        try:
            lines = path.read_text(errors="replace").splitlines()
        except FileNotFoundError:
            lines = []
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") not in {"design_rejected", "failed_design"}:
                continue
            compact = {
                "event": record.get("event"),
                "spec_id": record.get("spec_id"),
                "task_id": record.get("task_id"),
                "title": record.get("task_title"),
                "edit_scope": record.get("task_edit_scope"),
                "risk_level": record.get("task_risk_level"),
                "tactic_stage": record.get("task_tactic_stage"),
                "target_symbols": record.get("task_target_symbols"),
                "target_regions": record.get("task_target_regions"),
                "issues": record.get("issues"),
                "rewrite_attempt": record.get("rewrite_attempt"),
            }
            records.append(
                {key: value for key, value in compact.items() if value not in (None, "", [], {})}
            )
            if len(records) >= limit:
                break
        if not records:
            return ""
        records.reverse()
        return (
            "Recent rejected design shapes from this run follow. Treat them as "
            "negative design memory, not as domain tactics or hidden answers. Do "
            "not regenerate the same shape unless the new task is materially "
            "narrower, has a clearer validator/failure condition, and addresses "
            "the listed issues. If no bounded/verifiable implementation unit can "
            "be formed, emit a context/read task instead of CODE work.\n"
            + json.dumps(records, ensure_ascii=False, indent=2)
        )

    def _spec_rewrite_focus_context(self) -> str:
        focus = self.state.scratch.get("spec_rewrite_focus")
        if not isinstance(focus, str) or not focus.strip():
            return ""
        return "Spec rewrite focus:\n" + focus.strip()

    def _correct_survivor_spec_context(self) -> str:
        workflow = self.config.get("workflow", {})
        if workflow.get("spec_include_correct_survivor_context", True) is False:
            return ""
        state_path = self._workflow_artifact_path(
            "last_correct_state_path",
            ".local_micro_agent/last_correct_state.json",
        )
        try:
            record = json.loads(state_path.read_text(errors="replace"))
        except (FileNotFoundError, json.JSONDecodeError):
            return ""
        if not isinstance(record, dict):
            return ""
        compact = {
            "candidate_id": record.get("candidate_id"),
            "status": record.get("status"),
            "metric": record.get("metric"),
            "reason": record.get("reason"),
            "strategy_axis": record.get("strategy_axis"),
            "strategy_axes": record.get("strategy_axes"),
            "region_keys": record.get("region_keys"),
            "changes": record.get("changes"),
            "failure_class": record.get("failure_class"),
            "stage_result": record.get("stage_result"),
            "summary": self._truncate_text(str(record.get("summary", "")), 600),
            "patch_path": record.get("patch_path"),
        }
        return (
            "Correctness-preserving survivor from this run follows. Treat it as "
            "safe composition evidence, not as a completed optimization. Later SPEC "
            "tasks should explicitly decide whether to build on it, narrow it, or "
            "discard it.\n"
            + json.dumps(
                {key: value for key, value in compact.items() if value not in (None, "", [], {})},
                ensure_ascii=False,
                indent=2,
            )
        )

    def _schedule_spec_task(self) -> None:
        spec = self.state.scratch.get("run_spec")
        if not isinstance(spec, dict):
            spec = self._load_run_spec(self._run_spec_path())
            if spec:
                spec = self._normalize_run_spec(spec)
        if not isinstance(spec, dict) or int(spec.get("version", 1) or 1) < 2:
            self.state.notes.append("Spec mode requires run_spec version 2")
            self.state.current = self._retry_or_fail_without_spec()
            return
        tasks = spec.get("task_graph")
        if not isinstance(tasks, list) or not tasks:
            self.state.current = self._retry_or_fail_without_spec()
            return
        self._clear_active_spec_task()
        if all(str(task.get("status")) == "closed" for task in tasks if isinstance(task, dict)):
            self._persist_run_spec(spec)
            self._append_spec_progress_event("done", spec)
            self.state.current = AgentStateName.DONE
            return
        if self._spec_global_loop_cap_reached():
            spec["progress"] = self._run_spec_progress(spec)
            spec["last_stop_reason"] = "max_code_test_loops"
            self._persist_run_spec(spec)
            self.state.notes.append(
                f"Spec mode reached max_code_test_loops={self.state.max_loops}"
            )
            self._append_spec_progress_event(
                "failed",
                spec,
                extra={"reason": "max_code_test_loops"},
            )
            self.state.current = AgentStateName.FAILED
            return
        open_candidates = self._schedulable_spec_tasks(tasks)
        if not open_candidates:
            restored = self._restore_deferred_spec_tasks(tasks)
            if restored:
                open_candidates = self._schedulable_spec_tasks(tasks)
        if not open_candidates and not self._spec_global_loop_cap_reached():
            reopened = self._reopen_failed_spec_prerequisites(tasks)
            if reopened:
                self._persist_run_spec(spec)
                self._append_spec_progress_event(
                    "reopened",
                    spec,
                    extra={
                        "reason": "failed_prerequisite_recovery",
                        "reopened_tasks": reopened,
                        "remaining_loops": self._spec_remaining_loop_budget(),
                    },
                )
                open_candidates = self._schedulable_spec_tasks(tasks)
        if not open_candidates and not self._spec_global_loop_cap_reached():
            relaxed = self._relax_failed_spec_dependencies(tasks)
            if relaxed:
                self._persist_run_spec(spec)
                self._append_spec_progress_event(
                    "dependencies_relaxed",
                    spec,
                    extra={
                        "relaxed_tasks": relaxed,
                        "remaining_loops": self._spec_remaining_loop_budget(),
                    },
                )
                open_candidates = self._schedulable_spec_tasks(tasks)
        if not open_candidates and not self._spec_global_loop_cap_reached():
            reopened = self._reopen_failed_spec_portfolio_tasks(tasks)
            if reopened:
                self._persist_run_spec(spec)
                self._append_spec_progress_event(
                    "portfolio_reopened",
                    spec,
                    extra={
                        "reopened_tasks": reopened,
                        "remaining_loops": self._spec_remaining_loop_budget(),
                    },
                )
                open_candidates = self._schedulable_spec_tasks(tasks)
            else:
                exhausted = self._defer_exhausted_spec_portfolio_tasks(tasks)
                if exhausted:
                    self._persist_run_spec(spec)
                    for task in tasks:
                        if (
                            isinstance(task, dict)
                            and str(task.get("task_id") or "") in exhausted
                        ):
                            self._append_failure_signature(
                                phase="portfolio_recovery",
                                spec=spec,
                                task=task,
                                status="deferred_portfolio_exhausted",
                                failure_class="portfolio_exhausted",
                                issue_code="portfolio_recovery_budget_exhausted",
                                issue_scope="spec_task",
                                summary=str(task.get("decision_hint") or ""),
                                extra={
                                    "recovery_rounds": task.get("recovery_rounds"),
                                    "remaining_loops": self._spec_remaining_loop_budget(),
                                },
                            )
                    self._append_spec_progress_event(
                        "portfolio_exhausted",
                        spec,
                        extra={
                            "exhausted_tasks": exhausted,
                            "remaining_loops": self._spec_remaining_loop_budget(),
                        },
                    )
        if not open_candidates:
            if self._maybe_recover_spec_search_frontier(spec, tasks):
                return
            blocked_extra = self._spec_blocked_event_extra(tasks, spec=spec)
            partial_success = (
                blocked_extra.get("stop_reason")
                in {
                    "partial_success_design_deferred",
                    "partial_success_search_frontier_exhausted",
                }
            )
            if partial_success:
                self._defer_design_failed_spec_tasks(tasks)
            spec["last_stop_reason"] = blocked_extra["stop_reason"]
            spec["progress"] = self._run_spec_progress(spec)
            self._persist_run_spec(spec)
            self.state.notes.append(
                "Spec scheduler found no runnable task: "
                + self._spec_blocked_task_summary(tasks)
            )
            self._append_spec_progress_event("blocked", spec, extra=blocked_extra)
            self.state.current = AgentStateName.DONE if partial_success else AgentStateName.FAILED
            return
        task = self._select_spec_task(tasks, open_candidates)
        design_issues = self._spec_task_design_contract_issues(spec, task)
        if design_issues:
            if (
                task.pop("design_contract_advisory_once", False)
                and self._spec_gate_soft_fallback_enabled()
                and self.state.loop_count == 0
            ):
                task["design_contract"] = {
                    "status": "soft_fallback_advisory",
                    "issues": design_issues,
                    "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                self._append_spec_progress_event(
                    "design_soft_fallback",
                    spec,
                    task,
                    extra={"issues": design_issues},
                )
            else:
                self._reject_spec_task_for_design_contract(spec, task, design_issues)
                return
        task["status"] = "in_progress"
        task["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        spec["active_task_id"] = task.get("task_id")
        self._persist_run_spec(spec)
        todo = self._spec_task_to_active_todo(task)
        self.state.scratch["active_todo"] = todo
        self.state.scratch["current_spec_task_id"] = task.get("task_id")
        self.state.scratch["current_spec_task"] = task
        self._persist_todo_plan(todo)
        self.state.notes.append(f"Scheduled spec task: {task.get('task_id')}")
        self._append_spec_progress_event("scheduled", spec, task)
        self.state.current = AgentStateName.TASK_READ

    def _retry_or_fail_without_spec(self) -> AgentStateName:
        if self.state.scratch.get("spec_synth_budget_exhausted"):
            self.state.notes.append("Spec mode has no v2 run_spec and SPEC call budget is exhausted")
            return AgentStateName.FAILED
        if self._spec_mode_enabled() and not self.state.scratch.get("spec_synth_attempted"):
            self.state.scratch["spec_synth_attempted"] = True
            self.state.notes.append("Spec mode has no v2 run_spec; attempting SPEC_SYNTH")
            return AgentStateName.SPEC_SYNTH
        return AgentStateName.FAILED

    def _spec_global_loop_cap_reached(self) -> bool:
        max_loops = self.state.max_loops
        return isinstance(max_loops, int) and max_loops >= 0 and self.state.loop_count >= max_loops

    def _spec_remaining_loop_budget(self) -> int | None:
        max_loops = self.state.max_loops
        if not isinstance(max_loops, int) or max_loops < 0:
            return None
        return max(0, max_loops - self.state.loop_count)

    @staticmethod
    def _spec_blocked_task_summary(tasks: list[Any]) -> str:
        closed = {
            str(task.get("task_id"))
            for task in tasks
            if isinstance(task, dict) and str(task.get("status")) == "closed"
        }
        blocked: list[str] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            status = str(task.get("status", "open"))
            task_id = str(task.get("task_id") or "<unknown>")
            if status in {"failed_design", "deferred_design_invalid"}:
                blocked.append(f"{task_id} status={status}")
                continue
            if status not in {
                "open",
                "deferred",
                "in_progress",
                "needs_design",
                "needs_contract_rewrite",
            }:
                continue
            deps = task.get("depends_on")
            missing = [
                str(dep)
                for dep in deps
                if str(dep) not in closed
            ] if isinstance(deps, list) else []
            if missing:
                blocked.append(f"{task_id} waiting on {', '.join(missing)}")
            else:
                blocked.append(f"{task_id} status={status}")
        return "; ".join(blocked[:6]) if blocked else "no open/deferred tasks"

    def _clear_active_spec_task(self) -> None:
        self.state.scratch.pop("current_spec_task_id", None)
        self.state.scratch.pop("current_spec_task", None)
        self.state.scratch.pop("active_todo", None)
        active_path = self._workflow_artifact_path(
            "active_todo_path", ".local_micro_agent/active_todo.json"
        )
        if active_path.exists():
            active_path.unlink()

    def _schedulable_spec_tasks(self, tasks: list[Any]) -> list[dict[str, Any]]:
        closed = {
            str(task.get("task_id"))
            for task in tasks
            if isinstance(task, dict) and str(task.get("status")) == "closed"
        }
        open_statuses = {"open"}
        if self._spec_design_contract_gate_enabled():
            open_statuses.add("needs_design")
        open_statuses.add("needs_contract_rewrite")
        candidates = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            if str(task.get("status", "open")) not in open_statuses:
                continue
            deps = task.get("depends_on")
            if isinstance(deps, list) and any(str(dep) not in closed for dep in deps):
                continue
            candidates.append(task)
        return candidates

    def _restore_deferred_spec_tasks(self, tasks: list[Any]) -> bool:
        restored = False
        for task in tasks:
            if not isinstance(task, dict):
                continue
            if str(task.get("status")) != "deferred":
                continue
            if task.get("deferred_revisited"):
                continue
            task["status"] = "open"
            task["deferred_revisited"] = True
            restored = True
        return restored

    def _reopen_failed_spec_prerequisites(self, tasks: list[Any]) -> list[str]:
        workflow = self.config.get("workflow", {})
        max_rounds = int(workflow.get("spec_task_recovery_rounds", 2) or 0)
        if max_rounds <= 0:
            return []
        blocking_ids = self._failed_spec_prerequisite_ids(tasks)
        if not blocking_ids:
            return []
        reopened: list[str] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            task_id = str(task.get("task_id") or "")
            if task_id not in blocking_ids or str(task.get("status")) != "failed":
                continue
            rounds = int(task.get("recovery_rounds", 0) or 0)
            if rounds >= max_rounds:
                continue
            budget = task.setdefault("budget", {})
            if not isinstance(budget, dict):
                budget = {}
                task["budget"] = budget
            prior_attempts = int(budget.get("attempts_used", task.get("attempts", 0)) or 0)
            task["attempts_total"] = int(task.get("attempts_total", 0) or 0) + prior_attempts
            budget["attempts_used"] = 0
            task["attempts"] = 0
            task["status"] = "open"
            task["recovery_rounds"] = rounds + 1
            observation = task.get("last_observation")
            summary = ""
            if isinstance(observation, dict):
                summary = self._truncate_text(str(observation.get("summary", "")), 320)
            hint = "recovery_after_failure"
            if summary:
                hint += f": {summary}"
            task["decision_hint"] = hint
            task["reopened_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self.state.notes.append(
                f"Reopened failed prerequisite spec task: {task_id} "
                f"(recovery {rounds + 1}/{max_rounds})"
            )
            reopened.append(task_id)
        return reopened

    def _relax_failed_spec_dependencies(self, tasks: list[Any]) -> list[str]:
        workflow = self.config.get("workflow", {})
        if not workflow.get("spec_relax_failed_dependencies_with_budget"):
            return []
        remaining = self._spec_remaining_loop_budget()
        if remaining is not None and remaining <= 0:
            return []
        failed = {
            str(task.get("task_id"))
            for task in tasks
            if isinstance(task, dict)
            and str(task.get("status")) in {"failed", "failed_design"}
        }
        if not failed:
            return []
        relaxed: list[str] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            if str(task.get("status", "open")) not in {"open", "deferred"}:
                continue
            deps = task.get("depends_on")
            if not isinstance(deps, list):
                continue
            kept = [str(dep) for dep in deps if str(dep) not in failed]
            if len(kept) == len(deps):
                continue
            task["depends_on"] = kept
            task_id = str(task.get("task_id") or "")
            task["decision_hint"] = (
                "dependency_relaxed_after_failed_prerequisite: try this tactic as an "
                "independent alternative; do not repeat the failed prerequisite patch."
            )
            self.state.notes.append(
                f"Relaxed failed dependencies for spec task: {task_id}"
            )
            relaxed.append(task_id)
        return relaxed

    def _reopen_failed_spec_portfolio_tasks(self, tasks: list[Any]) -> list[str]:
        workflow = self.config.get("workflow", {})
        if not workflow.get("spec_reopen_failed_portfolio_tasks"):
            return []
        remaining = self._spec_remaining_loop_budget()
        if remaining is not None and remaining <= 0:
            return []
        max_rounds = int(
            workflow.get(
                "spec_portfolio_recovery_rounds",
                workflow.get("spec_task_recovery_rounds", 0),
            )
            or 0
        )
        if max_rounds <= 0:
            return []
        batch_size = int(workflow.get("spec_portfolio_reopen_batch_size", 0) or 0)
        reopened: list[str] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            if str(task.get("status")) != "failed":
                continue
            deliverables = task.get("deliverables")
            if not isinstance(deliverables, list) or not any(
                str(item).strip() for item in deliverables
            ):
                continue
            rounds = int(task.get("recovery_rounds", 0) or 0)
            if rounds >= max_rounds:
                continue
            budget = task.setdefault("budget", {})
            if not isinstance(budget, dict):
                budget = {}
                task["budget"] = budget
            prior_attempts = int(budget.get("attempts_used", task.get("attempts", 0)) or 0)
            task["attempts_total"] = int(task.get("attempts_total", 0) or 0) + prior_attempts
            budget["attempts_used"] = 0
            task["attempts"] = 0
            task["status"] = "open"
            task["recovery_rounds"] = rounds + 1
            task["decision_hint"] = (
                "portfolio_revisit_after_failure: previous attempts for this tactic "
                "failed. Try a materially different local edit or narrower variant, "
                "using the latest observation as negative evidence."
            )
            task["reopened_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            task_id = str(task.get("task_id") or "")
            self.state.notes.append(
                f"Reopened failed portfolio spec task: {task_id} "
                f"(recovery {rounds + 1}/{max_rounds})"
            )
            reopened.append(task_id)
            if batch_size > 0 and len(reopened) >= batch_size:
                break
        return reopened

    def _defer_exhausted_spec_portfolio_tasks(self, tasks: list[Any]) -> list[str]:
        workflow = self.config.get("workflow", {})
        if not workflow.get("spec_reopen_failed_portfolio_tasks"):
            return []
        max_rounds = int(
            workflow.get(
                "spec_portfolio_recovery_rounds",
                workflow.get("spec_task_recovery_rounds", 0),
            )
            or 0
        )
        if max_rounds <= 0:
            return []
        exhausted: list[str] = []
        for task in tasks:
            if not isinstance(task, dict) or str(task.get("status")) != "failed":
                continue
            deliverables = task.get("deliverables")
            if not isinstance(deliverables, list) or not any(
                str(item).strip() for item in deliverables
            ):
                continue
            rounds = int(task.get("recovery_rounds", 0) or 0)
            if rounds < max_rounds:
                continue
            task["status"] = "deferred_portfolio_exhausted"
            task["portfolio_exhausted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            task["decision_hint"] = (
                "portfolio_recovery_exhausted: repeated failed or non-improving "
                "attempts exhausted the portfolio revisit budget. Move to another "
                "runnable task or stop cleanly instead of reopening this tactic."
            )
            task_id = str(task.get("task_id") or "")
            self.state.notes.append(
                f"Deferred exhausted portfolio spec task: {task_id} "
                f"(recovery {rounds}/{max_rounds})"
            )
            if task_id:
                exhausted.append(task_id)
        return exhausted

    @staticmethod
    def _failed_spec_prerequisite_ids(tasks: list[Any]) -> set[str]:
        failed = {
            str(task.get("task_id"))
            for task in tasks
            if isinstance(task, dict)
            and str(task.get("status")) in {"failed", "failed_design"}
        }
        blocking: set[str] = set()
        for task in tasks:
            if not isinstance(task, dict):
                continue
            status = str(task.get("status", "open"))
            if status not in {
                "open",
                "deferred",
                "in_progress",
                "needs_design",
                "needs_contract_rewrite",
            }:
                continue
            deps = task.get("depends_on")
            if not isinstance(deps, list):
                continue
            for dep in deps:
                dep_id = str(dep)
                if dep_id in failed:
                    blocking.add(dep_id)
        return blocking

    def _spec_blocked_event_extra(
        self,
        tasks: list[Any],
        *,
        spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        failed_prerequisites = sorted(self._failed_spec_prerequisite_ids(tasks))
        blocked_tasks = self._spec_blocked_task_ids(tasks)
        design_failed = self._design_failed_spec_task_ids(tasks)
        drift_deferred = self._contract_drift_deferred_spec_task_ids(tasks)
        portfolio_exhausted = self._portfolio_exhausted_spec_task_ids(tasks)
        remaining = self._spec_remaining_loop_budget()
        stop_reason = "no_recovery_possible"
        search = spec.get("search") if isinstance(spec, dict) and isinstance(spec.get("search"), dict) else {}
        reseed_attempts = int(search.get("reseed_attempts", 0) or 0)
        reseed_attempts_max = int(search.get("reseed_attempts_max", 0) or 0)
        reseed_exhausted = bool(
            reseed_attempts_max > 0 and reseed_attempts >= reseed_attempts_max
        )
        if reseed_exhausted and (design_failed or drift_deferred or portfolio_exhausted):
            stop_reason = (
                "partial_success_search_frontier_exhausted"
                if self._spec_has_closed_task(tasks)
                else "search_frontier_exhausted_after_graph_reseed_exhausted"
            )
        elif design_failed and self._all_remaining_spec_tasks_design_failed(tasks):
            stop_reason = (
                "partial_success_design_deferred"
                if self._spec_has_closed_task(tasks)
                else "spec_design_contract_incomplete"
            )
        elif (
            design_failed
            and drift_deferred
            and self._all_remaining_spec_tasks_design_or_drift_exhausted(tasks)
        ):
            stop_reason = (
                "partial_success_search_frontier_exhausted"
                if self._spec_has_closed_task(tasks)
                else "search_frontier_exhausted_after_design_invalid"
            )
        elif drift_deferred and self._all_remaining_spec_tasks_contract_drift_deferred(tasks):
            stop_reason = "no_runnable_tasks_after_drift_deferred"
        elif portfolio_exhausted and self._all_remaining_spec_tasks_exhausted(tasks):
            stop_reason = "no_runnable_tasks_after_portfolio_exhausted"
        elif (
            failed_prerequisites
            and remaining
            and remaining > 0
            and self._reopenable_failed_spec_prerequisite_ids(tasks)
        ):
            stop_reason = "dependency_blocked_before_budget_exhaustion"
        elif remaining and remaining > 0 and self._reopenable_failed_portfolio_task_ids(tasks):
            stop_reason = "portfolio_exhausted_before_budget_exhaustion"
        return {
            "stop_reason": stop_reason,
            "remaining_loops": remaining,
            "runnable_tasks_at_exit": 0,
            "blocked_tasks": blocked_tasks,
            "failed_prerequisites": failed_prerequisites,
            "design_failed_tasks": design_failed,
            "drift_deferred_tasks": drift_deferred,
            "portfolio_exhausted_tasks": portfolio_exhausted,
            "graph_reseed_attempts": reseed_attempts,
            "graph_reseed_attempts_max": reseed_attempts_max,
            "graph_reseed_exhausted": reseed_exhausted,
        }

    @staticmethod
    def _design_failed_spec_task_ids(tasks: list[Any]) -> list[str]:
        return [
            str(task.get("task_id") or "")
            for task in tasks
            if isinstance(task, dict)
            and str(task.get("status")) in {"failed_design", "deferred_design_invalid"}
            and str(task.get("task_id") or "")
        ]

    @staticmethod
    def _contract_drift_deferred_spec_task_ids(tasks: list[Any]) -> list[str]:
        return [
            str(task.get("task_id") or "")
            for task in tasks
            if isinstance(task, dict)
            and str(task.get("status")) == "deferred_contract_drift"
            and str(task.get("task_id") or "")
        ]

    @staticmethod
    def _portfolio_exhausted_spec_task_ids(tasks: list[Any]) -> list[str]:
        return [
            str(task.get("task_id") or "")
            for task in tasks
            if isinstance(task, dict)
            and str(task.get("status")) == "deferred_portfolio_exhausted"
            and str(task.get("task_id") or "")
        ]

    @staticmethod
    def _spec_has_closed_task(tasks: list[Any]) -> bool:
        return any(
            isinstance(task, dict) and str(task.get("status")) == "closed"
            for task in tasks
        )

    @staticmethod
    def _defer_design_failed_spec_tasks(tasks: list[Any]) -> None:
        for task in tasks:
            if not isinstance(task, dict):
                continue
            if str(task.get("status")) not in {"failed_design", "deferred_design_invalid"}:
                continue
            task["status"] = "deferred_design"
            task["decision_hint"] = (
                "design_deferred_after_partial_success: preserve the closed task "
                "result and retry this design in a future SPEC rewrite."
            )

    def _all_remaining_spec_tasks_design_failed(self, tasks: list[Any]) -> bool:
        saw_remaining = False
        for task in tasks:
            if not isinstance(task, dict):
                continue
            status = str(task.get("status", "open"))
            if status == "closed":
                continue
            saw_remaining = True
            if status not in {"failed_design", "deferred_design_invalid"}:
                return False
        return saw_remaining

    def _all_remaining_spec_tasks_contract_drift_deferred(self, tasks: list[Any]) -> bool:
        saw_remaining = False
        for task in tasks:
            if not isinstance(task, dict):
                continue
            status = str(task.get("status", "open"))
            if status == "closed":
                continue
            saw_remaining = True
            if status != "deferred_contract_drift":
                return False
        return saw_remaining

    def _all_remaining_spec_tasks_design_or_drift_exhausted(self, tasks: list[Any]) -> bool:
        exhausted_statuses = {
            "deferred_contract_drift",
            "deferred_design_invalid",
            "failed_design",
        }
        saw_remaining = False
        for task in tasks:
            if not isinstance(task, dict):
                continue
            status = str(task.get("status", "open"))
            if status == "closed":
                continue
            saw_remaining = True
            if status not in exhausted_statuses:
                return False
        return saw_remaining

    def _all_remaining_spec_tasks_exhausted(self, tasks: list[Any]) -> bool:
        exhausted_statuses = {
            "deferred_contract_drift",
            "deferred_portfolio_exhausted",
        }
        saw_remaining = False
        for task in tasks:
            if not isinstance(task, dict):
                continue
            status = str(task.get("status", "open"))
            if status == "closed":
                continue
            saw_remaining = True
            if status not in exhausted_statuses:
                return False
        return saw_remaining

    def _reopenable_failed_spec_prerequisite_ids(self, tasks: list[Any]) -> set[str]:
        workflow = self.config.get("workflow", {})
        max_rounds = int(workflow.get("spec_task_recovery_rounds", 2) or 0)
        if max_rounds <= 0:
            return set()
        blocking_ids = self._failed_spec_prerequisite_ids(tasks)
        reopenable: set[str] = set()
        for task in tasks:
            if not isinstance(task, dict):
                continue
            task_id = str(task.get("task_id") or "")
            if task_id not in blocking_ids or str(task.get("status")) != "failed":
                continue
            rounds = int(task.get("recovery_rounds", 0) or 0)
            if rounds < max_rounds:
                reopenable.add(task_id)
        return reopenable

    def _reopenable_failed_portfolio_task_ids(self, tasks: list[Any]) -> set[str]:
        workflow = self.config.get("workflow", {})
        if not workflow.get("spec_reopen_failed_portfolio_tasks"):
            return set()
        max_rounds = int(
            workflow.get(
                "spec_portfolio_recovery_rounds",
                workflow.get("spec_task_recovery_rounds", 0),
            )
            or 0
        )
        if max_rounds <= 0:
            return set()
        reopenable: set[str] = set()
        for task in tasks:
            if not isinstance(task, dict) or str(task.get("status")) != "failed":
                continue
            rounds = int(task.get("recovery_rounds", 0) or 0)
            if rounds < max_rounds:
                reopenable.add(str(task.get("task_id") or ""))
        return {task_id for task_id in reopenable if task_id}

    @staticmethod
    def _spec_blocked_task_ids(tasks: list[Any]) -> list[str]:
        closed = {
            str(task.get("task_id"))
            for task in tasks
            if isinstance(task, dict) and str(task.get("status")) == "closed"
        }
        blocked: list[str] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            status = str(task.get("status", "open"))
            if status not in {
                "open",
                "deferred",
                "in_progress",
                "needs_design",
                "needs_contract_rewrite",
            }:
                continue
            deps = task.get("depends_on")
            missing = [
                str(dep)
                for dep in deps
                if str(dep) not in closed
            ] if isinstance(deps, list) else []
            if missing:
                blocked.append(str(task.get("task_id") or "<unknown>"))
        return blocked

    def _select_spec_task(
        self, tasks: list[Any], candidates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        dependent_counts = self._spec_dependent_counts(tasks)
        order = {
            str(task.get("task_id")): index
            for index, task in enumerate(tasks)
            if isinstance(task, dict)
        }
        return sorted(
            candidates,
            key=lambda task: (
                0 if task.get("deferred_revisited") else 1,
                -dependent_counts.get(str(task.get("task_id")), 0),
                order.get(str(task.get("task_id")), 10_000),
            ),
        )[0]

    def _spec_dependent_counts(self, tasks: list[Any]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in tasks:
            if not isinstance(task, dict):
                continue
            deps = task.get("depends_on")
            if not isinstance(deps, list):
                continue
            for dep in deps:
                key = str(dep)
                counts[key] = counts.get(key, 0) + 1
        return counts

    def _spec_task_to_active_todo(self, task: dict[str, Any]) -> dict[str, Any]:
        budget = task.get("budget") if isinstance(task.get("budget"), dict) else {}
        acceptance = task.get("acceptance") if isinstance(task.get("acceptance"), dict) else {}
        micro_goal = "Complete the scheduled spec task and satisfy its frozen command acceptance."
        if acceptance.get("kind") == "metric":
            micro_goal = (
                "Try this independent metric tactic and improve the configured "
                "benchmark metric without breaking correctness."
            )
        edit_scope = str(task.get("edit_scope") or "").strip()
        if edit_scope:
            micro_goal = (
                "Implement the scheduled design contract within this scope: "
                f"{edit_scope}"
            )
        return {
            "todo_id": str(task.get("task_id")),
            "spec_task_id": str(task.get("task_id")),
            "status": "active",
            "strategy_axis": str(task.get("strategy_axis") or "general_edit"),
            "family_key": str(task.get("family_key") or task.get("strategy_axis") or ""),
            "title": str(task.get("title") or task.get("task_id")),
            "context": str(task.get("expected_signal") or task.get("title") or ""),
            "micro_goal": micro_goal,
            "implementation_hint": str(task.get("decision_hint") or ""),
            "allowed_files": list(task.get("deliverables") or []),
            "expected_signal": str(task.get("expected_signal") or ""),
            "risk_level": str(task.get("risk_level") or ""),
            "tactic_stage": str(task.get("tactic_stage") or "local_edit"),
            "risk_evidence": task.get("risk_evidence")
            if isinstance(task.get("risk_evidence"), dict)
            else {},
            "probe_plan": str(task.get("probe_plan") or ""),
            "probe_diff_contract": task.get("probe_diff_contract")
            if isinstance(task.get("probe_diff_contract"), dict)
            else {},
            "invariant_evidence": self._normalize_string_list(
                task.get("invariant_evidence")
            ),
            "target_symbols": self._normalize_string_list(task.get("target_symbols")),
            "target_regions": self._normalize_string_list(task.get("target_regions")),
            "preserved_invariants": self._normalize_string_list(
                task.get("preserved_invariants")
            ),
            "edit_scope": edit_scope,
            "validator": task.get("validator")
            if isinstance(task.get("validator"), dict)
            else {},
            "correctness_rationale": str(task.get("correctness_rationale") or ""),
            "fallback_plan": str(task.get("fallback_plan") or ""),
            "rollback_or_shrink_plan": str(task.get("rollback_or_shrink_plan") or ""),
            "attempts": int(task.get("attempts", 0) or 0),
            "budget": budget,
            "source": "spec_scheduler",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "created_loop": self.state.loop_count,
        }

    async def _read_current_spec_task_context(self) -> None:
        task = self._current_spec_task()
        if not task:
            self.state.current = AgentStateName.SCHEDULE
            return
        paths = self._spec_task_read_paths(task)
        self.state.planned_files = self._filter_read_files(paths)
        self.state.file_context = []
        for rel_path in self.state.planned_files:
            abs_path = self.state.repo_root / rel_path
            try:
                content = await self.mcp.read_file(str(abs_path))
            except FileNotFoundError:
                continue
            content = self._context_for_file(rel_path, content)
            self.state.file_context.append(FileSnapshot(path=rel_path, content=content))
        await self._load_external_contexts()
        if self._spec_task_is_context_only(task):
            self._close_context_only_spec_task(task)
            return
        await self._ensure_spec_task_boundary_snapshot(task)
        self.state.current = AgentStateName.ACCEPT_SYNTH

    def _spec_task_is_context_only(self, task: dict[str, Any]) -> bool:
        deliverables = task.get("deliverables")
        if isinstance(deliverables, list) and any(str(item).strip() for item in deliverables):
            return False
        acceptance = task.get("acceptance")
        commands = acceptance.get("commands") if isinstance(acceptance, dict) else None
        if isinstance(commands, list) and any(str(command).strip() for command in commands):
            return False
        return True

    def _close_context_only_spec_task(self, task: dict[str, Any]) -> None:
        spec = self.state.scratch.get("run_spec")
        if not isinstance(spec, dict):
            self.state.current = AgentStateName.SCHEDULE
            return
        task["status"] = "closed"
        task["closed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        task["decision_hint"] = "context_only"
        task["last_observation"] = {
            "loop": self.state.loop_count,
            "failed": False,
            "budget_counted": False,
            "summary": "Context-only spec task closed after TASK_READ.",
        }
        self.state.notes.append(f"Closed context-only spec task: {task.get('task_id')}")
        self._persist_run_spec(spec)
        self._append_spec_progress_event(
            "closed",
            spec,
            task,
            extra={"reason": "context_only"},
        )
        self.state.current = AgentStateName.SCHEDULE

    def _spec_task_read_paths(self, task: dict[str, Any]) -> list[str]:
        paths: list[str] = []
        for key in ("read_hints", "deliverables"):
            value = task.get(key)
            if isinstance(value, list):
                paths.extend(str(item) for item in value if str(item).strip())
        spec = self.state.scratch.get("run_spec")
        tasks = spec.get("task_graph") if isinstance(spec, dict) else []
        deps = task.get("depends_on") if isinstance(task.get("depends_on"), list) else []
        for dep in deps:
            dep_task = next(
                (
                    item
                    for item in tasks
                    if isinstance(item, dict) and str(item.get("task_id")) == str(dep)
                ),
                None,
            )
            deliverables = dep_task.get("deliverables") if isinstance(dep_task, dict) else []
            if isinstance(deliverables, list):
                paths.extend(str(item) for item in deliverables if str(item).strip())
        return list(dict.fromkeys(paths))

    def _current_spec_task(self) -> dict[str, Any] | None:
        task = self.state.scratch.get("current_spec_task")
        if isinstance(task, dict):
            return task
        task_id = str(self.state.scratch.get("current_spec_task_id", "") or "")
        spec = self.state.scratch.get("run_spec")
        tasks = spec.get("task_graph") if isinstance(spec, dict) else []
        for item in tasks:
            if isinstance(item, dict) and str(item.get("task_id")) == task_id:
                self.state.scratch["current_spec_task"] = item
                return item
        return None

    def _current_spec_task_writable_files(self) -> list[str]:
        if not self._spec_mode_enabled():
            return []
        task = self._current_spec_task()
        if not task:
            return []
        deliverables = task.get("deliverables")
        if not isinstance(deliverables, list):
            return []
        global_writable = self.config.get("workflow", {}).get("writable_files")
        if not isinstance(global_writable, list) or not global_writable:
            return [str(item) for item in deliverables if str(item).strip()]
        allowed = {str(item) for item in global_writable}
        patterns = [str(item) for item in global_writable]
        return [
            str(item)
            for item in deliverables
            if str(item) in allowed
            or any(fnmatch.fnmatch(str(item), pattern) for pattern in patterns)
        ]

    def _spec_acceptance_dir(self) -> Path:
        return self._workflow_artifact_path("spec_acceptance_dir", ".lma_acceptance")

    def _spec_acceptance_rel_dir(self) -> str:
        try:
            return self._spec_acceptance_dir().resolve(strict=False).relative_to(
                self.state.repo_root.resolve(strict=False)
            ).as_posix()
        except ValueError:
            return str(self._spec_acceptance_dir())

    def _is_spec_acceptance_path(self, path: str) -> bool:
        key = self._repo_path_key(path)
        acceptance_key = self._repo_path_key(str(self._spec_acceptance_dir()))
        return key == acceptance_key or key.startswith(f"{acceptance_key}/")

    def _test_commands_for_current_scope(self) -> list[str]:
        task = self._current_spec_task() if self._spec_mode_enabled() else None
        acceptance = task.get("acceptance") if isinstance(task, dict) else None
        if isinstance(acceptance, dict) and acceptance.get("kind") in {"command", "metric", "synthesized"}:
            commands = acceptance.get("commands")
            if isinstance(commands, list) and commands:
                return [str(command) for command in commands if str(command).strip()]
        return [str(command) for command in self.config.get("workflow", {}).get("test_commands", [])]

    async def _ensure_current_spec_task_acceptance(self) -> None:
        task = self._current_spec_task()
        spec = self.state.scratch.get("run_spec")
        if not isinstance(task, dict) or not isinstance(spec, dict):
            self.state.current = AgentStateName.CODE
            return
        acceptance = task.setdefault("acceptance", {})
        if not isinstance(acceptance, dict):
            acceptance = {}
            task["acceptance"] = acceptance
        kind = str(acceptance.get("kind") or "command")
        if kind != "synthesized" or acceptance.get("frozen_sha256"):
            self.state.current = AgentStateName.CODE
            return
        ok = await self._synthesize_and_freeze_acceptance(task)
        self._persist_run_spec(spec)
        if ok:
            self.state.current = AgentStateName.CODE
            return
        if task.get("status") == "closed":
            self._append_spec_progress_event(
                "closed",
                spec,
                task,
                extra={"acceptance": "red_first_already_green"},
            )
            self.state.current = AgentStateName.SCHEDULE
            return
        self.state.current = AgentStateName.CODE

    async def _synthesize_and_freeze_acceptance(self, task: dict[str, Any]) -> bool:
        workflow = self.config.get("workflow", {})
        retries = int(workflow.get("spec_acceptance_synth_retries", 1) or 1)
        attempts = max(1, retries + 1)
        task_id = str(task.get("task_id") or "task")
        task_dir = self._spec_acceptance_dir() / task_id
        rel_task_dir = f"{self._spec_acceptance_rel_dir().rstrip('/')}/{task_id}"
        for attempt in range(1, attempts + 1):
            try:
                output = await self._model_chat(
                    "coder",
                    acceptance_synth_prompt(self.state, task, rel_task_dir),
                    call_site="acceptance_synth",
                )
                data = parse_json_object(output)
                files = self._normalize_acceptance_files(data, task_id)
                commands = self._acceptance_commands_for_task_dir(rel_task_dir)
            except Exception as exc:
                self.state.notes.append(
                    f"Acceptance synth failed for {task_id} attempt {attempt}: {type(exc).__name__}: {exc}"
                )
                continue
            if not files or not commands:
                self.state.notes.append(
                    f"Acceptance synth failed for {task_id} attempt {attempt}: empty files or commands"
                )
                continue
            task_dir.mkdir(parents=True, exist_ok=True)
            written_paths: list[str] = []
            for rel_name, content in files:
                path = task_dir / rel_name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
                written_paths.append(
                    path.resolve(strict=False)
                    .relative_to(self.state.repo_root.resolve(strict=False))
                    .as_posix()
                )
            preflight = await self._preflight_acceptance_files(written_paths)
            if preflight:
                self.state.test_results = preflight
                self.state.notes.append(
                    f"Acceptance preflight failed for {task_id} attempt {attempt}"
                )
                continue
            red_results = await self._run_acceptance_commands(commands)
            red_failed = any(result.exit_code != 0 for result in red_results)
            if not red_failed and self._acceptance_results_ran_zero_tests(red_results):
                self.state.test_results = red_results
                self.state.notes.append(
                    f"Acceptance synth failed for {task_id} attempt {attempt}: zero tests discovered"
                )
                continue
            if not red_failed:
                task["status"] = "closed"
                task["closed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                task["last_observation"] = {
                    "acceptance": "red_first_already_green",
                    "summary": "\n".join(result.stdout[-800:] + result.stderr[-800:] for result in red_results),
                }
                self.state.notes.append(
                    f"Acceptance red-first already green for {task_id}; closing task"
                )
                return False
            acceptance = task.setdefault("acceptance", {})
            acceptance.update(
                {
                    "kind": "synthesized",
                    "test_paths": written_paths,
                    "commands": commands,
                    "frozen_sha256": self._hash_acceptance_files(written_paths),
                    "synthesized_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "red_first": True,
                }
            )
            self.state.notes.append(f"Frozen synthesized acceptance for {task_id}")
            return True
        fallback_commands = self.config.get("workflow", {}).get("test_commands", [])
        task["acceptance"] = {
            "kind": "command",
            "commands": [str(command) for command in fallback_commands if str(command).strip()],
            "synthesized_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self.state.notes.append(
            f"Acceptance synth exhausted for {task_id}; downgraded to command acceptance"
        )
        return False

    @staticmethod
    def _acceptance_results_ran_zero_tests(results: list[TestResult]) -> bool:
        for result in results:
            output = f"{result.stdout}\n{result.stderr}"
            if re.search(r"\bRan\s+0\s+tests\b", output):
                return True
        return False

    def _normalize_acceptance_files(
        self, data: dict[str, Any], task_id: str
    ) -> list[tuple[str, str]]:
        raw_files = data.get("files")
        if not isinstance(raw_files, list):
            return []
        files: list[tuple[str, str]] = []
        for index, item in enumerate(raw_files, start=1):
            if not isinstance(item, dict):
                continue
            raw_path = str(item.get("path") or f"test_{task_id}_{index}.py").strip()
            path = Path(raw_path)
            if path.is_absolute() or ".." in path.parts:
                continue
            if not fnmatch.fnmatch(path.name, "test*.py"):
                continue
            content = str(item.get("content") or "")
            if not content.strip():
                continue
            files.append((path.as_posix(), content))
        return files

    def _acceptance_commands_for_task_dir(self, rel_task_dir: str) -> list[str]:
        workflow = self.config.get("workflow", {})
        template = str(
            workflow.get(
                "spec_acceptance_command_template",
                "{quoted_python} -m unittest discover -s {quoted_dir} -p 'test*.py'",
            )
        ).strip()
        if not template:
            template = "{quoted_python} -m unittest discover -s {quoted_dir} -p 'test*.py'"
        python = sys.executable or "python3"
        command = template.format(
            dir=rel_task_dir,
            quoted_dir=shlex.quote(rel_task_dir),
            python=python,
            quoted_python=shlex.quote(python),
        ).strip()
        return [command] if command else []

    def _spec_acceptance_policy_context(self) -> str:
        workflow = self.config.get("workflow", {})
        default_kind = str(workflow.get("spec_default_acceptance_kind", "") or "").strip()
        force_default = bool(workflow.get("spec_force_default_acceptance_kind"))
        if not default_kind and not force_default:
            return ""
        lines = ["Spec acceptance policy:"]
        if default_kind:
            lines.append(f"- Default acceptance kind: {default_kind}")
        if force_default:
            lines.append("- The controller will force every task to the default acceptance kind.")
        if self._spec_force_metric_acceptance_enabled():
            lines.append("- The controller will force metric acceptance for this metric-search run.")
        if self._spec_tactic_portfolio_enabled():
            lines.append("- Treat implementation tasks as an independent tactic portfolio, not a waterfall plan.")
            lines.append("- Use depends_on: [] unless a task consumes a concrete artifact from another task.")
        if default_kind == "metric":
            lines.append("- Use metric acceptance for performance tasks; do not synthesize unit tests for metric optimization tasks.")
            lines.append("- Leave commands empty when the configured workflow metric/test commands should be used.")
        elif default_kind == "command":
            lines.append("- Use command acceptance unless a task explicitly requires synthesized tests.")
        return "\n".join(lines)

    async def _preflight_acceptance_files(self, paths: list[str]) -> list[TestResult]:
        candidate = CodeCandidate(
            "acceptance",
            [
                CodeChange(path=path, reason="acceptance preflight")
                for path in paths
            ],
            "acceptance preflight",
        )
        return await self._run_candidate_preflight(candidate, set(paths))

    async def _run_acceptance_commands(self, commands: list[str]) -> list[TestResult]:
        results: list[TestResult] = []
        workflow = self.config.get("workflow", {})
        for command in commands:
            result = await self.mcp.run_command(
                command,
                cwd=str(self.state.repo_root),
                timeout_seconds=workflow.get("command_timeout_seconds", 120),
                output_limit=workflow.get("command_output_limit", 200_000),
            )
            results.append(TestResult(**result))
        return results

    def _hash_acceptance_files(self, paths: list[str]) -> str:
        digest = hashlib.sha256()
        for rel_path in sorted(paths):
            digest.update(rel_path.encode())
            digest.update(b"\0")
            path = self.state.repo_root / rel_path
            if path.exists():
                digest.update(path.read_bytes())
            else:
                digest.update(b"<missing>")
            digest.update(b"\0")
        return digest.hexdigest()

    def _frozen_acceptance_changed(self) -> bool:
        task = self._current_spec_task() if self._spec_mode_enabled() else None
        acceptance = task.get("acceptance") if isinstance(task, dict) else None
        if not isinstance(acceptance, dict):
            return False
        frozen = str(acceptance.get("frozen_sha256") or "")
        paths = acceptance.get("test_paths")
        if not frozen or not isinstance(paths, list):
            return False
        rel_paths = [str(path) for path in paths if str(path).strip()]
        current = self._hash_acceptance_files(rel_paths)
        if current == frozen:
            return False
        self.state.test_results = [
            TestResult(
                command="preflight:acceptance-frozen",
                exit_code=1,
                stderr="Frozen synthesized acceptance files changed before TEST",
            )
        ]
        self.state.notes.append("Frozen synthesized acceptance hash mismatch")
        return True

    async def _run_spec_regression_gate(self) -> list[TestResult]:
        task = self._current_spec_task()
        spec = self.state.scratch.get("run_spec")
        if not isinstance(task, dict) or not isinstance(spec, dict):
            return []
        workflow = self.config.get("workflow", {})
        scope = str(workflow.get("spec_regression_scope", "all") or "all")
        if scope == "none":
            return []
        current_id = str(task.get("task_id") or "")
        tasks = spec.get("task_graph")
        if not isinstance(tasks, list):
            return []
        closed_tasks = [
            item
            for item in tasks
            if isinstance(item, dict)
            and str(item.get("status")) == "closed"
            and str(item.get("task_id")) != current_id
        ]
        if scope == "dependents":
            deps = self._spec_transitive_dependencies(current_id, tasks)
            closed_tasks = [
                item for item in closed_tasks if str(item.get("task_id")) in deps
            ]
        results: list[TestResult] = []
        for closed_task in closed_tasks:
            commands = self._spec_task_acceptance_commands(closed_task)
            if not commands:
                continue
            frozen_result = self._frozen_acceptance_result_for_task(closed_task)
            if frozen_result is not None:
                results.append(frozen_result)
                continue
            task_results = await self._run_acceptance_commands(commands)
            for result in task_results:
                result.command = (
                    f"regression:{closed_task.get('task_id')} {result.command}"
                )
            results.extend(task_results)
        invariant_commands = workflow.get("spec_invariant_commands", [])
        if isinstance(invariant_commands, str):
            invariant_commands = [invariant_commands]
        invariant_commands = [
            str(command).strip()
            for command in invariant_commands
            if str(command).strip()
        ] if isinstance(invariant_commands, list) else []
        if invariant_commands:
            invariant_results = await self._run_acceptance_commands(invariant_commands)
            for result in invariant_results:
                result.command = f"invariant:{result.command}"
            results.extend(invariant_results)
        if any(result.exit_code != 0 for result in results):
            self.state.notes.append("Spec regression gate failed")
        elif results:
            self.state.notes.append("Spec regression gate passed")
        return results

    def _spec_task_acceptance_commands(self, task: dict[str, Any]) -> list[str]:
        acceptance = task.get("acceptance")
        if not isinstance(acceptance, dict):
            return []
        commands = acceptance.get("commands")
        if not isinstance(commands, list):
            return []
        return [str(command).strip() for command in commands if str(command).strip()]

    def _frozen_acceptance_result_for_task(self, task: dict[str, Any]) -> TestResult | None:
        acceptance = task.get("acceptance")
        if not isinstance(acceptance, dict):
            return None
        frozen = str(acceptance.get("frozen_sha256") or "")
        paths = acceptance.get("test_paths")
        if not frozen or not isinstance(paths, list):
            return None
        rel_paths = [str(path) for path in paths if str(path).strip()]
        if self._hash_acceptance_files(rel_paths) == frozen:
            return None
        return TestResult(
            command=f"regression:{task.get('task_id')} preflight:acceptance-frozen",
            exit_code=1,
            stderr="Frozen synthesized acceptance files changed before regression gate",
        )

    def _spec_transitive_dependencies(self, task_id: str, tasks: list[Any]) -> set[str]:
        by_id = {
            str(task.get("task_id")): task
            for task in tasks
            if isinstance(task, dict)
        }
        seen: set[str] = set()
        stack = [task_id]
        while stack:
            current = stack.pop()
            task = by_id.get(current)
            deps = task.get("depends_on") if isinstance(task, dict) else []
            if not isinstance(deps, list):
                continue
            for dep in deps:
                dep_id = str(dep)
                if dep_id in seen:
                    continue
                seen.add(dep_id)
                stack.append(dep_id)
        return seen

    async def _ensure_spec_task_boundary_snapshot(self, task: dict[str, Any]) -> None:
        task_id = str(task.get("task_id") or "")
        current = self.state.scratch.get("spec_task_boundary_snapshot")
        if isinstance(current, dict) and current.get("task_id") == task_id:
            return
        paths = sorted(self._writable_files())
        self.state.scratch["spec_task_boundary_snapshot"] = {
            "task_id": task_id,
            "files": await self._snapshot_files(paths),
        }
        self.state.notes.append(f"Captured spec task boundary snapshot: {task_id}")

    async def _restore_spec_task_boundary_snapshot(self) -> None:
        snapshot = self.state.scratch.get("spec_task_boundary_snapshot")
        if not isinstance(snapshot, dict):
            return
        files = snapshot.get("files")
        if not isinstance(files, dict):
            return
        await self._restore_snapshot(files)
        self.state.notes.append(
            f"Restored spec task boundary snapshot: {snapshot.get('task_id')}"
        )

    async def _handle_spec_task_test_result(self, failed: bool) -> None:
        task = self._current_spec_task()
        spec = self.state.scratch.get("run_spec")
        if not isinstance(task, dict) or not isinstance(spec, dict):
            self.state.current = AgentStateName.FAILED
            return
        budget = task.setdefault("budget", {})
        budget_counted = self._spec_current_attempt_counts_toward_budget(failed)
        attempts_used = int(budget.get("attempts_used", task.get("attempts", 0)) or 0)
        if budget_counted:
            attempts_used += 1
        attempts_max = int(
            budget.get(
                "attempts_max",
                self.config.get("workflow", {}).get("spec_task_attempt_budget", 3),
            )
            or 1
        )
        budget["attempts_used"] = attempts_used
        task["attempts"] = attempts_used
        metric_acceptance = self.state.scratch.get("metric_acceptance")
        metric_observation = metric_acceptance if isinstance(metric_acceptance, dict) else {}
        task["last_observation"] = {
            "loop": self.state.loop_count,
            "failed": failed,
            "budget_counted": budget_counted,
            "summary": self.state.latest_test_summary(),
        }
        if metric_observation:
            task["last_observation"].update(
                {
                    key: value
                    for key, value in metric_observation.items()
                    if value is not None
                }
            )
            metric_summary = str(metric_observation.get("summary") or "").strip()
            if metric_summary:
                task["last_observation"]["summary"] = metric_summary
        candidate_observation = self.state.scratch.get("last_candidate_observation")
        if isinstance(candidate_observation, dict):
            for key in (
                "failure_class",
                "stage_result",
                "recovery_hint",
                "failure_origin",
                "issue_scope",
                "repo_valid_after_restore",
                "repair_task_eligible",
                "memory_use",
            ):
                value = candidate_observation.get(key)
                if value not in (None, "", [], {}):
                    task["last_observation"][key] = value
            candidate_summary = str(candidate_observation.get("summary") or "").strip()
            if candidate_summary and not metric_observation:
                task["last_observation"]["summary"] = candidate_summary
        if failed and metric_observation.get("failure_class") == "no_improvement":
            task["decision_hint"] = (
                "metric_no_improvement: tests passed but the measured metric did not "
                "improve. Ensure any new branch or helper is called by the benchmark path."
            )
        if self._should_rewrite_spec_after_task_failure(task, failed):
            await self._restore_spec_task_boundary_snapshot()
            self.state.scratch.pop("spec_task_boundary_snapshot", None)
            task["status"] = "needs_design"
            task["decision_hint"] = (
                "repeated_correctness_failure_requires_design_rewrite: do not retry "
                "the same tactic family. Rewrite this task as a smaller executable "
                "design contract before CODE."
            )
            self.state.loop_count += 1
            self._persist_run_spec(spec)
            self._append_spec_progress_event(
                "needs_design",
                spec,
                task,
                extra={
                    "reason": "repeated_correctness_failure",
                    "failure_class": task["last_observation"].get("failure_class"),
                },
            )
            rewrite_focus = self._spec_design_rewrite_focus(
                task,
                ["repeated correctness_failure"],
                failure_summary=self.state.latest_test_summary(),
            )
            self.state.scratch["spec_rewrite_focus"] = rewrite_focus
            self.state.scratch["spec_rewrite_target_task_id"] = str(
                task.get("task_id") or ""
            )
            if self._spec_global_loop_cap_reached():
                spec["last_stop_reason"] = "max_code_test_loops"
                spec["pending_spec_rewrite_reason"] = {
                    "task_id": task.get("task_id"),
                    "reason": "repeated_correctness_failure",
                    "issues": ["repeated correctness_failure"],
                    "failure_summary": self._truncate_text(
                        self.state.latest_test_summary(),
                        800,
                    ),
                }
                spec["progress"] = self._run_spec_progress(spec)
                self._persist_run_spec(spec)
                self.state.notes.append(
                    "Spec rewrite deferred because max_code_test_loops was reached"
                )
                self._append_spec_progress_event(
                    "failed",
                    spec,
                    task,
                    extra={
                        "reason": "max_code_test_loops",
                        "pending_spec_rewrite": True,
                    },
                )
                self.state.current = AgentStateName.FAILED
                return
            self.state.current = AgentStateName.SPEC_SYNTH
            return
        drift_recovery = self._active_task_drift_recovery_decision(task, failed)
        if drift_recovery:
            await self._restore_spec_task_boundary_snapshot()
            self.state.scratch.pop("spec_task_boundary_snapshot", None)
            self.state.loop_count += 1
            task["last_observation"]["active_task_drift_streak"] = drift_recovery.get(
                "per_task_streak"
            )
            task["last_observation"]["active_task_drift_same_fingerprint_streak"] = (
                drift_recovery.get("same_fingerprint_streak")
            )
            if drift_recovery.get("action") == "rewrite" and not self._spec_global_loop_cap_reached():
                drift_telemetry = (
                    drift_recovery.get("drift_telemetry")
                    if isinstance(drift_recovery.get("drift_telemetry"), dict)
                    else {}
                )
                task["status"] = "needs_contract_rewrite"
                task["decision_hint"] = (
                    "repeated_active_task_drift_requires_contract_rewrite: CODE "
                    "kept violating or over-broadening this active task contract. "
                    "Rewrite the task as a smaller executable probe, or retire it "
                    "in favor of a different runnable task."
                )
                task["contract_rewrite"] = {
                    "status": "requested",
                    "reason": "repeated_active_task_drift",
                    "rewrite_attempt_key": drift_recovery.get("rewrite_attempt_key"),
                    "rewrite_attempts": drift_recovery.get("rewrite_attempts"),
                    "rewrite_attempts_max": drift_recovery.get("rewrite_attempts_max"),
                    "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                spec["active_task_id"] = None
                self._persist_run_spec(spec)
                self._append_spec_progress_event(
                    "drift_recovery",
                    spec,
                    task,
                    extra={
                        "reason": "repeated_active_task_drift",
                        "action": "rewrite",
                        "per_task_streak": drift_recovery.get("per_task_streak"),
                        "same_fingerprint_streak": drift_recovery.get(
                            "same_fingerprint_streak"
                        ),
                        "fingerprint": drift_recovery.get("fingerprint"),
                        **drift_telemetry,
                    },
                )
                self._append_failure_signature(
                    phase="active_task",
                    spec=spec,
                    task=task,
                    status="needs_contract_rewrite",
                    failure_class="active_task_drift",
                    issue_code=str(drift_recovery.get("fingerprint") or "active_task_drift"),
                    issue_scope="candidate_delta",
                    summary=str(drift_recovery.get("summary") or ""),
                    extra={
                        "action": "rewrite",
                        "per_task_streak": drift_recovery.get("per_task_streak"),
                        "same_fingerprint_streak": drift_recovery.get(
                            "same_fingerprint_streak"
                        ),
                        "drift_fingerprint": drift_recovery.get("fingerprint"),
                        **drift_telemetry,
                    },
                )
                rewrite_focus = self._spec_design_rewrite_focus(
                    task,
                    ["repeated active_task_drift"],
                    failure_summary=str(drift_recovery.get("summary") or ""),
                )
                self.state.scratch["spec_rewrite_focus"] = rewrite_focus
                self.state.scratch["spec_rewrite_target_task_id"] = str(
                    task.get("task_id") or ""
                )
                self.state.current = AgentStateName.SPEC_SYNTH
                return
            drift_telemetry = (
                drift_recovery.get("drift_telemetry")
                if isinstance(drift_recovery.get("drift_telemetry"), dict)
                else {}
            )
            task["status"] = "deferred_contract_drift"
            task["decision_hint"] = (
                "contract_drift_streak_deferred: active-task drift repeated after "
                "the contract rewrite budget was exhausted. Keep this task out of "
                "CODE and schedule a different runnable task."
            )
            task["contract_rewrite"] = {
                "status": "deferred",
                "reason": "repeated_active_task_drift",
                "rewrite_attempt_key": drift_recovery.get("rewrite_attempt_key"),
                "rewrite_attempts": drift_recovery.get("rewrite_attempts"),
                "rewrite_attempts_max": drift_recovery.get("rewrite_attempts_max"),
                "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            spec["active_task_id"] = None
            self._persist_run_spec(spec)
            self._append_spec_progress_event(
                "drift_recovery",
                spec,
                task,
                extra={
                    "reason": "repeated_active_task_drift",
                    "action": "defer",
                    "per_task_streak": drift_recovery.get("per_task_streak"),
                    "same_fingerprint_streak": drift_recovery.get(
                        "same_fingerprint_streak"
                    ),
                    "fingerprint": drift_recovery.get("fingerprint"),
                    **drift_telemetry,
                },
            )
            self._append_failure_signature(
                phase="active_task",
                spec=spec,
                task=task,
                status="deferred_contract_drift",
                failure_class="active_task_drift",
                issue_code=str(drift_recovery.get("fingerprint") or "active_task_drift"),
                issue_scope="candidate_delta",
                summary=str(drift_recovery.get("summary") or ""),
                extra={
                    "action": "defer",
                    "per_task_streak": drift_recovery.get("per_task_streak"),
                    "same_fingerprint_streak": drift_recovery.get(
                        "same_fingerprint_streak"
                    ),
                    "drift_fingerprint": drift_recovery.get("fingerprint"),
                    **drift_telemetry,
                },
            )
            self.state.current = AgentStateName.SCHEDULE
            return
        if not failed:
            task["status"] = "closed"
            task["closed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self.state.notes.append(f"Closed spec task: {task.get('task_id')}")
            self.state.loop_count += 1
            self.state.scratch.pop("spec_task_boundary_snapshot", None)
            self._persist_run_spec(spec)
            self._append_spec_progress_event("closed", spec, task)
            self.state.current = AgentStateName.SCHEDULE
            return
        if attempts_used >= attempts_max:
            await self._restore_spec_task_boundary_snapshot()
            self.state.scratch.pop("spec_task_boundary_snapshot", None)
            task["status"] = "deferred" if not task.get("deferred_revisited") else "failed"
            task["decision_hint"] = "budget_exhausted"
            self.state.notes.append(
                f"Spec task {task.get('task_id')} {task['status']} after {attempts_used} attempts"
            )
            self.state.loop_count += 1
            self._persist_run_spec(spec)
            self._append_spec_progress_event(str(task["status"]), spec, task)
            self.state.current = AgentStateName.SCHEDULE
            return
        self.state.loop_count += 1
        self._persist_run_spec(spec)
        self._append_spec_progress_event(
            "retry",
            spec,
            task,
            extra={"budget_counted": budget_counted},
        )
        if self._spec_global_loop_cap_reached():
            spec["last_stop_reason"] = "max_code_test_loops"
            spec["progress"] = self._run_spec_progress(spec)
            self._persist_run_spec(spec)
            self.state.notes.append(
                f"Spec mode reached max_code_test_loops={self.state.max_loops}"
            )
            self._append_spec_progress_event(
                "failed",
                spec,
                task,
                extra={"reason": "max_code_test_loops"},
            )
            self.state.current = AgentStateName.FAILED
            return
        self.state.current = self._retry_state_after_failure()

    def _should_rewrite_spec_after_task_failure(
        self, task: dict[str, Any], failed: bool
    ) -> bool:
        if not failed or not self._spec_design_contract_gate_enabled():
            return False
        observation = task.get("last_observation")
        failure_class = (
            str(observation.get("failure_class") or "")
            if isinstance(observation, dict)
            else ""
        )
        if failure_class != "correctness_failure":
            task.pop("correctness_failure_streak", None)
            return False
        streak = int(task.get("correctness_failure_streak", 0) or 0) + 1
        task["correctness_failure_streak"] = streak
        threshold = int(
            self.config.get("workflow", {}).get(
                "spec_redesign_after_correctness_failures",
                2,
            )
            or 0
        )
        return threshold > 0 and streak >= threshold

    def _active_task_drift_recovery_decision(
        self, task: dict[str, Any], failed: bool
    ) -> dict[str, Any] | None:
        if not failed or not self._spec_mode_enabled():
            return None
        workflow = self.config.get("workflow", {})
        per_task_limit = int(
            workflow.get("spec_active_task_drift_streak_limit", 0) or 0
        )
        same_fingerprint_limit = int(
            workflow.get("spec_active_task_drift_same_fingerprint_limit", 0) or 0
        )
        if per_task_limit <= 0 and same_fingerprint_limit <= 0:
            return None
        task_id = str(task.get("task_id") or "")
        if not task_id:
            return None
        latest = self._latest_active_task_drift_attempt(task_id)
        if latest is None:
            return None
        per_task_streak, same_fingerprint_streak, fingerprint = (
            self._active_task_drift_streaks(task_id, latest)
        )
        hit_per_task = per_task_limit > 0 and per_task_streak >= per_task_limit
        hit_fingerprint = (
            same_fingerprint_limit > 0
            and same_fingerprint_streak >= same_fingerprint_limit
        )
        if not hit_per_task and not hit_fingerprint:
            return None
        attempt_key = self._spec_rewrite_origin_task_id(task, task_id)
        contract = task.get("contract_rewrite")
        prior_rewrites = (
            int(contract.get("rewrite_attempts", 0) or 0)
            if isinstance(contract, dict)
            else 0
        )
        max_rewrites = int(
            workflow.get("spec_active_task_drift_rewrite_attempts", 1) or 0
        )
        next_rewrite = prior_rewrites + 1
        action = "rewrite" if max_rewrites > 0 and prior_rewrites < max_rewrites else "defer"
        summary = self._active_task_drift_rewrite_summary(task_id, latest)
        drift_telemetry = self._active_task_drift_record_extra(latest, task=task)
        return {
            "action": action,
            "per_task_streak": per_task_streak,
            "same_fingerprint_streak": same_fingerprint_streak,
            "fingerprint": fingerprint,
            "rewrite_attempt_key": attempt_key,
            "rewrite_attempts": min(next_rewrite, max(prior_rewrites, max_rewrites)),
            "rewrite_attempts_max": max_rewrites,
            "summary": summary,
            "drift_telemetry": drift_telemetry,
        }

    def _active_task_drift_record_extra(
        self,
        attempt: dict[str, Any],
        *,
        task: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source = task if isinstance(task, dict) else self._active_task_drift_source()
        declared_regions = self._normalize_string_list(source.get("target_regions"))
        declared_symbols = self._normalize_string_list(source.get("target_symbols"))
        attempted_regions = self._active_task_drift_attempted_regions(attempt)
        tactic_stage = str(
            attempt.get("tactic_stage") or source.get("tactic_stage") or "tactic-unknown"
        )
        fingerprint = self._normalize_failure_issue_code(
            self._active_task_drift_fingerprint(attempt)
        )
        region_hash = self._failure_signature_target_region_hash(
            declared_regions,
            target_symbols=declared_symbols,
        )
        extra: dict[str, Any] = {
            "drift_declared_regions": declared_regions,
            "drift_declared_symbols": declared_symbols,
            "drift_attempted_regions": attempted_regions,
            "drift_target_region_hash": region_hash,
            "drift_cooldown_key": ":".join(
                [region_hash, tactic_stage or "tactic-unknown", fingerprint]
            ),
        }
        pairs = self._active_task_drift_region_pairs(
            declared_regions,
            attempted_regions,
        )
        if pairs:
            extra["drift_region_pairs"] = pairs
        return {
            key: value
            for key, value in extra.items()
            if value not in (None, "", [], {})
        }

    def _active_task_drift_source(self) -> dict[str, Any]:
        task = self.state.scratch.get("current_spec_task")
        if isinstance(task, dict) and task:
            return task
        active_todo = self.state.scratch.get("active_todo")
        if not isinstance(active_todo, dict):
            active_todo = self._load_active_todo()
        return active_todo if isinstance(active_todo, dict) else {}

    def _active_task_drift_attempted_regions(self, attempt: dict[str, Any]) -> list[str]:
        regions: list[str] = []
        changes = attempt.get("changes")
        if isinstance(changes, list):
            for change in changes:
                if not isinstance(change, dict):
                    continue
                target_region = str(change.get("target_region") or "").strip()
                if target_region:
                    regions.append(target_region)
                    continue
                path = str(change.get("path") or "").strip()
                if path:
                    regions.append(path)
        summary = attempt.get("probe_diff_summary")
        if isinstance(summary, dict):
            changed_files = self._normalize_string_list(summary.get("changed_files"))
            for symbol in self._normalize_string_list(summary.get("touched_symbols")):
                if "::" in symbol:
                    regions.append(symbol)
            if len(changed_files) == 1:
                path = changed_files[0]
                for function in self._normalize_string_list(summary.get("changed_functions")):
                    if "::" in function:
                        regions.append(function)
                    else:
                        regions.append(f"{path}::{function}")
            else:
                regions.extend(self._normalize_string_list(summary.get("changed_functions")))
            regions.extend(changed_files)
        detail = " ".join(
            str(attempt.get(key) or "")
            for key in ("failure_detail", "no_change_reason", "summary")
        )
        regions.extend(
            match.strip()
            for match in re.findall(r"change target_region\s+([^ ]+)\s+is outside", detail)
            if match.strip()
        )
        regions.extend(
            match.strip()
            for match in re.findall(r"change path\s+([^ ]+)\s+is outside", detail)
            if match.strip()
        )
        return list(dict.fromkeys(region for region in regions if region))

    @staticmethod
    def _active_task_drift_region_pairs(
        declared_regions: list[str],
        attempted_regions: list[str],
    ) -> list[dict[str, str]]:
        if not attempted_regions:
            return []
        if not declared_regions:
            return [{"declared": "", "attempted": attempted} for attempted in attempted_regions]
        pairs: list[dict[str, str]] = []
        for declared in declared_regions:
            for attempted in attempted_regions:
                pairs.append({"declared": declared, "attempted": attempted})
        return pairs

    def _latest_active_task_drift_attempt(self, task_id: str) -> dict[str, Any] | None:
        for attempt in reversed(self._recent_todo_attempts(task_id)):
            if (
                int(attempt.get("loop", -1)) == self.state.loop_count
                and self._attempt_is_active_task_drift(attempt)
            ):
                return attempt
        observation = self.state.scratch.get("last_candidate_observation")
        if isinstance(observation, dict) and self._attempt_is_active_task_drift(observation):
            record = dict(observation)
            record.setdefault("todo_id", task_id)
            record.setdefault("loop", self.state.loop_count)
            return record
        return None

    def _active_task_drift_streaks(
        self, task_id: str, latest: dict[str, Any]
    ) -> tuple[int, int, str]:
        latest_fingerprint = self._active_task_drift_fingerprint(latest)
        per_task_streak = 0
        same_fingerprint_streak = 0
        same_fingerprint_contiguous = True
        attempts = [
            attempt
            for attempt in self._recent_todo_attempts(task_id)
            if str(attempt.get("todo_id") or task_id) == task_id
        ]
        if latest not in attempts:
            attempts.append(latest)
        attempts.sort(key=lambda attempt: int(attempt.get("loop", -1)))
        for attempt in reversed(attempts):
            if not self._attempt_is_active_task_drift(attempt):
                break
            per_task_streak += 1
            if (
                same_fingerprint_contiguous
                and self._active_task_drift_fingerprint(attempt) == latest_fingerprint
            ):
                same_fingerprint_streak += 1
            else:
                same_fingerprint_contiguous = False
        return per_task_streak, same_fingerprint_streak, latest_fingerprint

    @staticmethod
    def _attempt_is_active_task_drift(attempt: dict[str, Any]) -> bool:
        if attempt.get("budget_counted") is not False:
            return False
        return (
            str(attempt.get("failure_class") or "") == "active_task_drift"
            or str(attempt.get("status") or "")
            in {
                "rejected_active_task_file_drift",
                "rejected_active_task_region_drift",
                "rejected_active_task_shape_drift",
                "rejected_todo_axis_drift",
                "rejected_todo_family_drift",
                "rejected_todo_scope_drift",
            }
        )

    def _active_task_drift_fingerprint(self, attempt: dict[str, Any]) -> str:
        explicit = str(attempt.get("fingerprint") or "").strip()
        if explicit:
            return explicit
        detail = " ".join(
            str(attempt.get(key, ""))
            for key in (
                "status",
                "failure_class",
                "stage_result",
                "tactic_stage",
                "failure_detail",
                "no_change_reason",
                "summary",
            )
        )
        return self._normalize_fingerprint_text(detail)[:240]

    def _active_task_drift_rewrite_summary(
        self, task_id: str, latest: dict[str, Any]
    ) -> str:
        attempts = [
            attempt
            for attempt in self._recent_todo_attempts(task_id)
            if self._attempt_is_active_task_drift(attempt)
        ][-5:]
        lines = [
            "Repeated active-task drift blocked CODE execution. Rewrite only the "
            "targeted task contract as a smaller executable probe, or defer it "
            "and keep sibling tasks runnable.",
            "Latest drift: "
            + self._truncate_text(
                str(
                    latest.get("no_change_reason")
                    or latest.get("failure_detail")
                    or latest.get("summary")
                    or ""
                ),
                600,
            ),
        ]
        if attempts:
            lines.append("Recent drift attempts:")
            for attempt in attempts:
                lines.append(
                    "- loop={loop} status={status} stage={stage} detail={detail}".format(
                        loop=attempt.get("loop"),
                        status=attempt.get("status"),
                        stage=attempt.get("tactic_stage") or attempt.get("stage_result"),
                        detail=self._truncate_text(
                            str(
                                attempt.get("no_change_reason")
                                or attempt.get("failure_detail")
                                or attempt.get("summary")
                                or ""
                            ),
                            220,
                        ),
                    )
                )
        return "\n".join(lines)

    def _spec_current_attempt_counts_toward_budget(self, failed: bool) -> bool:
        if not failed:
            return True
        active = self.state.scratch.get("active_todo")
        if not isinstance(active, dict):
            return True
        for key in ("last_non_budget_attempt", "last_attempt"):
            attempt = active.get(key)
            if not isinstance(attempt, dict):
                continue
            if int(attempt.get("loop", -1)) != self.state.loop_count:
                continue
            if attempt.get("budget_counted") is False:
                return False
        return True

    def _persist_spec_report(self) -> None:
        if not self._spec_mode_enabled():
            return
        spec = self.state.scratch.get("run_spec")
        if not isinstance(spec, dict):
            spec = self._load_run_spec(self._run_spec_path())
        if not isinstance(spec, dict) or not spec:
            quality_report = self.state.scratch.get("spec_quality_report")
            budget_exhausted = bool(self.state.scratch.get("spec_synth_budget_exhausted"))
            if (
                self.state.current != AgentStateName.FAILED
                or (
                    not budget_exhausted
                    and (
                        not isinstance(quality_report, dict)
                        or not self._spec_quality_report_failed(quality_report)
                    )
                )
            ):
                return
            spec = {
                "version": 2,
                "spec_id": (
                    quality_report.get("spec_id")
                    if isinstance(quality_report, dict)
                    else ""
                )
                or "",
                "objective": "Run spec generation failed before persistence.",
                "task_graph": [],
                "last_stop_reason": (
                    "spec_budget_exhausted"
                    if budget_exhausted
                    else "spec_quality_gate_failed"
                ),
            }
            if isinstance(quality_report, dict):
                spec["spec_quality_report"] = quality_report
        progress = self._run_spec_progress(spec)
        self._persist_spec_terminal_state(spec, progress)
        report_path = self._workflow_artifact_path(
            "spec_report_path", ".local_micro_agent/spec_report.md"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        tasks = spec.get("task_graph") if isinstance(spec.get("task_graph"), list) else []
        lines = [
            "# Spec Mode Report",
            "",
            f"- status: `{self.state.current}`",
            f"- spec_id: `{spec.get('spec_id', '')}`",
            f"- objective: {spec.get('objective', '')}",
            f"- progress: {progress.get('closed', 0)}/{progress.get('total', 0)} closed, "
            f"{progress.get('deferred', 0)} deferred, {progress.get('failed', 0)} failed",
            f"- code_test_loop_count: {self.state.loop_count}",
            f"- fsm_step_count: {self.state.fsm_step_count}",
            f"- max_code_test_loops: {self.state.max_loops}",
            f"- stop_reason: `{spec.get('last_stop_reason', '')}`",
            f"- zero_code_attempt: `{str(self.state.loop_count == 0).lower()}`",
            "",
            "## Tasks",
            "",
            "| task_id | status | attempts | recovery | deliverables | acceptance |",
            "|---|---:|---:|---:|---|---|",
        ]
        for task in tasks:
            if not isinstance(task, dict):
                continue
            acceptance = task.get("acceptance") if isinstance(task.get("acceptance"), dict) else {}
            commands = acceptance.get("commands") if isinstance(acceptance, dict) else []
            test_paths = acceptance.get("test_paths") if isinstance(acceptance, dict) else []
            acceptance_bits = []
            if isinstance(acceptance, dict):
                acceptance_bits.append(str(acceptance.get("kind", "")))
                if acceptance.get("frozen_sha256"):
                    acceptance_bits.append("frozen")
            if isinstance(test_paths, list) and test_paths:
                acceptance_bits.append("tests=" + ",".join(str(path) for path in test_paths))
            if isinstance(commands, list) and commands:
                acceptance_bits.append("commands=" + str(len(commands)))
            budget = task.get("budget") if isinstance(task.get("budget"), dict) else {}
            attempts_used = budget.get("attempts_used", task.get("attempts", 0))
            attempts_total = int(task.get("attempts_total", 0) or 0)
            if attempts_total > 0:
                attempts = f"{attempts_used} current / {attempts_total} prior"
            else:
                attempts = str(attempts_used)
            recovery = task.get("recovery_rounds", 0)
            deliverables = task.get("deliverables") if isinstance(task.get("deliverables"), list) else []
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(task.get("task_id", "")),
                        str(task.get("status", "")),
                        str(attempts),
                        str(recovery),
                        ", ".join(str(item) for item in deliverables),
                        "; ".join(bit for bit in acceptance_bits if bit),
                    ]
                )
                + " |"
            )
        notes_tail = self.state.notes[-12:]
        if notes_tail:
            lines.extend(["", "## Notes Tail", ""])
            lines.extend(f"- {note}" for note in notes_tail)
        quality_report = (
            spec.get("spec_quality_report")
            if isinstance(spec.get("spec_quality_report"), dict)
            else self.state.scratch.get("spec_quality_report")
        )
        if isinstance(quality_report, dict) and (
            self._spec_quality_report_failed(quality_report)
            or spec.get("last_stop_reason") == "spec_quality_gate_failed"
        ):
            issues = [
                issue
                for issue in quality_report.get("issues", [])
                if isinstance(issue, dict)
            ]
            issue_codes = [
                str(code)
                for code in quality_report.get("issue_codes", [])
                if str(code)
            ]
            lines.extend(
                [
                    "",
                    "## Quality Gate",
                    "",
                    f"- status: `{quality_report.get('status', '')}`",
                    f"- attempt: `{quality_report.get('attempt', '')}`",
                    "- issue_codes: "
                    + (", ".join(f"`{code}`" for code in issue_codes) or "`none`"),
                ]
            )
            for issue in issues[:8]:
                lines.append(
                    "- "
                    + str(issue.get("code") or "issue")
                    + ": "
                    + str(issue.get("detail") or "")
                )
        survivor = self._terminal_survivor_summary()
        if survivor:
            lines.extend(
                [
                    "",
                    "## Survivor",
                    "",
                    f"- status: `{survivor.get('status', '')}`",
                    f"- loop: `{survivor.get('loop', '')}`",
                    f"- candidate_id: `{survivor.get('candidate_id', '')}`",
                    f"- spec_task_id: `{survivor.get('spec_task_id', '')}`",
                    f"- metric: `{survivor.get('metric', '')}`",
                    f"- patch_path: `{survivor.get('patch_path', '')}`",
                ]
            )
        trajectory_quality = self._terminal_trajectory_quality(tasks)
        if trajectory_quality:
            lines.extend(
                [
                    "",
                    "## Trajectory Quality",
                    "",
                    f"- label: `{trajectory_quality.get('label', '')}`",
                    "- spec_aligned_success_count: "
                    f"`{trajectory_quality.get('spec_aligned_success_count', 0)}`",
                    f"- scope_drift_count: `{trajectory_quality.get('scope_drift_count', 0)}`",
                    "- budget_free_contract_rejection_count: "
                    f"`{trajectory_quality.get('budget_free_contract_rejection_count', 0)}`",
                    "- improved_candidate_spec_task_id: "
                    f"`{trajectory_quality.get('improved_candidate_spec_task_id', '')}`",
                    "- improved_candidate_matches_probe_plan: "
                    + str(
                        bool(
                            trajectory_quality.get(
                                "improved_candidate_matches_probe_plan"
                            )
                        )
                    ).lower(),
                ]
            )
        failure_signatures = self._read_spec_jsonl(self._failure_signature_path())
        if failure_signatures:
            class_counts = self._count_jsonl_values(
                failure_signatures, "failure_class"
            )
            issue_counts = self._count_jsonl_values(failure_signatures, "issue_code")
            lines.extend(
                [
                    "",
                    "## Failure Signatures",
                    "",
                    "- failure_class_counts: "
                    + json.dumps(class_counts, ensure_ascii=False, sort_keys=True),
                    "- issue_code_counts: "
                    + json.dumps(issue_counts, ensure_ascii=False, sort_keys=True),
                ]
            )
            latest = failure_signatures[-1]
            lines.extend(
                [
                    f"- latest_failure_class: `{latest.get('failure_class', '')}`",
                    f"- latest_issue_code: `{latest.get('issue_code', '')}`",
                    f"- latest_cooldown_key: `{latest.get('cooldown_key', '')}`",
                ]
            )
        spec_model_profile = self._spec_synth_profile_summary()
        lines.extend(
            [
                "",
                "## SPEC Calls",
                "",
                f"- spec_synth_call_count: {self._spec_synth_call_count()}",
                f"- spec_synth_calls_used: {self._spec_synth_call_count()}",
                f"- spec_synth_call_budget: {self._spec_synth_call_budget()}",
                f"- spec_model_call_count: {spec_model_profile.get('model_call_count', 0)}",
                f"- spec_model_elapsed_ms: {spec_model_profile.get('elapsed_ms', 0)}",
                "- spec_synth_budget_exhausted: "
                + str(bool(self.state.scratch.get("spec_synth_budget_exhausted"))).lower(),
            ]
        )
        report_path.write_text("\n".join(lines).rstrip() + "\n")
        self.state.notes.append(f"Persisted spec report: {report_path}")

    def _persist_spec_terminal_state(
        self,
        spec: dict[str, Any],
        progress: dict[str, int],
    ) -> None:
        path = self._workflow_artifact_path(
            "spec_terminal_state_path",
            ".local_micro_agent/terminal_state.json",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        spec_events = self._read_spec_jsonl(
            self._workflow_artifact_path(
                "spec_progress_path", ".local_micro_agent/spec_progress.jsonl"
            )
        )
        spec_model_profile = self._spec_synth_profile_summary()
        candidate_events = self._read_spec_jsonl(
            self._workflow_artifact_path(
                "candidate_history_path", ".local_micro_agent/candidates.jsonl"
            )
        )
        tasks = spec.get("task_graph") if isinstance(spec.get("task_graph"), list) else []
        survivor = self._terminal_survivor_summary(candidate_events)
        trajectory_quality = self._terminal_trajectory_quality(tasks, candidate_events)
        drift_recovery_summary = self._terminal_drift_recovery_summary(
            tasks,
            candidate_events,
            spec_events,
        )
        portfolio_recovery_summary = self._terminal_portfolio_recovery_summary(
            tasks,
            spec_events,
        )
        failure_signatures = self._read_spec_jsonl(self._failure_signature_path())
        graph_candidates = self._read_spec_jsonl(self._spec_graph_candidates_path())
        search = spec.get("search") if isinstance(spec.get("search"), dict) else {}
        terminal = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "state": str(self.state.current),
            "stop_reason": spec.get("last_stop_reason", ""),
            "code_test_loop_count": self.state.loop_count,
            "fsm_step_count": self.state.fsm_step_count,
            "max_code_test_loops": self.state.max_loops,
            "zero_code_attempt": self.state.loop_count == 0 and not candidate_events,
            "spec_synth_call_count": self._spec_synth_call_count(),
            "spec_synth_calls_used": self._spec_synth_call_count(),
            "spec_synth_call_budget": self._spec_synth_call_budget(),
            "spec_synth_budget_exhausted": bool(
                self.state.scratch.get("spec_synth_budget_exhausted")
            ),
            "spec_synth_model_profile": spec_model_profile,
            "spec_id": spec.get("spec_id"),
            "selected_graph_id": self._spec_graph_id(spec),
            "graph_reseed_attempts": int(search.get("reseed_attempts", 0) or 0),
            "graph_reseed_attempts_max": int(
                search.get("reseed_attempts_max", 0) or 0
            ),
            "cooldown_keys": search.get("cooldown_keys", []),
            "active_task_id": spec.get("active_task_id"),
            "progress": progress,
            "pending_spec_rewrite_reason": spec.get("pending_spec_rewrite_reason"),
            "tasks": [
                self._terminal_spec_task_snapshot(task)
                for task in tasks
                if isinstance(task, dict)
            ],
            "spec_progress_counts": self._count_jsonl_values(spec_events, "event"),
            "candidate_status_counts": self._count_jsonl_values(
                candidate_events, "status"
            ),
            "candidate_failure_class_counts": self._count_jsonl_values(
                candidate_events, "failure_class"
            ),
            "failure_signature_counts": self._count_jsonl_values(
                failure_signatures, "failure_class"
            ),
            "failure_signature_issue_counts": self._count_jsonl_values(
                failure_signatures, "issue_code"
            ),
            "graph_candidate_counts": self._count_jsonl_values(
                graph_candidates, "status"
            ),
            "graph_candidate_event_counts": self._count_jsonl_values(
                graph_candidates, "event"
            ),
            "last_spec_progress_event": spec_events[-1] if spec_events else None,
            "last_candidate_event": candidate_events[-1] if candidate_events else None,
            "last_failure_signature": (
                failure_signatures[-1] if failure_signatures else None
            ),
            "last_graph_candidate_event": (
                graph_candidates[-1] if graph_candidates else None
            ),
        }
        terminal.update(drift_recovery_summary)
        terminal.update(portfolio_recovery_summary)
        if survivor:
            terminal["survivor"] = survivor
        if trajectory_quality:
            terminal["trajectory_quality"] = trajectory_quality
        quality_report = (
            spec.get("spec_quality_report")
            if isinstance(spec.get("spec_quality_report"), dict)
            else self.state.scratch.get("spec_quality_report")
        )
        if isinstance(quality_report, dict) and (
            self._spec_quality_report_failed(quality_report)
            or spec.get("last_stop_reason") == "spec_quality_gate_failed"
        ):
            terminal["spec_quality_report"] = quality_report
        path.write_text(json.dumps(terminal, ensure_ascii=False, indent=2) + "\n")

    def _terminal_trajectory_quality(
        self,
        tasks: list[Any],
        candidate_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        records = candidate_events
        if records is None:
            records = self._read_spec_jsonl(
                self._workflow_artifact_path(
                    "candidate_history_path", ".local_micro_agent/candidates.jsonl"
                )
            )
        records = [record for record in records if isinstance(record, dict)]
        drift_statuses = {
            "rejected_active_task_file_drift",
            "rejected_active_task_region_drift",
            "rejected_active_task_shape_drift",
            "rejected_todo_axis_drift",
            "rejected_todo_family_drift",
            "rejected_todo_scope_drift",
        }
        drift_records = [
            record
            for record in records
            if str(record.get("status")) in drift_statuses
            or str(record.get("failure_class")) == "active_task_drift"
        ]
        improved_records = [
            record
            for record in records
            if str(record.get("status")) in {"accepted", "improved"}
        ]
        spec_aligned_records = [
            record
            for record in improved_records
            if self._candidate_record_matches_spec_task(record, tasks)
        ]
        latest_improved = improved_records[-1] if improved_records else {}
        latest_matches = (
            self._candidate_record_matches_spec_task(latest_improved, tasks)
            if latest_improved
            else False
        )
        if spec_aligned_records and drift_records:
            label = "spec_aligned_success_with_drift"
        elif spec_aligned_records:
            label = "spec_aligned_success"
        elif improved_records:
            label = "lucky_pass_risk"
        elif drift_records:
            label = "chaotic_retry"
        else:
            label = "no_success"
        return {
            "label": label,
            "spec_aligned_success_count": len(spec_aligned_records),
            "improved_count": len(improved_records),
            "scope_drift_count": len(drift_records),
            "budget_free_contract_rejection_count": sum(
                1 for record in drift_records if record.get("budget_counted") is False
            ),
            "improved_candidate_id": latest_improved.get("candidate_id"),
            "improved_candidate_spec_task_id": latest_improved.get("spec_task_id"),
            "improved_candidate_matches_probe_plan": latest_matches,
        }

    def _terminal_drift_recovery_summary(
        self,
        tasks: list[Any],
        candidate_events: list[dict[str, Any]],
        spec_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        workflow = self.config.get("workflow", {})
        drift_records = [
            record
            for record in candidate_events
            if isinstance(record, dict)
            and (
                str(record.get("failure_class")) == "active_task_drift"
                or self._is_active_todo_drift_record(record)
            )
        ]
        max_streak = 0
        current_streak = 0
        previous_task_id = ""
        for record in candidate_events:
            if not isinstance(record, dict) or not (
                str(record.get("failure_class")) == "active_task_drift"
                or self._is_active_todo_drift_record(record)
            ):
                current_streak = 0
                previous_task_id = ""
                continue
            task_id = str(record.get("todo_id") or record.get("spec_task_id") or "")
            if task_id and task_id == previous_task_id:
                current_streak += 1
            else:
                current_streak = 1
                previous_task_id = task_id
            max_streak = max(max_streak, current_streak)
        attempted_region_counts = self._count_drift_regions(
            drift_records,
            "drift_attempted_regions",
        )
        pair_counts = self._count_drift_region_pairs(drift_records)
        cooldown_key_counts = self._count_drift_regions(
            drift_records,
            "drift_cooldown_key",
        )
        saturation_threshold = int(
            workflow.get("spec_drift_saturation_threshold", 3) or 0
        )
        saturated_keys = {
            key: count
            for key, count in cooldown_key_counts.items()
            if saturation_threshold > 0 and count >= saturation_threshold
        }
        recovery_events = [
            event
            for event in spec_events
            if isinstance(event, dict) and str(event.get("event")) == "drift_recovery"
        ]
        duplicate_rewrite_rejections = [
            event
            for event in spec_events
            if isinstance(event, dict)
            and str(event.get("event")) == "drift_recovery"
            and str(event.get("action")) == "rewrite_rejected_duplicate_drift"
        ]
        saved_budget = sum(
            int(event.get("spec_budget_saved_by_drift_backoff", 0) or 0)
            for event in spec_events
            if isinstance(event, dict)
        )
        return {
            "active_task_drift_count": len(drift_records),
            "max_active_task_drift_streak": max_streak,
            "drift_recovery_count": len(recovery_events),
            "drift_deferred_task_ids": [
                str(task.get("task_id") or "")
                for task in tasks
                if isinstance(task, dict)
                and str(task.get("status")) == "deferred_contract_drift"
                and str(task.get("task_id") or "")
            ],
            "active_task_drift_attempted_region_counts": attempted_region_counts,
            "active_task_drift_region_pair_counts": pair_counts,
            "drift_cooldown_key_counts": cooldown_key_counts,
            "drift_saturation_threshold": saturation_threshold,
            "same_region_drift_saturation_count": len(saturated_keys),
            "same_region_drift_saturated_keys": saturated_keys,
            "targeted_rewrite_rejected_duplicate_drift": len(
                duplicate_rewrite_rejections
            ),
            "spec_budget_saved_by_drift_backoff": saved_budget,
        }

    def _count_drift_regions(
        self,
        records: list[dict[str, Any]],
        key: str,
        *,
        limit: int = 12,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            raw = record.get(key)
            values = raw if isinstance(raw, list) else [raw]
            for value in values:
                text = str(value or "").strip()
                if not text:
                    continue
                counts[text] = counts.get(text, 0) + 1
        return dict(
            sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:limit]
        )

    def _count_drift_region_pairs(
        self,
        records: list[dict[str, Any]],
        *,
        limit: int = 12,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            raw_pairs = record.get("drift_region_pairs")
            if not isinstance(raw_pairs, list):
                continue
            for pair in raw_pairs:
                if not isinstance(pair, dict):
                    continue
                declared = str(pair.get("declared") or "").strip() or "<undeclared>"
                attempted = str(pair.get("attempted") or "").strip()
                if not attempted:
                    continue
                key = f"{declared} -> {attempted}"
                counts[key] = counts.get(key, 0) + 1
        return dict(
            sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:limit]
        )

    def _terminal_portfolio_recovery_summary(
        self,
        tasks: list[Any],
        spec_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        workflow = self.config.get("workflow", {})
        reopen_events = [
            event
            for event in spec_events
            if isinstance(event, dict) and str(event.get("event")) == "portfolio_reopened"
        ]
        exhausted_events = [
            event
            for event in spec_events
            if isinstance(event, dict) and str(event.get("event")) == "portfolio_exhausted"
        ]
        exhausted_task_ids = [
            str(task.get("task_id") or "")
            for task in tasks
            if isinstance(task, dict)
            and str(task.get("status")) == "deferred_portfolio_exhausted"
            and str(task.get("task_id") or "")
        ]
        return {
            "portfolio_reopened_count": len(reopen_events),
            "portfolio_exhausted_count": len(exhausted_events),
            "portfolio_exhausted_task_ids": exhausted_task_ids,
            "max_portfolio_recovery_rounds": int(
                workflow.get(
                    "spec_portfolio_recovery_rounds",
                    workflow.get("spec_task_recovery_rounds", 0),
                )
                or 0
            ),
        }

    def _candidate_record_matches_spec_task(
        self,
        record: dict[str, Any],
        tasks: list[Any],
    ) -> bool:
        spec_task_id = str(record.get("spec_task_id") or "")
        if not spec_task_id:
            return False
        task = next(
            (
                item
                for item in tasks
                if isinstance(item, dict) and str(item.get("task_id")) == spec_task_id
            ),
            None,
        )
        if not isinstance(task, dict):
            return False
        changes = record.get("changes")
        if not isinstance(changes, list) or not changes:
            return False
        contract = (
            task.get("probe_diff_contract")
            if isinstance(task.get("probe_diff_contract"), dict)
            else {}
        )
        allowed_files = set(self._string_list_from_any(contract.get("allowed_files")))
        allowed_files.update(self._string_list_from_any(task.get("deliverables")))
        target_regions = self._string_list_from_any(task.get("target_regions"))
        allowed_regions = set(self._string_list_from_any(contract.get("allowed_regions")))
        allowed_regions.update(
            self._string_list_from_any(contract.get("expected_changed_regions"))
        )
        allowed_regions.update(target_regions)
        if not allowed_files:
            allowed_files.update(
                region.split("::", 1)[0]
                for region in target_regions
                if region.split("::", 1)[0]
            )
        for change in changes:
            if not isinstance(change, dict):
                return False
            path = str(change.get("path") or "")
            if allowed_files and path not in allowed_files:
                return False
            region = str(change.get("target_region") or "")
            if region and allowed_regions and region not in allowed_regions:
                return False
        return True

    def _terminal_survivor_summary(
        self,
        candidate_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        records = candidate_events
        if records is None:
            records = self._read_spec_jsonl(
                self._workflow_artifact_path(
                    "candidate_history_path", ".local_micro_agent/candidates.jsonl"
                )
            )
        survivor_records = [
            record
            for record in records
            if isinstance(record, dict)
            and str(record.get("status")) in {"accepted", "improved"}
        ]
        if not survivor_records:
            return {}
        record = survivor_records[-1]
        patch_path = (
            record.get("last_correct_patch_path")
            or record.get("best_patch_path")
            or record.get("patch_path")
        )
        state_path = (
            record.get("last_correct_state_path")
            or record.get("best_state_path")
            or record.get("state_path")
        )
        return {
            key: value
            for key, value in {
                "status": record.get("status"),
                "loop": record.get("loop"),
                "candidate_id": record.get("candidate_id"),
                "metric": record.get("metric"),
                "spec_task_id": record.get("spec_task_id"),
                "todo_id": record.get("todo_id"),
                "patch_path": patch_path,
                "state_path": state_path,
            }.items()
            if value not in (None, "", [], {})
        }

    def _spec_synth_profile_summary(self) -> dict[str, Any]:
        records = self._read_spec_jsonl(
            self._workflow_artifact_path(
                "profile_events_path", ".local_micro_agent/profile_events.jsonl"
            )
        )
        call_sites = {
            "run_spec",
            "run_spec_fallback",
            "spec_idea",
            "spec_idea_rewrite",
            "spec_synth",
            "spec_synth_fallback",
        }
        calls = [
            record
            for record in records
            if record.get("event_type") == "model_call"
            and str(record.get("call_site") or "") in call_sites
        ]
        elapsed_ms = 0.0
        for record in calls:
            value = record.get("elapsed_ms")
            if isinstance(value, (int, float)):
                elapsed_ms += float(value)
        return {
            "model_call_count": len(calls),
            "elapsed_ms": round(elapsed_ms, 3),
            "call_site_counts": self._count_jsonl_values(calls, "call_site"),
        }

    @staticmethod
    def _read_spec_jsonl(
        path: Path,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        try:
            if limit is not None and limit > 0:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    lines = list(deque(handle, maxlen=limit))
            else:
                lines = path.read_text(errors="replace").splitlines()
        except FileNotFoundError:
            return records
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    @staticmethod
    def _count_jsonl_values(records: list[dict[str, Any]], key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            value = str(record.get(key) or "")
            if not value:
                continue
            counts[value] = counts.get(value, 0) + 1
        return counts

    def _terminal_spec_task_snapshot(self, task: dict[str, Any]) -> dict[str, Any]:
        budget = task.get("budget") if isinstance(task.get("budget"), dict) else {}
        snapshot = {
            "task_id": task.get("task_id"),
            "status": task.get("status"),
            "title": task.get("title"),
            "attempts": task.get("attempts"),
            "attempts_used": budget.get("attempts_used"),
            "attempts_max": budget.get("attempts_max"),
            "recovery_rounds": task.get("recovery_rounds"),
            "risk_level": task.get("risk_level"),
            "tactic_stage": task.get("tactic_stage"),
            "target_symbols": task.get("target_symbols"),
            "target_regions": task.get("target_regions"),
            "probe_diff_contract": task.get("probe_diff_contract"),
            "last_observation": task.get("last_observation"),
            "design_contract": task.get("design_contract"),
            "decision_hint": task.get("decision_hint"),
            "portfolio_exhausted_at": task.get("portfolio_exhausted_at"),
        }
        return {
            key: value
            for key, value in snapshot.items()
            if value not in (None, "", [], {})
        }

    def _todo_contract_soft_now(self) -> bool:
        if self._spec_hard_active_todo_contract_now():
            return False
        return (
            self._todo_soft_until_first_improvement_enabled()
            and not self._has_current_run_improvement()
        )

    def _validated_pattern_followup_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("validated_pattern_followup"))

    def _create_validated_pattern_followup_todo(self) -> None:
        if not self._validated_pattern_followup_enabled():
            return
        if self._has_active_todo_budget():
            return
        record = self._latest_candidate_record_with_status("improved")
        if not record:
            return
        axis = self._normalize_strategy_axis(str(record.get("strategy_axis", "")))
        if self._strict_strategy_axis_pool_enabled() and axis not in self._strategy_axis_pool():
            axes = record.get("strategy_axes")
            if isinstance(axes, list):
                axis = next(
                    (
                        normalized
                        for raw_axis in axes
                        if (normalized := self._normalize_strategy_axis(str(raw_axis)))
                        in self._strategy_axis_pool()
                    ),
                    "general_edit",
                )
            else:
                axis = "general_edit"
        if not axis:
            axes = record.get("strategy_axes")
            if isinstance(axes, list):
                axis = next(
                    (
                        normalized
                        for raw_axis in axes
                        if (normalized := self._normalize_strategy_axis(str(raw_axis)))
                    ),
                    "general_edit",
                )
            else:
                axis = "general_edit"
        family_aliases = [
            self._normalize_strategy_axis(str(alias))
            for alias in record.get("family_aliases", []) or []
            if self._normalize_strategy_axis(str(alias))
        ]
        family_key = family_aliases[0] if family_aliases else axis
        todo_id = f"todo-{self.state.loop_count:03d}-{axis}-followup"
        metric = record.get("metric")
        changes = record.get("changes", [])
        todo = {
            "todo_id": todo_id,
            "parent_tactic_id": f"validated-candidate-{record.get('loop')}-{record.get('candidate_id')}",
            "status": "active",
            "strategy_axis": axis,
            "family_key": family_key,
            "title": f"Follow up validated {axis} pattern",
            "context": (
                "A current-run candidate improved the metric. Explore one narrow "
                "follow-up on the same axis/family and nearby edited code. Do not "
                "import outside solution knowledge; extend only the validated local "
                "pattern from this run.\n\n"
                f"validated_metric: {metric}\n"
                f"validated_reason: {record.get('reason', '')}\n"
                f"validated_changes: {json.dumps(changes, ensure_ascii=False)}"
            ),
            "micro_goal": (
                "Find the smallest nearby extension of the validated pattern that is "
                "likely to improve the same measured metric."
            ),
            "implementation_hint": (
                "Inspect the edited region and look for the same local redundancy, "
                "missed symmetric case, or nearby repeated pattern. Keep the edit narrow."
            ),
            "allowed_files": sorted(self._writable_files()),
            "forbidden_patterns": [
                "unrelated rewrite",
                "changing tests or fixtures",
                "mixing a new tactic family before the follow-up is tried",
            ],
            "expected_signal": (
                "Tests pass and the metric improves, or the attempt yields a concrete "
                "failure reason that can guide one repair."
            ),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "created_loop": self.state.loop_count,
            "source": "validated_pattern_followup",
            "parent_candidate": {
                "loop": record.get("loop"),
                "candidate_id": record.get("candidate_id"),
                "metric": metric,
                "artifact_id": record.get("artifact_id"),
            },
        }
        self.state.scratch["active_todo"] = todo
        self._persist_todo_plan(todo)
        self.state.notes.append(f"Created validated-pattern follow-up todo: {todo_id}")

    def _latest_candidate_record_with_status(self, status: str) -> dict[str, Any] | None:
        path = self._candidate_history_path()
        if path is None or not path.exists():
            return None
        for line in reversed(path.read_text(errors="replace").splitlines()):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == status:
                return record
        return None

    def _brainstorm_all_tactics_failed_for_current_loop(self) -> bool:
        return self.state.scratch.get("brainstorm_all_tactics_failed_loop") == self.state.loop_count

    def _should_brainstorm(self) -> bool:
        workflow = self.config.get("workflow", {})
        threshold = int(workflow.get("brainstorm_after_rejections", 0) or 0)
        if threshold <= 0:
            return False
        if self.state.scratch.get("last_brainstorm_loop") == self.state.loop_count:
            return False
        if self._has_active_todo_brainstorm_budget():
            return False
        records = self._candidate_history_records(limit=max(threshold, 1))
        if len(records) < threshold:
            return False
        return all(str(record.get("status", "")).startswith("rejected") for record in records)

    def _has_active_todo_budget(self) -> bool:
        if self._todo_contract_soft_now():
            return False
        return self._active_todo_with_attempt_budget() is not None

    def _has_active_todo_brainstorm_budget(self) -> bool:
        if self._todo_contract_soft_now():
            workflow = self.config.get("workflow", {})
            if workflow.get("pre_improvement_todo_blocks_brainstorm", True) is False:
                return False
        return self._active_todo_with_attempt_budget() is not None

    def _active_todo_with_attempt_budget(self) -> dict[str, Any] | None:
        active_todo = self.state.scratch.get("active_todo")
        if not isinstance(active_todo, dict):
            active_todo = self._load_active_todo()
            if active_todo:
                self.state.scratch["active_todo"] = active_todo
        if not isinstance(active_todo, dict) or not active_todo:
            return None
        if active_todo.get("status") not in {"active", "attempted"}:
            return None
        if self._todo_attempt_budget_exhausted(active_todo):
            return None
        return active_todo

    def _format_tactic_library(self) -> str:
        tactic_library = self.state.scratch.get("tactic_library")
        if not isinstance(tactic_library, str) or not tactic_library.strip():
            return ""
        selected_tactic = self.state.scratch.get("selected_tactic")
        if isinstance(selected_tactic, dict) and selected_tactic:
            return (
                "Selected tactic for this CODE attempt:\n"
                + json.dumps(selected_tactic, ensure_ascii=False, indent=2)
                + "\n\nFull tactic library:\n"
                + tactic_library.strip()
            )
        return tactic_library.strip()

    def _format_active_todo(self) -> str:
        if self._todo_contract_soft_now():
            return ""
        active_todo = self.state.scratch.get("active_todo")
        if not isinstance(active_todo, dict):
            active_todo = self._load_active_todo()
            if active_todo:
                self.state.scratch["active_todo"] = active_todo
        if not isinstance(active_todo, dict) or not active_todo:
            return ""
        if active_todo.get("status") not in {"active", "attempted"}:
            return ""
        if self._todo_attempt_budget_exhausted(active_todo):
            return ""
        return json.dumps(active_todo, ensure_ascii=False, indent=2)

    def _format_todo_observation_chain(self) -> str:
        if not self.config.get("workflow", {}).get(
            "observation_backed_todo_continuation", True
        ):
            return ""
        active_todo = self.state.scratch.get("active_todo")
        if not isinstance(active_todo, dict):
            active_todo = self._load_active_todo()
            if active_todo:
                self.state.scratch["active_todo"] = active_todo
        if not isinstance(active_todo, dict) or not active_todo:
            return ""
        if active_todo.get("status") not in {"active", "attempted", "validated"}:
            return ""
        todo_id = str(active_todo.get("todo_id", ""))
        if not todo_id:
            return ""
        attempts = self._recent_todo_attempts(todo_id)
        attempt_limit = int(
            self.config.get("workflow", {}).get("todo_observation_chain_attempt_limit", 5)
            or 5
        )
        compact_attempts: list[dict[str, Any]] = []
        for attempt in attempts[-attempt_limit:]:
            if not isinstance(attempt, dict):
                continue
            compact: dict[str, Any] = {
                "loop": attempt.get("loop"),
                "status": attempt.get("status"),
                "metric": attempt.get("metric"),
                "strategy_axis": attempt.get("strategy_axis"),
                "tactic_stage": attempt.get("tactic_stage"),
                "stage_result": attempt.get("stage_result"),
                "failure_class": attempt.get("failure_class"),
                "summary": self._truncate_text(str(attempt.get("summary", "")), 320),
                "recovery_hint": self._truncate_text(
                    str(attempt.get("recovery_hint", "")), 280
                ),
                "diagnostic_summary": self._truncate_text(
                    str(attempt.get("diagnostic_summary", "")), 360
                ),
            }
            next_actions = attempt.get("next_actions")
            if isinstance(next_actions, list) and next_actions:
                compact["next_actions"] = [
                    self._truncate_text(str(action), 180)
                    for action in next_actions[:3]
                    if action
                ]
            compact_attempts.append(
                {
                    key: value
                    for key, value in compact.items()
                    if value not in (None, "", [], {})
                }
            )
        latest = compact_attempts[-1] if compact_attempts else {}
        continuation_focus = self._todo_continuation_focus(latest)
        chain = {
            "todo": {
                "todo_id": active_todo.get("todo_id"),
                "status": active_todo.get("status"),
                "strategy_axis": active_todo.get("strategy_axis"),
                "family_key": active_todo.get("family_key"),
                "tactic_stage": active_todo.get("tactic_stage", "local_edit"),
                "attempts": active_todo.get("attempts", 0),
                "non_budget_attempts": active_todo.get("non_budget_attempts", 0),
                "title": active_todo.get("title"),
                "context": self._truncate_text(str(active_todo.get("context", "")), 700),
                "micro_goal": active_todo.get("micro_goal"),
                "risk_level": active_todo.get("risk_level", ""),
                "risk_evidence": active_todo.get("risk_evidence", {}),
                "probe_plan": active_todo.get("probe_plan", ""),
                "probe_diff_contract": active_todo.get("probe_diff_contract", {}),
                "invariant_evidence": active_todo.get("invariant_evidence", []),
                "target_symbols": active_todo.get("target_symbols", []),
                "target_regions": active_todo.get("target_regions", []),
                "preserved_invariants": active_todo.get("preserved_invariants", []),
                "edit_scope": active_todo.get("edit_scope", ""),
                "validator": active_todo.get("validator", {}),
                "correctness_rationale": active_todo.get("correctness_rationale", ""),
                "fallback_plan": active_todo.get("fallback_plan", ""),
                "rollback_or_shrink_plan": active_todo.get("rollback_or_shrink_plan", ""),
            },
            "recent_attempts": compact_attempts,
            "continuation_focus": continuation_focus,
            "continuation_rule": (
                "Continue from the latest observation. A no-signal result means "
                "the edit did not change the measured observable enough; an invariant "
                "failure means preserve that invariant before trying to optimize. "
                "Only switch tactic when the observation chain explains why the "
                "current hypothesis is exhausted."
            ),
        }
        return json.dumps(chain, ensure_ascii=False, indent=2)

    @staticmethod
    def _todo_continuation_focus(latest_attempt: dict[str, Any]) -> str:
        failure_class = str(latest_attempt.get("failure_class", ""))
        if failure_class in {"invariant_broken", "scope_too_broad", "guard_missing"}:
            return (
                "Repair the named invariant at smaller scope before broadening the tactic."
            )
        if failure_class in {"no_improvement", "probe_no_signal"}:
            return (
                "Use diagnostics or generated artifacts to identify what did not change; "
                "then move the edit to the smallest code region that can change that observable."
            )
        if failure_class == "patch_miss":
            return "Refresh exact source context and repair the patch target before judging the tactic."
        if failure_class == "duplicate_variant":
            return "Change the implementation shape or edit site; do not rename the same variant."
        if failure_class:
            return "Use the latest failure class as the next hypothesis, not a reset signal."
        return "Design the next smallest measurable probe for this todo."

    def _format_structural_state_context(self) -> str:
        if not self._structural_state_enabled():
            return ""
        state = self._load_structural_state()
        checkpoints = state.get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            return ""
        limit = int(
            self.config.get("workflow", {}).get("structural_state_context_limit", 3)
            or 3
        )
        patch_limit = int(
            self.config.get("workflow", {}).get("structural_state_patch_context_limit", 1800)
            or 1800
        )
        items: list[dict[str, Any]] = []
        for checkpoint in checkpoints[-limit:]:
            if not isinstance(checkpoint, dict):
                continue
            item = {
                key: checkpoint.get(key)
                for key in (
                    "checkpoint_id",
                    "loop",
                    "candidate_id",
                    "strategy_axis",
                    "family_aliases",
                    "tactic_stage",
                    "stage_result",
                    "metric",
                    "todo_id",
                    "reason",
                    "changes",
                )
                if checkpoint.get(key) not in (None, "", [], {})
            }
            patch_path = checkpoint.get("patch_path")
            if patch_path:
                item["patch_path"] = patch_path
                path = self.state.repo_root / str(patch_path)
                if path.exists():
                    item["patch_excerpt"] = self._truncate_text(
                        path.read_text(errors="replace"), patch_limit
                    )
            items.append(item)
        return json.dumps(items, ensure_ascii=False, indent=2)

    def _load_active_todo(self) -> dict[str, Any] | None:
        path = self._workflow_artifact_path(
            "active_todo_path", ".local_micro_agent/active_todo.json"
        )
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(errors="replace"))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def _structural_state_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(
            workflow.get("structural_tactic_lifecycle", True)
            and workflow.get("structural_state_checkpoint", True)
        )

    def _structural_state_path(self) -> Path:
        return self._workflow_artifact_path(
            "structural_state_path", ".local_micro_agent/structural_state.json"
        )

    def _structural_checkpoint_dir(self) -> Path:
        return self._workflow_artifact_path(
            "structural_checkpoint_dir", ".local_micro_agent/structural_checkpoints"
        )

    def _load_structural_state(self) -> dict[str, Any]:
        path = self._structural_state_path()
        if not path.exists():
            return {"version": 1, "checkpoints": []}
        try:
            data = json.loads(path.read_text(errors="replace"))
        except json.JSONDecodeError:
            return {"version": 1, "checkpoints": []}
        if not isinstance(data, dict):
            return {"version": 1, "checkpoints": []}
        checkpoints = data.get("checkpoints")
        if not isinstance(checkpoints, list):
            data["checkpoints"] = []
        data.setdefault("version", 1)
        return data

    def _record_structural_checkpoint(
        self,
        candidate: CodeCandidate,
        status: str,
        metric: int | None,
        applied: int,
        failed: bool,
        patch_text: str,
        extra: dict[str, Any],
    ) -> None:
        if not self._structural_state_enabled():
            return
        if failed or applied <= 0 or metric is None or not patch_text.strip():
            return
        stage = str(extra.get("tactic_stage", ""))
        if not stage.startswith("structural_"):
            return
        stage_result = str(extra.get("stage_result", ""))
        if stage_result not in {
            "scaffold_validated",
            "probe_validated",
            "probe_validated_no_metric_gain",
        }:
            return

        checkpoint_id = f"loop-{self.state.loop_count:03d}-{candidate.candidate_id}"
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", checkpoint_id).strip("-")
        checkpoint_dir = self._structural_checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        patch_path = checkpoint_dir / f"{safe_id}.patch"
        patch_path.write_text(patch_text)

        todo_id = self._active_todo_id_for_record(
            {
                "strategy_axis": candidate.strategy_axis,
                "tactic_stage": stage,
            }
        )
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "checkpoint_id": checkpoint_id,
            "loop": self.state.loop_count,
            "candidate_id": candidate.candidate_id,
            "status": status,
            "metric": metric,
            "strategy_axis": candidate.strategy_axis,
            "strategy_axes": self._candidate_strategy_axes(candidate),
            "family_aliases": sorted(self._candidate_reason_family_aliases(candidate)),
            "tactic_stage": stage,
            "stage_result": stage_result,
            "failure_class": extra.get("failure_class"),
            "reason": self._truncate_text(candidate.reason, 700),
            "changes": self._summarize_changes(candidate.changes),
            "todo_id": todo_id,
            "artifact_id": extra.get("artifact_id"),
            "artifact_path": extra.get("artifact_path"),
            "patch_path": self._repo_relative_path(patch_path),
        }
        state = self._load_structural_state()
        checkpoints = state.setdefault("checkpoints", [])
        if not isinstance(checkpoints, list):
            checkpoints = []
            state["checkpoints"] = checkpoints
        checkpoints.append(
            {key: value for key, value in record.items() if value not in (None, "", [], {})}
        )
        limit = int(
            self.config.get("workflow", {}).get("structural_state_checkpoint_limit", 8)
            or 8
        )
        if limit > 0 and len(checkpoints) > limit:
            del checkpoints[:-limit]
        state["latest_checkpoint_id"] = checkpoint_id
        state["updated_at"] = record["ts"]
        path = self._structural_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        self.state.notes.append(f"Recorded structural checkpoint: {checkpoint_id}")

    def _todo_attempt_budget_exhausted(self, todo: dict[str, Any]) -> bool:
        budget = int(self.config.get("workflow", {}).get("todo_attempt_budget", 1) or 1)
        attempts = int(todo.get("attempts", 0) or 0)
        return attempts >= budget

    def _format_todo_ledger_summary(self) -> str:
        plan_path = self._workflow_artifact_path(
            "todo_plan_path", ".local_micro_agent/todo_plan.json"
        )
        plan = self._load_todo_plan(plan_path)
        todos = plan.get("todos")
        if not isinstance(todos, list) or not todos:
            return ""
        limit = int(self.config.get("workflow", {}).get("todo_ledger_summary_limit", 8) or 8)
        summary = []
        for todo in todos[-limit:]:
            if not isinstance(todo, dict):
                continue
            last_attempt = todo.get("last_attempt")
            last_failure_detail = (
                last_attempt.get("failure_detail") or last_attempt.get("no_change_reason")
                if isinstance(last_attempt, dict)
                else ""
            )
            last_summary = (
                last_attempt.get("summary") if isinstance(last_attempt, dict) else ""
            )
            last_recovery_hint = (
                last_attempt.get("recovery_hint") if isinstance(last_attempt, dict) else ""
            )
            last_diagnostic_summary = (
                last_attempt.get("diagnostic_summary") if isinstance(last_attempt, dict) else ""
            )
            summary.append(
                {
                    "todo_id": todo.get("todo_id"),
                    "status": todo.get("status"),
                    "strategy_axis": todo.get("strategy_axis"),
                    "tactic_stage": todo.get("tactic_stage", "local_edit"),
                    "attempts": todo.get("attempts", 0),
                    "context": self._truncate_text(str(todo.get("context", "")), 280),
                    "last_status": (
                        last_attempt.get("status") if isinstance(last_attempt, dict) else None
                    ),
                    "last_metric": (
                        last_attempt.get("metric") if isinstance(last_attempt, dict) else None
                    ),
                    "last_failure_class": (
                        last_attempt.get("failure_class")
                        if isinstance(last_attempt, dict)
                        else None
                    ),
                    "last_stage_result": (
                        last_attempt.get("stage_result")
                        if isinstance(last_attempt, dict)
                        else None
                    ),
                    "last_reason": self._truncate_text(
                        str(last_attempt.get("reason", "")), 220
                    )
                    if isinstance(last_attempt, dict)
                    else "",
                    "last_summary": self._truncate_text(str(last_summary), 260)
                    if last_summary
                    else "",
                    "last_failure_detail": self._truncate_text(
                        str(last_failure_detail), 260
                    )
                    if last_failure_detail
                    else "",
                    "last_recovery_hint": self._truncate_text(
                        str(last_recovery_hint), 260
                    )
                    if last_recovery_hint
                    else "",
                    "last_diagnostic_summary": self._truncate_text(
                        str(last_diagnostic_summary), 360
                    )
                    if last_diagnostic_summary
                    else "",
                }
            )
        return json.dumps(summary, ensure_ascii=False, indent=2)

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
            item = {
                "status": status,
                "metric": metric,
                "strategy_axis": axis,
                "strategy_axes": axes,
                "reason": reason,
            }
            failure_detail = record.get("failure_detail") or record.get("no_change_reason")
            for key in ("failure_class", "summary", "recovery_hint"):
                value = record.get(key)
                if value:
                    item[key] = self._truncate_text(str(value), 260)
            next_actions = record.get("next_actions")
            if isinstance(next_actions, list) and next_actions:
                item["next_actions"] = [
                    self._truncate_text(str(action), 180)
                    for action in next_actions[:3]
                    if action
                ]
            if failure_detail:
                item["failure_detail"] = self._truncate_text(str(failure_detail), 260)
            summary.append(item)
        return json.dumps(summary, ensure_ascii=False, indent=2)

    def _active_todo_stage(self) -> str:
        active_todo = self.state.scratch.get("active_todo")
        if not isinstance(active_todo, dict):
            active_todo = self._load_active_todo()
            if active_todo:
                self.state.scratch["active_todo"] = active_todo
        if not isinstance(active_todo, dict):
            return "local_edit"
        stage = str(active_todo.get("tactic_stage", "local_edit") or "local_edit")
        if stage in {
            "local_edit",
            "structural_scaffold",
            "structural_probe",
            "structural_expand",
        }:
            return stage
        return "local_edit"

    def _is_structural_learning_record(self, record: dict[str, Any]) -> bool:
        if not self.config.get("workflow", {}).get("structural_tactic_lifecycle", True):
            return False
        todo_id = str(record.get("todo_id", ""))
        if not todo_id:
            return False
        stage = str(record.get("tactic_stage", ""))
        if not stage.startswith("structural_"):
            return False
        failure_class = str(record.get("failure_class", ""))
        if failure_class not in {
            "scope_too_broad",
            "invariant_broken",
            "guard_missing",
            "probe_contract_mismatch",
        }:
            return False
        soft_limit = int(
            self.config.get("workflow", {}).get("structural_tactic_soft_failures", 2)
            or 2
        )
        prior_soft = 0
        for attempt in self._recent_todo_attempts(todo_id):
            if (
                isinstance(attempt, dict)
                and attempt.get("budget_counted") is False
                and str(attempt.get("failure_class", "")) in {
                    "scope_too_broad",
                    "invariant_broken",
                    "guard_missing",
                    "probe_contract_mismatch",
                }
            ):
                prior_soft += 1
        return prior_soft < soft_limit

    def _active_todo_id(self) -> str:
        if self._todo_contract_soft_now():
            workflow = self.config.get("workflow", {})
            if workflow.get("pre_improvement_todo_blocks_brainstorm", True) is False:
                return ""
        active_todo = self.state.scratch.get("active_todo")
        if isinstance(active_todo, dict) and active_todo.get("status") in {"active", "attempted"}:
            if self._todo_attempt_budget_exhausted(active_todo):
                return ""
            return str(active_todo.get("todo_id", ""))
        return ""

    def _active_todo_id_for_record(self, record: dict[str, Any]) -> str:
        active_todo = self.state.scratch.get("active_todo")
        if not isinstance(active_todo, dict):
            active_todo = self._load_active_todo()
            if active_todo:
                self.state.scratch["active_todo"] = active_todo
        if not isinstance(active_todo, dict):
            return ""
        if active_todo.get("status") not in {"active", "attempted", "validated"}:
            return ""
        if (
            active_todo.get("status") != "validated"
            and self._todo_attempt_budget_exhausted(active_todo)
        ):
            return ""
        stage = str(record.get("tactic_stage", ""))
        if not stage.startswith("structural_"):
            return ""
        todo_stage = str(active_todo.get("tactic_stage", ""))
        if not todo_stage.startswith("structural_"):
            return ""
        record_axis = self._normalize_strategy_axis(str(record.get("strategy_axis", "")))
        todo_axis = self._normalize_strategy_axis(str(active_todo.get("strategy_axis", "")))
        if record_axis and todo_axis and record_axis != todo_axis:
            axes = record.get("strategy_axes")
            if not isinstance(axes, list) or todo_axis not in {
                self._normalize_strategy_axis(str(axis)) for axis in axes
            }:
                return ""
        return str(active_todo.get("todo_id", ""))

    def _active_todo_spec_task_id(self) -> str:
        if self._todo_contract_soft_now():
            return ""
        active_todo = self.state.scratch.get("active_todo")
        if isinstance(active_todo, dict) and active_todo.get("status") in {"active", "attempted"}:
            if self._todo_attempt_budget_exhausted(active_todo):
                return ""
            return str(active_todo.get("spec_task_id", "") or "")
        return ""

    def _update_run_spec_from_candidate_record(self, record: dict[str, Any]) -> None:
        spec = self.state.scratch.get("run_spec")
        if not isinstance(spec, dict):
            return
        if self._spec_mode_enabled() and int(spec.get("version", 1) or 1) >= 2:
            return
        tasks = spec.get("task_graph")
        if not isinstance(tasks, list):
            return
        task = self._run_spec_task_for_record(tasks, record)
        if task is None:
            return
        task["attempts"] = int(task.get("attempts", 0) or 0) + 1
        task["last_observation"] = {
            "loop": record.get("loop"),
            "status": record.get("status"),
            "metric": record.get("metric"),
            "failure_class": record.get("failure_class"),
            "stage_result": record.get("stage_result"),
            "summary": self._truncate_text(str(record.get("summary", "")), 500),
            "recovery_hint": record.get("recovery_hint"),
        }
        status = str(record.get("status", ""))
        failure_class = str(record.get("failure_class", ""))
        if status in {"improved", "accepted"}:
            task["status"] = "validated"
            task["decision_hint"] = "deepen_or_create_followup_from_validated_signal"
        elif failure_class == "patch_miss":
            task["status"] = "needs_repair"
            task["decision_hint"] = "repair_with_fresh_source_context_before_retry"
        elif failure_class == "duplicate_variant":
            task["status"] = "stale_variant"
            task["decision_hint"] = "pivot_or_reformulate_before_retry"
        elif failure_class in {"invariant_broken", "scope_too_broad", "guard_missing"}:
            task["status"] = "needs_guard_or_smaller_scope"
            task["decision_hint"] = "shrink_scope_or_add_behavior_guard"
        elif failure_class == "probe_no_signal":
            task["status"] = "validated_no_metric_signal"
            task["decision_hint"] = "change_measured_scope_or_retire"
        elif status.startswith("rejected"):
            task["status"] = "attempted"
            task["decision_hint"] = "use_observation_before_next_action"
        spec["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.state.scratch["run_spec"] = spec
        path = self._workflow_artifact_path(
            "run_spec_path", ".local_micro_agent/run_spec.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")

    def _run_spec_task_for_record(
        self, tasks: list[Any], record: dict[str, Any]
    ) -> dict[str, Any] | None:
        spec_task_id = str(record.get("spec_task_id", "") or "")
        if spec_task_id:
            for task in tasks:
                if isinstance(task, dict) and task.get("task_id") == spec_task_id:
                    return task
        axis = self._normalize_strategy_axis(str(record.get("strategy_axis", "")))
        if not axis:
            return None
        for task in tasks:
            if not isinstance(task, dict):
                continue
            if str(task.get("status", "open")) in {"validated", "retired", "failed"}:
                continue
            if self._normalize_strategy_axis(str(task.get("strategy_axis", ""))) == axis:
                return task
        return None

    def _append_todo_attempt(self, candidate_record: dict[str, Any]) -> None:
        todo_id = str(candidate_record.get("todo_id", ""))
        if not todo_id:
            return
        path = self._workflow_artifact_path(
            "todo_attempts_path", ".local_micro_agent/todo_attempts.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        attempt = {
            "ts": candidate_record.get("ts"),
            "loop": candidate_record.get("loop"),
            "todo_id": todo_id,
            "candidate_id": candidate_record.get("candidate_id"),
            "status": candidate_record.get("status"),
            "metric": candidate_record.get("metric"),
            "failed": candidate_record.get("failed"),
            "strategy_axis": candidate_record.get("strategy_axis"),
            "strategy_axes": candidate_record.get("strategy_axes"),
            "tactic_stage": candidate_record.get("tactic_stage"),
            "stage_result": candidate_record.get("stage_result"),
            "reason": candidate_record.get("reason"),
            "spec_task_id": candidate_record.get("spec_task_id"),
        }
        if candidate_record.get("budget_counted") is False:
            attempt["budget_counted"] = False
        for key in (
            "failure_detail",
            "failure_class",
            "summary",
            "next_actions",
            "recovery_hint",
            "no_change_reason",
            "artifact_id",
            "artifact_path",
            "repair_parent_id",
            "patch_path",
            "test_output_path",
            "diagnostic_summary",
            "diagnostics",
            "failure_origin",
            "fingerprint",
            "issue_scope",
            "repo_valid_after_restore",
            "repair_task_eligible",
            "memory_use",
            "diff_contract_violations",
            "probe_diff_summary",
            "probe_diff_contract",
        ):
            value = candidate_record.get(key)
            if value not in (None, "", [], {}):
                attempt[key] = value
        with path.open("a") as handle:
            handle.write(json.dumps(attempt, ensure_ascii=False, sort_keys=True) + "\n")
        if attempt.get("budget_counted") is False:
            self._record_non_budget_todo_attempt(attempt)
            return
        self._update_todo_status_from_attempt(attempt)

    def _record_non_budget_todo_attempt(self, attempt: dict[str, Any]) -> None:
        plan_path = self._workflow_artifact_path(
            "todo_plan_path", ".local_micro_agent/todo_plan.json"
        )
        active_path = self._workflow_artifact_path(
            "active_todo_path", ".local_micro_agent/active_todo.json"
        )
        if not plan_path.exists():
            return
        try:
            plan = json.loads(plan_path.read_text(errors="replace"))
        except json.JSONDecodeError:
            return
        todos = plan.get("todos")
        if not isinstance(todos, list):
            return
        for todo in todos:
            if isinstance(todo, dict) and todo.get("todo_id") == attempt.get("todo_id"):
                todo["last_non_budget_attempt"] = attempt
                todo["non_budget_attempts"] = int(todo.get("non_budget_attempts", 0) or 0) + 1
                patch_detail = self._normalize_fingerprint_text(
                    " ".join(
                        str(attempt.get(key, ""))
                        for key in ("failure_class", "failure_detail", "no_change_reason")
                    )
                )
                if attempt.get("failure_class") == "patch_miss" or any(
                    indicator in patch_detail
                    for indicator in (
                        "target not found",
                        "patch rejected",
                        "patch apply failed",
                        "replacement target is ambiguous",
                        "no writable file content changed",
                        "no changes applied",
                        "no-op",
                        "only changes comments",
                    )
                ):
                    todo["last_patch_failure"] = attempt
                    todo["patch_failures"] = int(todo.get("patch_failures", 0) or 0) + 1
                self.state.scratch["active_todo"] = todo
                if active_path.exists() and plan.get("active_todo_id") == todo.get("todo_id"):
                    active_path.write_text(
                        json.dumps(todo, ensure_ascii=False, indent=2) + "\n"
                    )
                break
        plan["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")

    def _update_todo_status_from_attempt(self, attempt: dict[str, Any]) -> None:
        plan_path = self._workflow_artifact_path(
            "todo_plan_path", ".local_micro_agent/todo_plan.json"
        )
        active_path = self._workflow_artifact_path(
            "active_todo_path", ".local_micro_agent/active_todo.json"
        )
        if not plan_path.exists():
            return
        try:
            plan = json.loads(plan_path.read_text(errors="replace"))
        except json.JSONDecodeError:
            return
        todos = plan.get("todos")
        if not isinstance(todos, list):
            return
        status = str(attempt.get("status", ""))
        for todo in todos:
            if isinstance(todo, dict) and todo.get("todo_id") == attempt.get("todo_id"):
                previous_status = todo.get("status")
                previous_attempts = int(todo.get("attempts", 0) or 0)
                next_attempts = previous_attempts + 1
                next_status = self._todo_status_after_attempt(
                    attempt, previous_status, next_attempts
                )
                if todo.get("status") == "validated" and next_status != "validated":
                    next_status = "validated"
                todo["status"] = next_status
                todo["last_attempt"] = attempt
                todo["attempts"] = next_attempts
                if active_path.exists():
                    active_path.write_text(
                        json.dumps(todo, ensure_ascii=False, indent=2) + "\n"
                    )
                self.state.scratch["active_todo"] = todo
                if plan.get("active_todo_id") == todo.get("todo_id") and next_status in {
                    "failed",
                    "validated",
                }:
                    plan["active_todo_id"] = None
                if not (previous_status == "validated" and status.startswith("rejected")):
                    self._append_todo_outcome_artifact(todo, next_status)
        plan["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")

    def _todo_status_after_attempt(
        self, attempt: dict[str, Any], previous_status: Any, next_attempts: int
    ) -> str:
        status = str(attempt.get("status", ""))
        failure_class = str(attempt.get("failure_class", ""))
        if status in {"improved", "accepted"}:
            return "validated"
        if failure_class in {"scaffold_validated", "probe_no_signal"}:
            return "validated"
        if not status.startswith("rejected"):
            return "attempted"
        if previous_status == "validated":
            return "validated"
        budget = int(self.config.get("workflow", {}).get("todo_attempt_budget", 1) or 1)
        if next_attempts >= budget:
            return "failed"
        return "attempted"

    def _is_patch_application_failure_record(self, record: dict[str, Any]) -> bool:
        workflow = self.config.get("workflow", {})
        if not workflow.get("todo_ignore_patch_failures_for_budget", True):
            return False
        if str(record.get("status", "")) != "rejected_no_changes":
            return False
        reason = self._normalize_fingerprint_text(
            " ".join(
                str(record.get(key, ""))
                for key in ("no_change_reason", "failure_detail")
            )
        )
        indicators = workflow.get("todo_patch_failure_indicators")
        if not isinstance(indicators, list) or not indicators:
            indicators = [
                "target not found",
                "patch rejected",
                "patch apply failed",
                "replacement target is ambiguous",
            ]
        return any(str(indicator).lower() in reason for indicator in indicators)

    @staticmethod
    def _is_active_todo_drift_record(record: dict[str, Any]) -> bool:
        status = str(record.get("status", ""))
        return status in {
            "rejected_active_task_file_drift",
            "rejected_active_task_region_drift",
            "rejected_active_task_shape_drift",
            "rejected_todo_axis_drift",
            "rejected_todo_family_drift",
            "rejected_todo_scope_drift",
        }

    def _append_todo_outcome_artifact(self, todo: dict[str, Any], status: str) -> None:
        if status == "validated":
            path = self._workflow_artifact_path(
                "validated_patterns_path", ".local_micro_agent/validated_patterns.jsonl"
            )
        elif status == "failed":
            path = self._workflow_artifact_path(
                "failed_tactics_path", ".local_micro_agent/failed_tactics.jsonl"
            )
        else:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "todo_id": todo.get("todo_id"),
            "strategy_axis": todo.get("strategy_axis"),
            "family_key": todo.get("family_key") or self._tactic_family_key(str(todo.get("context", ""))),
            "status": status,
            "attempts": todo.get("attempts", 0),
            "context": todo.get("context", ""),
            "last_attempt": todo.get("last_attempt"),
        }
        with path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
