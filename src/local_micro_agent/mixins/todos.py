"""Durable todo lifecycle, run-spec task graph, and structural checkpoints.

Extracted from orchestrator.py; mixed into MicroAgent.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ..decisions import CodeCandidate
from ..prompts import spec_prompt
from ..validators import parse_json_object


class TodoLifecycleMixin:
    async def _maybe_refresh_run_spec(self) -> None:
        workflow = self.config.get("workflow", {})
        path = self._workflow_artifact_path(
            "run_spec_path", ".local_micro_agent/run_spec.json"
        )
        if not self._run_spec_enabled():
            self.state.scratch.pop("run_spec", None)
            return
        if workflow.get("run_spec_after_read"):
            self.state.scratch.pop("run_spec", None)
            focus = self._focused_read_model_context(str(workflow.get("run_spec_focus", "")))
            role = str(workflow.get("run_spec_model_role", "planner"))
            try:
                output = await self._model_chat(
                    role,
                    spec_prompt(self.state, focus=focus),
                    call_site="run_spec",
                )
                spec = parse_json_object(output)
            except Exception as exc:
                self.state.notes.append(
                    f"Run spec model call failed: {type(exc).__name__}: {exc}"
                )
                return
            spec = self._normalize_run_spec(spec)
            if not spec:
                self.state.notes.append("Run spec discarded: no task_graph")
                return
            self.state.scratch["run_spec"] = spec
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
            self.state.notes.append(f"Persisted run spec: {path}")
            return
        if path.exists():
            spec = self._load_run_spec(path)
            if spec:
                self.state.scratch["run_spec"] = spec
                self.state.notes.append(f"Loaded run spec: {path}")

    def _run_spec_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("run_spec_after_read") or workflow.get("run_spec_enabled"))

    @staticmethod
    def _load_run_spec(path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(errors="replace"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _normalize_run_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(spec, dict):
            return {}
        tasks = spec.get("task_graph")
        if not isinstance(tasks, list) or not tasks:
            return {}
        normalized_tasks = []
        known_axes = set(self._known_strategy_axes())
        for index, task in enumerate(tasks, start=1):
            if not isinstance(task, dict):
                continue
            axis = self._normalize_strategy_axis(str(task.get("strategy_axis", "")))
            if not axis:
                axis = "general_edit"
            if self._strict_strategy_axis_pool_enabled() and axis not in known_axes:
                axis = "general_edit"
            task_id = str(task.get("task_id") or f"task-{index:03d}").strip()
            normalized_tasks.append(
                {
                    "task_id": task_id,
                    "title": str(task.get("title") or task_id).strip(),
                    "strategy_axis": axis,
                    "family_key": self._normalize_strategy_axis(
                        str(task.get("family_key", ""))
                    ),
                    "expected_signal": str(task.get("expected_signal", "")).strip(),
                    "status": str(task.get("status") or "open").strip(),
                    "attempts": int(task.get("attempts", 0) or 0),
                    "last_observation": task.get("last_observation", ""),
                    "decision_hint": task.get("decision_hint", ""),
                }
            )
        if not normalized_tasks:
            return {}
        return {
            "version": 1,
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

    def _todo_soft_until_first_improvement_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("todo_soft_until_first_improvement", True))

    def _todo_contract_soft_now(self) -> bool:
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
        if failure_class not in {"scope_too_broad", "invariant_broken", "guard_missing"}:
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
        ):
            value = candidate_record.get(key)
            if value:
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
