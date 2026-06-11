"""Candidate history, observations, artifacts, and target-not-found repair.

Extracted from orchestrator.py; mixed into MicroAgent.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ..decisions import (
    CodeCandidate,
    CodeDecision,
)
from ..state import (
    CodeChange,
    TestResult,
)
from ..validators import (
    JsonValidationError,
    parse_json_object,
    retry_repair_prompt,
)


class CandidateRecordsMixin:
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
        formatted = []
        for record in records:
            item = {
                "status": record.get("status"),
                "metric": record.get("metric"),
                "failed": record.get("failed"),
                "strategy_axis": record.get("strategy_axis", ""),
                "strategy_axes": record.get("strategy_axes", []),
                "region_keys": record.get("region_keys", []),
                "tactic_stage": record.get("tactic_stage", ""),
                "stage_result": record.get("stage_result", ""),
                "changes": record.get("changes", []),
            }
            for key in (
                "failure_class",
                "failure_origin",
                "issue_scope",
                "repo_valid_after_restore",
                "repair_task_eligible",
                "memory_use",
                "no_change_reason",
                "failure_detail",
                "recovery_hint",
                "repair_parent_id",
                "artifact_id",
                "patch_path",
                "test_output_path",
                "diagnostic_summary",
                "last_correct_state_path",
                "last_correct_patch_path",
            ):
                value = record.get(key)
                if value not in (None, "", [], {}):
                    item[key] = (
                        self._truncate_text(str(value), 500)
                        if key in {"no_change_reason", "failure_detail", "diagnostic_summary"}
                        else value
                    )
            diagnostics = record.get("diagnostics")
            if isinstance(diagnostics, list) and diagnostics:
                item["diagnostics"] = diagnostics[:3]
            formatted.append(item)
        return json.dumps(
            formatted,
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

    def _candidate_rejection_extra(
        self, candidate: CodeCandidate, status: str, failure_detail: str
    ) -> dict[str, Any]:
        return self._candidate_history_extra(
            candidate,
            status=status,
            metric=None,
            applied=0,
            failed=True,
            patch_text="",
            results=[],
            failure_detail=failure_detail,
        )

    def _candidate_history_extra(
        self,
        candidate: CodeCandidate,
        status: str,
        metric: int | None,
        applied: int,
        failed: bool,
        patch_text: str,
        results: list[TestResult],
        failure_detail: str = "",
        no_change_reason: str = "",
        repair_parent_id: str = "",
        diagnostic_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if failure_detail:
            extra["failure_detail"] = self._truncate_text(failure_detail, 2000)
        if no_change_reason:
            extra["no_change_reason"] = self._truncate_text(no_change_reason, 1000)
        if repair_parent_id:
            extra["repair_parent_id"] = repair_parent_id
        observation = self._candidate_observation(
            candidate,
            status=status,
            metric=metric,
            applied=applied,
            failed=failed,
            results=results,
            failure_detail=failure_detail,
            no_change_reason=no_change_reason,
            diagnostic_results=diagnostic_results or [],
        )
        failure_scope = self._candidate_failure_scope(
            status=status,
            applied=applied,
            failed=failed,
            failure_class=str(observation.get("failure_class", "")),
            failure_detail=failure_detail,
            no_change_reason=no_change_reason,
            results=results,
        )
        observation.update(failure_scope)
        extra.update(observation)
        if observation.get("failure_class") == "patch_miss" or repair_parent_id:
            extra.update(self._patch_miss_history_extra())
        if diagnostic_results:
            extra["diagnostics"] = self._compact_diagnostic_results(diagnostic_results)
            diagnostic_summary = self._diagnostic_summary(diagnostic_results)
            if diagnostic_summary:
                extra["diagnostic_summary"] = diagnostic_summary
        extra.update(
            self._remember_failure_lesson(
                candidate,
                status=status,
                metric=metric,
                applied=applied,
                failed=failed,
                failure_class=observation.get("failure_class", ""),
                failure_detail=failure_detail,
                no_change_reason=no_change_reason,
                diagnostic_results=diagnostic_results or [],
                recovery_hint=str(observation.get("recovery_hint", "")),
                failure_scope=failure_scope,
            )
        )
        extra.update(
            self._write_candidate_artifacts(
                candidate,
                status=status,
                metric=metric,
                applied=applied,
                failed=failed,
                patch_text=patch_text,
                results=results,
                failure_detail=failure_detail,
                no_change_reason=no_change_reason,
                repair_parent_id=repair_parent_id,
                observation=observation,
                diagnostic_results=diagnostic_results or [],
            )
        )
        return extra

    def _patch_miss_history_extra(self) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        events = self.state.scratch.get("patch_miss_events")
        if isinstance(events, list) and events:
            latest = events[-1]
            if isinstance(latest, dict):
                extra.update(
                    {
                        key: value
                        for key, value in latest.items()
                        if value not in (None, "", [], {})
                    }
                )
                compact_events = [
                    event
                    for event in events[-3:]
                    if isinstance(event, dict)
                ]
                if len(compact_events) > 1:
                    extra["patch_miss_events"] = compact_events
        parent_events = self.state.scratch.get("repair_parent_patch_miss_events")
        if isinstance(parent_events, list) and parent_events:
            parent_records = [
                event for event in parent_events[-3:] if isinstance(event, dict)
            ]
            if parent_records:
                latest_parent = parent_records[-1]
                extra["repair_parent_patch_miss"] = latest_parent
                extra["repair_parent_patch_miss_kind"] = latest_parent.get(
                    "patch_miss_kind"
                )
                extra["repair_parent_patch_miss_path"] = latest_parent.get(
                    "patch_miss_path"
                )
                if len(parent_records) > 1:
                    extra["repair_parent_patch_miss_events"] = parent_records
        repair_attempted = self.state.scratch.get("patch_miss_repair_attempted")
        repair_status = self.state.scratch.get("patch_miss_repair_status")
        if repair_attempted is not None:
            extra["repair_attempted"] = bool(repair_attempted)
        if repair_status:
            extra["repair_status"] = str(repair_status)
        return extra

    def _candidate_observation(
        self,
        candidate: CodeCandidate,
        status: str,
        metric: int | None,
        applied: int,
        failed: bool,
        results: list[TestResult],
        failure_detail: str = "",
        no_change_reason: str = "",
        diagnostic_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        failure_class = self._candidate_failure_class(
            status=status,
            metric=metric,
            applied=applied,
            failed=failed,
            results=results,
            failure_detail=failure_detail,
            no_change_reason=no_change_reason,
        )
        summary = self._candidate_observation_summary(
            candidate,
            status=status,
            metric=metric,
            applied=applied,
            failure_class=failure_class,
            failure_detail=failure_detail,
            no_change_reason=no_change_reason,
            diagnostic_results=diagnostic_results or [],
        )
        next_actions = self._candidate_next_actions(failure_class)
        recovery_hint = self._candidate_recovery_hint(failure_class)
        tactic_stage = self._active_todo_stage()
        observation: dict[str, Any] = {
            "tactic_stage": tactic_stage,
            "stage_result": self._candidate_stage_result(
                tactic_stage, failure_class, metric=metric, failed=failed
            ),
            "failure_class": failure_class,
            "summary": self._truncate_text(summary, 700),
            "next_actions": [
                self._truncate_text(action, 220)
                for action in next_actions
                if action
            ],
            "recovery_hint": self._truncate_text(recovery_hint, 700),
        }
        return observation

    @staticmethod
    def _candidate_stage_result(
        tactic_stage: str, failure_class: str, metric: int | None, failed: bool
    ) -> str:
        if tactic_stage == "structural_scaffold":
            if not failed and metric is not None:
                return "scaffold_validated"
            if failure_class in {"scope_too_broad", "invariant_broken", "guard_missing"}:
                return "scaffold_needs_smaller_scope"
        if tactic_stage == "structural_probe":
            if failure_class == "probe_no_signal":
                return "probe_validated_no_metric_gain"
            if not failed and metric is not None:
                return "probe_validated"
            if failure_class in {
                "scope_too_broad",
                "invariant_broken",
                "guard_missing",
                "probe_contract_mismatch",
            }:
                return "probe_needs_guard_or_scope"
        if tactic_stage == "structural_expand":
            if failure_class == "no_improvement":
                return "expand_no_improvement"
            if failure_class in {"scope_too_broad", "invariant_broken", "guard_missing"}:
                return "expand_needs_previous_probe"
        return failure_class

    def _candidate_failure_class(
        self,
        status: str,
        metric: int | None,
        applied: int,
        failed: bool,
        results: list[TestResult],
        failure_detail: str = "",
        no_change_reason: str = "",
    ) -> str:
        status_text = str(status)
        detail_text = self._normalize_fingerprint_text(
            " ".join([status_text, failure_detail, no_change_reason])
        )
        if status_text in {"improved", "accepted"}:
            return status_text
        if "duplicate" in status_text or "repeated_pattern" in status_text:
            return "duplicate_variant"
        if "axis_drift" in status_text or "cooled_axis" in status_text:
            return "axis_mismatch"
        if "family_drift" in status_text:
            return "family_mismatch"
        if "probe_contract" in status_text or "probe diff contract" in detail_text:
            return "probe_contract_mismatch"
        if status_text.startswith("rejected_todo"):
            return "contract_mismatch"
        patch_indicators = (
            "target not found",
            "patch rejected",
            "patch apply failed",
            "replacement target is ambiguous",
            "no writable file content changed",
            "no changes applied",
            "no-op",
            "only changes comments",
        )
        if status_text == "rejected_no_changes" or any(
            indicator in detail_text for indicator in patch_indicators
        ):
            return "patch_miss"
        tactic_stage = self._active_todo_stage()
        structural = tactic_stage.startswith("structural_")
        if any(result.exit_code != 0 for result in results):
            if structural:
                if re.search(r"\b(assert|invariant|mismatch|expected|actual)\b", detail_text):
                    return "invariant_broken"
                return "scope_too_broad"
            return "correctness_failure"
        if metric is None and (failed or results):
            return "metric_missing"
        if metric is not None and not failed and status_text.startswith("rejected"):
            if tactic_stage == "structural_scaffold":
                return "scaffold_validated"
            if tactic_stage == "structural_probe":
                return "probe_no_signal"
            return "no_improvement"
        if failed:
            if structural:
                return "scope_too_broad"
            return "correctness_failure"
        return "rejected"

    def _candidate_observation_summary(
        self,
        candidate: CodeCandidate,
        status: str,
        metric: int | None,
        applied: int,
        failure_class: str,
        failure_detail: str = "",
        no_change_reason: str = "",
        diagnostic_results: list[dict[str, Any]] | None = None,
    ) -> str:
        axis = candidate.strategy_axis or ",".join(self._candidate_strategy_axes(candidate))
        parts = [
            f"{failure_class}: status={status}",
            f"applied={applied}",
            f"metric={metric}",
        ]
        if axis:
            parts.append(f"axis={axis}")
        detail = no_change_reason or failure_detail
        if detail:
            parts.append(self._truncate_text(detail, 280))
        diagnostic_summary = self._diagnostic_summary(diagnostic_results or [], limit=320)
        if diagnostic_summary:
            parts.append(f"diagnostics={diagnostic_summary}")
        return "; ".join(parts)

    def _compact_diagnostic_results(
        self, diagnostic_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for result in diagnostic_results:
            output = "\n".join(
                part
                for part in (
                    str(result.get("stdout", "")).strip(),
                    str(result.get("stderr", "")).strip(),
                )
                if part
            )
            compact.append(
                {
                    "name": result.get("name"),
                    "command": result.get("command"),
                    "exit_code": result.get("exit_code"),
                    "output": self._truncate_text(output, 1200),
                }
            )
        return compact

    @staticmethod
    def _candidate_next_actions(failure_class: str) -> list[str]:
        actions_by_class = {
            "patch_miss": [
                "Re-read the current file region before editing.",
                "Retarget the smallest replacement or patch hunk that still exists.",
            ],
            "correctness_failure": [
                "Use the failing command output as the next hypothesis.",
                "Make a smaller semantic change that preserves observed invariants.",
            ],
            "scope_too_broad": [
                "Retry the same structural tactic at a smaller guarded scope.",
                "Add or tighten the invariant guard before expanding behavior.",
            ],
            "invariant_broken": [
                "Identify the exact invariant broken by the test failure.",
                "Preserve the old behavior by default, then enable only a guarded probe.",
            ],
            "guard_missing": [
                "Add an explicit safety predicate before applying the structural change.",
                "Prefer a scaffold that can fall back to the original path.",
            ],
            "metric_missing": [
                "Preserve the configured metric output path and format.",
                "Run the benchmark command locally before changing tactic.",
            ],
            "no_improvement": [
                "Do not repeat the same tactic shape.",
                "Keep correctness, but change the bottleneck hypothesis or edit location.",
            ],
            "probe_no_signal": [
                "Keep the validated structural probe, but change the measured scope.",
                "Do not treat this as a correctness failure; seek a narrower performance signal.",
            ],
            "scaffold_validated": [
                "Build the next guarded probe on this scaffold.",
                "Keep the scaffold behavior-preserving until a probe has evidence.",
            ],
            "duplicate_variant": [
                "Change the implementation family or touched code region.",
                "Avoid cosmetic rewrites of the same candidate fingerprint.",
            ],
            "axis_mismatch": [
                "Align the declared strategy_axis with the active contract.",
                "If the contract is stale, request or produce evidence before switching axis.",
            ],
            "family_mismatch": [
                "Stay within the active family_key or explicitly create a new todo.",
                "Avoid mixing unrelated tactic families in a follow-up attempt.",
            ],
            "contract_mismatch": [
                "Read the active todo contract and produce a candidate that satisfies it.",
                "If the contract blocks valid work, surface that as evidence for relaxation.",
            ],
        }
        return actions_by_class.get(
            failure_class,
            ["Use the structured failure fields before choosing the next edit."],
        )

    @staticmethod
    def _candidate_recovery_hint(failure_class: str) -> str:
        hints = {
            "patch_miss": (
                "Recover by refreshing file context and editing an exact current target; "
                "do not count this as evidence that the tactic is bad."
            ),
            "correctness_failure": (
                "Recover from the concrete failing assertion/command before optimizing further."
            ),
            "scope_too_broad": (
                "Treat this as structural learning, not tactic failure: shrink scope, add guards, "
                "and preserve fallback behavior before retrying."
            ),
            "invariant_broken": (
                "Recover by naming the broken invariant and making the next structural probe "
                "behavior-preserving by default."
            ),
            "guard_missing": (
                "Recover by adding an explicit guard or fallback path before enabling the change."
            ),
            "metric_missing": (
                "Recover by restoring benchmark/metric observability before judging speed."
            ),
            "no_improvement": (
                "Treat this as valid negative performance evidence; vary the tactic, axis, "
                "or edit site rather than retrying the same shape."
            ),
            "probe_no_signal": (
                "Treat this as a validated probe without performance gain; expand or move the "
                "probe only if the scaffold stayed correct."
            ),
            "scaffold_validated": (
                "Persist the scaffold and ask the next CODE attempt for one guarded probe."
            ),
            "duplicate_variant": (
                "Recover by making a substantively different candidate, not a renamed variant."
            ),
            "axis_mismatch": (
                "Recover by satisfying the active axis contract or creating evidence that it "
                "should be relaxed."
            ),
            "family_mismatch": (
                "Recover by keeping the active family coherent until the todo is exhausted."
            ),
            "contract_mismatch": (
                "Recover by following the visible active contract, or produce structured "
                "evidence for a controller-level relaxation."
            ),
            "improved": "Persist the validated pattern and explore follow-ups from the measured gain.",
            "accepted": "Persist the accepted result and stop unless the workflow asks to continue.",
        }
        return hints.get(
            failure_class,
            "Recover by using the status, metric, and artifacts as the next observation.",
        )

    async def _repair_target_not_found_candidate(
        self,
        candidate: CodeCandidate,
        failure_detail: str,
        allowed: set[str],
    ) -> CodeCandidate | None:
        workflow = self.config.get("workflow", {})
        if not workflow.get("repair_target_not_found"):
            return None
        if "Replacement target not found" not in failure_detail:
            return None
        self.state.scratch["patch_miss_repair_attempted"] = True
        self.state.scratch["patch_miss_repair_status"] = "requested"
        messages = await self._target_not_found_repair_prompt(
            candidate,
            failure_detail=failure_detail,
            allowed=allowed,
        )
        try:
            decision = await self._target_not_found_repair_call(candidate, messages)
        except JsonValidationError as exc:
            self.state.scratch["patch_miss_repair_status"] = "parse_failed"
            self.state.notes.append(
                f"Candidate {candidate.candidate_id} target-not-found repair rejected: {exc}"
            )
            return None
        if not decision.candidates:
            self.state.scratch["patch_miss_repair_status"] = "empty"
            self.state.notes.append(
                f"Candidate {candidate.candidate_id} target-not-found repair returned no candidate"
            )
            return None
        repaired = decision.candidates[0]
        repaired.candidate_id = f"{candidate.candidate_id}-repair1"
        if not repaired.reason:
            repaired.reason = candidate.reason
        if not repaired.strategy_axis:
            repaired.strategy_axis = candidate.strategy_axis
        rejection = (
            self._active_todo_contract_rejection(repaired)
            or self._candidate_axis_contract_rejection(repaired)
            or self._candidate_family_contract_rejection(repaired)
        )
        if rejection is not None:
            _status, note = rejection
            self.state.scratch["patch_miss_repair_status"] = "scope_rejected"
            self.state.notes.append(
                f"Candidate {candidate.candidate_id} target-not-found repair rejected: {note}"
            )
            return None
        target_miss = await self._candidate_current_target_miss(repaired, allowed)
        if target_miss:
            repair_status = "ambiguous" if "ambiguous" in target_miss else "still_missing"
            self.state.scratch["patch_miss_repair_status"] = repair_status
            self.state.notes.append(
                "Candidate "
                f"{candidate.candidate_id} target-not-found repair rejected: {target_miss}"
            )
            return None
        self.state.scratch["patch_miss_repair_status"] = "generated"
        return repaired

    async def _candidate_current_target_miss(
        self, candidate: CodeCandidate, allowed: set[str]
    ) -> str:
        for change in candidate.changes:
            if change.target is None or change.replacement is None:
                continue
            if change.path not in allowed:
                return f"repaired change touches out-of-plan file {change.path}"
            try:
                content = await self.mcp.read_file(str(self.state.repo_root / change.path))
            except FileNotFoundError:
                return f"repaired change targets missing file {change.path}"
            resolved = self._resolve_replacement_target(content, change)
            if resolved is None:
                latest_kind = ""
                events = self.state.scratch.get("patch_miss_events")
                if isinstance(events, list) and events and isinstance(events[-1], dict):
                    latest_kind = str(events[-1].get("patch_miss_kind", ""))
                if latest_kind == "ambiguous_target":
                    return f"repaired target is ambiguous in {change.path}"
                return f"repaired target still not found in {change.path}"
            _start, _end, retargeted, retarget_mode = resolved
            if retarget_mode:
                change.target = retargeted
                if retargeted.endswith("\n") and not change.replacement.endswith("\n"):
                    change.replacement += "\n"
                qualifier = " whitespace" if "whitespace" in retarget_mode else ""
                self.state.notes.append(
                    f"Retargeted repaired search block to exact current source{qualifier} "
                    f"in {change.path}"
                )
            has_location_hint = any(
                (
                    change.start_line is not None,
                    change.end_line is not None,
                    bool(change.anchor_before),
                    bool(change.anchor_after),
                )
            )
            if content.count(change.target) != 1 and not has_location_hint:
                return f"repaired target is ambiguous in {change.path}"
        return ""

    @staticmethod
    def _unique_stripped_line_match(content: str, target: str) -> str | None:
        target_lines = target.splitlines()
        if not target_lines:
            return None
        target_key = [line.strip() for line in target_lines]
        content_lines = content.splitlines(keepends=True)
        width = len(target_lines)
        matches: list[str] = []
        for index in range(0, len(content_lines) - width + 1):
            window = content_lines[index : index + width]
            if [line.strip() for line in window] == target_key:
                matches.append("".join(window))
                if len(matches) > 1:
                    return None
        return matches[0] if matches else None

    async def _target_not_found_repair_prompt(
        self,
        candidate: CodeCandidate,
        failure_detail: str,
        allowed: set[str],
    ) -> list[dict[str, str]]:
        output_format = str(self.config.get("workflow", {}).get("code_output_format", "json"))
        source_context = await self._candidate_repair_source_context(candidate, allowed)
        candidate_record = {
            "candidate_id": candidate.candidate_id,
            "reason": candidate.reason,
            "strategy_axis": candidate.strategy_axis,
            "strategy_axes": self._candidate_strategy_axes(candidate),
            "changes": [
                {
                    "path": change.path,
                    "reason": change.reason,
                    "target": self._truncate_text(change.target or "", 4000),
                    "replacement": self._truncate_text(change.replacement or "", 4000),
                    "target_region": change.target_region,
                    "start_line": change.start_line,
                    "end_line": change.end_line,
                    "anchor_before": self._truncate_text(change.anchor_before or "", 1000),
                    "anchor_after": self._truncate_text(change.anchor_after or "", 1000),
                    "patch": self._truncate_text(change.patch or "", 4000),
                    "content": self._truncate_text(change.content or "", 4000),
                }
                for change in candidate.changes
            ],
            "active_todo_id": self._active_todo_id(),
        }
        if output_format == "xml":
            system = (
                "Repair one failed CODE candidate. The previous candidate was rejected "
                "because a <search> block did not match the current source. Output "
                "exactly one <candidate> in the same XML-like CODE format, with one "
                "<change>. Do not invent a new tactic. Preserve the strategy_axis and "
                "todo context. The new <search> block must be copied verbatim from the "
                "current source below and must match exactly. Preserve or add line/anchor "
                "location hints when the output format supports them; line numbers are "
                "hints only and must not appear inside <search> or <replace>."
            )
        else:
            system = (
                "Repair one failed CODE candidate. The previous candidate was rejected "
                "because a target string did not match the current source. Output strict "
                "JSON with a top-level candidates array containing exactly one candidate "
                "and one change. Do not invent a new tactic. Preserve the strategy_axis "
                "and todo context. The new target must be copied verbatim from the "
                "current source below and must match exactly. Include start_line/end_line "
                "and anchor_before/anchor_after hints when available, without copying "
                "line-number prefixes into target or replacement."
            )
        user = (
            f"Failure detail:\n{failure_detail}\n\n"
            "Original candidate summary:\n"
            f"{json.dumps(candidate_record, ensure_ascii=False, indent=2)}\n\n"
            "Current source context for repair:\n"
            f"{source_context}\n\n"
            "Return only the repaired candidate. Change the search/target text to match "
            "the current source exactly, and keep the replacement focused on the same "
            "intended edit."
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if output_format == "xml":
            messages.append(self._candidate_queue_message("xml"))
        else:
            messages.append(self._candidate_queue_message("json"))
        return messages

    async def _candidate_repair_source_context(
        self, candidate: CodeCandidate, allowed: set[str]
    ) -> str:
        limit = int(
            self.config.get("workflow", {}).get("repair_source_context_char_limit", 20000)
        )
        context_lines = int(
            self.config.get("workflow", {}).get("repair_anchor_context_lines", 18) or 18
        )
        blocks = []
        seen = set()
        for change in candidate.changes:
            if change.path in seen or change.path not in allowed:
                continue
            seen.add(change.path)
            try:
                content = await self.mcp.read_file(str(self.state.repo_root / change.path))
            except FileNotFoundError:
                blocks.append(f"### {change.path}\n<missing>")
                continue
            anchor_excerpt = ""
            if change.target:
                anchor_limit = min(limit, 6000)
                anchor_excerpt = self._best_anchor_excerpt(
                    content,
                    change.target,
                    context_lines=context_lines,
                    limit=anchor_limit,
                )
            if anchor_excerpt:
                blocks.append(
                    "### "
                    f"{change.path}\n"
                    "Original missing target/search text follows; it did not match the "
                    "file exactly. Use it only to find the intended nearby region.\n"
                    "```text\n"
                    f"{self._truncate_text(change.target or '', 3000)}\n"
                    "```\n"
                    "Best current-source excerpt for that region follows with line numbers. "
                    "Copy the repaired target/search text verbatim from these current lines, "
                    "without the line-number prefixes.\n"
                    "```text\n"
                    f"{anchor_excerpt}\n"
                    "```"
                )
                continue
            blocks.append(
                f"### {change.path}\n```text\n{self._line_numbered_context(content, limit)}\n```"
            )
        return "\n\n".join(blocks) if blocks else "No writable source context available."

    async def _target_not_found_repair_call(
        self, candidate: CodeCandidate, messages: list[dict[str, str]]
    ) -> CodeDecision:
        try:
            output = await self._model_chat(
                "coder",
                messages,
                call_site="target_not_found_repair",
            )
        except Exception as exc:
            raise JsonValidationError(
                f"coder target-not-found repair model call failed: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            return self._parse_decision(output, CodeDecision)
        except JsonValidationError as exc:
            try:
                return self._parse_loose_target_not_found_repair(output, candidate)
            except JsonValidationError:
                self._record_raw_model_output(
                    "coder", "target-not-found-repair", output, exc
                )
                try:
                    repaired = await self._model_chat(
                        "coder",
                        retry_repair_prompt(output, exc),
                        call_site="target_not_found_repair_json_repair",
                    )
                except Exception as repair_call_exc:
                    raise JsonValidationError(
                        "target-not-found repair JSON repair model call failed: "
                        f"{type(repair_call_exc).__name__}: {repair_call_exc}"
                    ) from repair_call_exc
                try:
                    return self._parse_decision(repaired, CodeDecision)
                except JsonValidationError as repair_parse_exc:
                    try:
                        return self._parse_loose_target_not_found_repair(
                            repaired, candidate
                        )
                    except JsonValidationError:
                        self._record_raw_model_output(
                            "coder",
                            "target-not-found-repair-json",
                            repaired,
                            repair_parse_exc,
                        )
                        raise JsonValidationError(
                            f"target-not-found repair parse failed: {repair_parse_exc}"
                        ) from repair_parse_exc

    def _parse_loose_target_not_found_repair(
        self, output: str, original: CodeCandidate
    ) -> CodeDecision:
        if not original.changes:
            raise JsonValidationError("No original change available for loose repair")
        original_change = original.changes[0]
        path = self._loose_repair_field(output, "path") or original_change.path
        target = (
            self._loose_repair_field(output, "search")
            or self._loose_repair_field(output, "target")
        )
        replacement = (
            self._loose_repair_field(output, "replace")
            or self._loose_repair_field(output, "replacement")
        )
        if not target or not replacement:
            raise JsonValidationError("Loose repair output missing search/replace")
        reason = self._loose_repair_field(output, "reason") or original_change.reason
        strategy_axis = self._loose_repair_field(output, "strategy_axis") or original.strategy_axis
        change = CodeChange(
            path=path,
            reason=reason,
            target=self._trim_repair_block(target),
            replacement=self._trim_repair_block(replacement),
        )
        candidate_id = self._loose_repair_candidate_id(output) or original.candidate_id
        repaired = CodeCandidate(
            candidate_id=candidate_id,
            changes=[change],
            reason=reason or original.reason,
            strategy_axis=strategy_axis,
        )
        return CodeDecision(changes=[change], candidates=[repaired])

    def _loose_repair_field(self, output: str, field: str) -> str:
        match = re.search(rf"<{field}>(.*?)</{field}>", output, re.DOTALL)
        if match:
            return match.group(1)
        try:
            data = parse_json_object(output)
        except JsonValidationError:
            return ""
        candidates = data.get("candidates")
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
            data = candidates[0]
        value = data.get(field)
        if value is None and field == "search":
            value = data.get("target")
        if value is None and field == "replace":
            value = data.get("replacement")
        return str(value) if value is not None else ""

    @staticmethod
    def _loose_repair_candidate_id(output: str) -> str:
        match = re.search(r"<candidate(?:\s+id=\"([^\"]*)\")?", output)
        if match and match.group(1):
            return match.group(1).strip()
        try:
            data = parse_json_object(output)
        except JsonValidationError:
            return ""
        candidates = data.get("candidates")
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
            return str(candidates[0].get("id") or "")
        return str(data.get("id") or "")

    @staticmethod
    def _trim_repair_block(text: str) -> str:
        lines = text.splitlines()
        if lines and not lines[0].strip():
            lines = lines[1:]
        if lines and not lines[-1].strip():
            lines = lines[:-1]
        return "\n".join(lines)

    def _candidate_failure_detail(
        self,
        notes: list[str],
        results: list[TestResult],
        failed: bool,
    ) -> str:
        details: list[str] = []
        note_text = "; ".join(
            self._truncate_text(note, 400)
            for note in notes
            if note.strip()
        )
        if note_text:
            details.append(note_text)
        if failed:
            for result in results:
                if result.exit_code == 0:
                    continue
                output = "\n".join(
                    part
                    for part in (
                        result.stdout[-1200:],
                        result.stderr[-1200:],
                    )
                    if part
                )
                command_detail = (
                    f"command={result.command!r} exit_code={result.exit_code}"
                )
                if output:
                    command_detail += f" output_tail={self._truncate_text(output, 1600)}"
                details.append(command_detail)
        return self._truncate_text(" | ".join(details), 2500)

    def _candidate_artifact_dir(self) -> Path | None:
        workflow = self.config.get("workflow", {})
        if not workflow.get("record_candidate_artifacts"):
            return None
        return self._workflow_artifact_path(
            "candidate_artifact_dir", ".local_micro_agent/candidate_artifacts"
        )

    def _candidate_artifact_id(self, candidate: CodeCandidate) -> str:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", candidate.candidate_id).strip("-")
        if not safe_id:
            safe_id = "candidate"
        return f"loop-{self.state.loop_count:03d}-{safe_id[:80]}"

    def _write_candidate_artifacts(
        self,
        candidate: CodeCandidate,
        status: str,
        metric: int | None,
        applied: int,
        failed: bool,
        patch_text: str,
        results: list[TestResult],
        failure_detail: str = "",
        no_change_reason: str = "",
        repair_parent_id: str = "",
        observation: dict[str, Any] | None = None,
        diagnostic_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        artifact_dir = self._candidate_artifact_dir()
        if artifact_dir is None:
            return {}
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_id = self._candidate_artifact_id(candidate)
        metadata_path = artifact_dir / f"{artifact_id}.json"
        patch_path = artifact_dir / f"{artifact_id}.patch"
        test_output_path = artifact_dir / f"{artifact_id}.test.txt"
        metadata: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "loop": self.state.loop_count,
            "candidate_id": candidate.candidate_id,
            "status": status,
            "metric": metric,
            "applied": applied,
            "failed": failed,
            "reason": candidate.reason,
            "strategy_axis": candidate.strategy_axis,
            "strategy_axes": self._candidate_strategy_axes(candidate),
            "changes": self._summarize_changes(candidate.changes),
        }
        if failure_detail:
            metadata["failure_detail"] = self._truncate_text(failure_detail, 4000)
        if no_change_reason:
            metadata["no_change_reason"] = self._truncate_text(no_change_reason, 2000)
        if repair_parent_id:
            metadata["repair_parent_id"] = repair_parent_id
        if diagnostic_results:
            metadata["diagnostics"] = self._compact_diagnostic_results(diagnostic_results)
        if observation:
            metadata.update(
                {
                    key: value
                    for key, value in observation.items()
                    if value not in (None, "", [], {})
                }
            )
        output_limit = int(
            self.config.get("workflow", {}).get("candidate_artifact_output_limit", 12000)
        )
        if patch_text:
            patch_path.write_text(patch_text)
            metadata["patch_path"] = self._repo_relative_path(patch_path)
        if results:
            test_text = self._format_test_results_for_artifact(results, output_limit)
            if test_text.strip():
                test_output_path.write_text(test_text)
                metadata["test_output_path"] = self._repo_relative_path(test_output_path)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        extra = {
            "artifact_id": artifact_id,
            "artifact_path": self._repo_relative_path(metadata_path),
        }
        for key in ("patch_path", "test_output_path"):
            if key in metadata:
                extra[key] = metadata[key]
        return extra

    def _persist_correct_survivor(
        self,
        candidate: CodeCandidate,
        status: str,
        metric: int | None,
        patch_text: str,
        results: list[TestResult],
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        workflow = self.config.get("workflow", {})
        if workflow.get("preserve_correct_survivors", True) is False:
            return {}
        if not patch_text.strip():
            return {}
        state_path = self._workflow_artifact_path(
            "last_correct_state_path",
            ".local_micro_agent/last_correct_state.json",
        )
        patch_path = self._workflow_artifact_path(
            "last_correct_patch_path",
            ".local_micro_agent/last_correct.patch",
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(patch_text)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "loop": self.state.loop_count,
            "candidate_id": candidate.candidate_id,
            "status": status,
            "metric": metric,
            "reason": candidate.reason,
            "strategy_axis": candidate.strategy_axis,
            "strategy_axes": self._candidate_strategy_axes(candidate),
            "family_aliases": sorted(self._candidate_reason_family_aliases(candidate)),
            "region_keys": self._candidate_region_keys(candidate),
            "changes": self._summarize_changes(candidate.changes),
            "patch_path": self._repo_relative_path(patch_path),
            "test_commands": [result.command for result in results],
            "failure_class": observation.get("failure_class"),
            "stage_result": observation.get("stage_result"),
            "summary": observation.get("summary"),
        }
        state_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        self.state.scratch["last_correct_metric"] = metric
        self.state.scratch["last_correct_patch_path"] = self._repo_relative_path(patch_path)
        self.state.scratch["last_correct_state_path"] = self._repo_relative_path(state_path)
        return {
            "last_correct_state_path": self._repo_relative_path(state_path),
            "last_correct_patch_path": self._repo_relative_path(patch_path),
        }

    def _format_test_results_for_artifact(
        self, results: list[TestResult], limit: int
    ) -> str:
        blocks = []
        for result in results:
            blocks.append(
                "\n".join(
                    [
                        f"$ {result.command}",
                        f"exit_code={result.exit_code}",
                        "stdout:",
                        result.stdout,
                        "stderr:",
                        result.stderr,
                    ]
                )
            )
        return self._slice_text("\n\n".join(blocks), limit)

    def _repo_relative_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.state.repo_root))
        except ValueError:
            return str(path)

    def _append_candidate_history(
        self,
        candidate: CodeCandidate,
        status: str,
        metric: int | None,
        applied: int,
        failed: bool,
        extra: dict[str, Any] | None = None,
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
            "family_aliases": sorted(self._candidate_reason_family_aliases(candidate)),
            "region_keys": self._candidate_region_keys(candidate),
            "changes": self._summarize_changes(candidate.changes),
            "todo_id": self._active_todo_id(),
            "spec_task_id": self._active_todo_spec_task_id(),
        }
        if extra:
            record.update(
                {
                    key: value
                    for key, value in extra.items()
                    if value not in (None, "", [], {})
                }
            )
        if not record.get("todo_id"):
            structural_todo_id = self._active_todo_id_for_record(record)
            if structural_todo_id:
                record["todo_id"] = structural_todo_id
        if self._is_patch_application_failure_record(record):
            record["budget_counted"] = False
        if self._is_structural_learning_record(record):
            record["budget_counted"] = False
        with path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._append_todo_attempt(record)
        self._update_run_spec_from_candidate_record(record)

    @staticmethod
    def _summarize_changes(changes: list[CodeChange]) -> list[dict[str, Any]]:
        summary = []
        for change in changes:
            mode = "empty"
            if change.target is not None and change.replacement is not None:
                mode = "replacement"
            elif change.patch:
                mode = "patch"
            elif change.content is not None:
                mode = "content"
            item = {
                "path": change.path,
                "reason": change.reason,
                "mode": mode,
            }
            if change.target_region:
                item["target_region"] = change.target_region
            if change.start_line is not None or change.end_line is not None:
                item["line_range"] = {
                    "start_line": change.start_line,
                    "end_line": change.end_line,
                }
            summary.append(item)
        return summary
