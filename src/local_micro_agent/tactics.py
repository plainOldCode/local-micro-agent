"""BRAINSTORM tactic generation, scoring, gating, and tactic-to-todo creation.

Extracted from orchestrator.py; mixed into MicroAgent.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .prompts import brainstorm_prompt
from .strategy import (
    explicit_tactic_family_key,
    extract_tactic_axis,
    signature_similarity,
    tactic_novelty_lane,
    tactic_signature,
)


class BrainstormTacticsMixin:
    async def _brainstorm(self) -> None:
        reject_summary = self._format_recent_reject_summary()
        feedback_notes_limit = int(
            self.config.get("workflow", {}).get("brainstorm_feedback_notes_limit", 8)
        )
        try:
            new_family_required = self._brainstorm_new_family_required()
            output = await self._model_chat(
                "brainstorm",
                brainstorm_prompt(
                    self.state,
                    reject_summary=reject_summary,
                    cooled_axes=self._current_cooled_axes(),
                    known_axes=self._brainstorm_known_axes(),
                    todo_ledger_summary=self._format_todo_ledger_summary(),
                    forbidden_family_aliases=self._forbidden_tactic_family_aliases()
                    if new_family_required
                    else [],
                    open_novelty_lanes=self._open_novelty_lanes()
                    if self._brainstorm_open_novelty_lanes_enabled()
                    else [],
                    new_family_required=new_family_required,
                    feedback_notes_limit=feedback_notes_limit,
                ),
                call_site="brainstorm",
            )
        except Exception as exc:
            self.state.notes.append(
                f"Brainstorm model call failed: {type(exc).__name__}: {exc}"
            )
            return
        brainstorm = output.strip()
        if brainstorm:
            self.state.scratch["tactic_library"] = brainstorm
            selected_tactic = self._select_brainstorm_tactic(brainstorm)
            if selected_tactic:
                self.state.scratch["selected_tactic"] = selected_tactic
                self.state.scratch["selected_tactic_loop"] = self.state.loop_count
                self._create_active_todo_from_selected_tactic(selected_tactic)
                self.state.notes.append(
                    "Selected brainstorm tactic axis: "
                    f"{selected_tactic.get('strategy_axis')}"
                )
            self.state.scratch["last_brainstorm_loop"] = self.state.loop_count
            self._persist_brainstorm_tactics(brainstorm, reject_summary)
            self._persist_brainstorm_selection()
            self.state.notes.append("Brainstorm tactics added for next CODE attempt")

    def _brainstorm_refresh_read_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("brainstorm_refresh_read_after_selection"))

    def _prepare_brainstorm_refresh_epoch(self) -> None:
        selected_tactic = self._selected_tactic_for_current_loop()
        if not selected_tactic:
            return
        failure_memory = self._format_failure_memory()
        focus = {
            "selected_tactic": selected_tactic,
            "recent_reject_summary": self._format_recent_reject_summary(),
            "failure_memory": failure_memory,
            "current_best_metric": self.state.scratch.get("best_metric"),
            "instruction": (
                "Refresh file context, semantic analysis, and run spec around this "
                "brainstorm hypothesis before the next CODE attempt. Preserve accepted "
                "patterns, and use failure memory to avoid invalid diagnostic shortcuts."
            ),
        }
        self.state.scratch["brainstorm_refresh_focus"] = focus
        self.state.scratch["focused_read_context"] = json.dumps(
            focus, ensure_ascii=False, indent=2
        )
        self.state.notes.append(
            "Brainstorm selected tactic promoted to focused READ/SPEC refresh"
        )

    def _select_brainstorm_tactic(self, brainstorm: str) -> dict[str, str] | None:
        strict_axis_pool = self._strict_strategy_axis_pool_enabled()
        known_axes = set(self._brainstorm_known_axes())
        failed_signatures = self._failed_tactic_signatures()
        failed_family_keys = self._failed_tactic_family_keys()
        selection_records: list[dict[str, Any]] = []
        selectable: list[tuple[float, int, dict[str, str], dict[str, Any]]] = []
        self.state.scratch.pop("brainstorm_all_tactics_failed_loop", None)
        blocks = re.split(r"\n(?=\s*\d+\.)", brainstorm.strip())
        for order, block in enumerate(blocks):
            declared_axis, axis_source = self._extract_tactic_axis(block, known_axes)
            if not declared_axis:
                continue
            axis = declared_axis
            explicit_family_key = self._explicit_tactic_family_key(block)
            family_key = self._tactic_family_key(block)
            family_axes = self._family_key_strategy_axes(family_key)
            family_aliases = sorted(self._tactic_family_aliases(block))
            axis_normalized_from = ""
            if axis not in known_axes:
                family_axis = self._canonical_axis_from_family_key(family_key, known_axes)
                if family_axis:
                    axis = family_axis
                    axis_normalized_from = "family_key"
            if strict_axis_pool and axis not in self._strategy_axis_pool():
                selection_records.append(
                    {
                        "axis": axis,
                        "declared_axis": declared_axis,
                        "family_key": family_key,
                        "family_aliases": family_aliases,
                        "selected": False,
                        "skipped": True,
                        "reason": "unknown_axis",
                    }
                )
                if axis_source:
                    selection_records[-1]["axis_source"] = axis_source
                continue
            if explicit_family_key and self._brainstorm_axis_family_mismatch_reject_enabled():
                if family_axes and axis not in family_axes:
                    selection_records.append(
                        {
                            "axis": axis,
                            "declared_axis": declared_axis,
                            "family_key": family_key,
                            "family_aliases": family_aliases,
                            "family_axes": family_axes,
                            "selected": False,
                            "skipped": True,
                            "reason": "axis_family_mismatch",
                        }
                    )
                    if axis_source:
                        selection_records[-1]["axis_source"] = axis_source
                    self.state.notes.append(
                        "Skipped brainstorm tactic "
                        f"axis={axis} family_key={family_key} reason=axis_family_mismatch"
                    )
                    continue
            failed_match = self._failed_tactic_match_reason(
                block, failed_signatures, failed_family_keys
            )
            if failed_match:
                gate_decision = self._adaptive_gate_decision(
                    gate="brainstorm_failed_tactic",
                    match_reason=failed_match,
                    family_aliases=family_aliases,
                    tactic_text=block,
                )
                if gate_decision["mode"] != "hard":
                    self.state.notes.append(
                        "Adaptive gate allowed brainstorm tactic "
                        f"axis={axis} mode={gate_decision['mode']} "
                        f"reason={gate_decision['reason']}"
                    )
                    self._persist_gate_decision(gate_decision)
                    selected = {
                        "strategy_axis": axis,
                        "family_key": family_key,
                        "novelty_lane": self._tactic_novelty_lane(block),
                        "spec_task_id": self._explicit_spec_task_id(block),
                        "text": block.strip(),
                    }
                    record = {
                        "axis": axis,
                        "declared_axis": declared_axis,
                        "family_key": family_key,
                        "family_aliases": family_aliases,
                        "selected": False,
                        "skipped": False,
                        "reason": failed_match,
                        "gate_mode": gate_decision["mode"],
                        "gate_reason": gate_decision["reason"],
                    }
                    if axis_normalized_from:
                        record["axis_normalized_from"] = axis_normalized_from
                    if axis_source:
                        record["axis_source"] = axis_source
                    score, reasons = self._score_brainstorm_tactic(
                        block,
                        axis,
                        family_key,
                        family_aliases,
                        order,
                        explicit_family_key=bool(explicit_family_key),
                    )
                    record["score"] = score
                    record["score_reasons"] = reasons
                    selection_records.append(record)
                    selectable.append((score, order, selected, record))
                    continue
                selection_records.append(
                    {
                        "axis": axis,
                        "declared_axis": declared_axis,
                        "family_key": family_key,
                        "family_aliases": family_aliases,
                        "selected": False,
                        "skipped": True,
                        "reason": failed_match,
                        "gate_mode": gate_decision["mode"],
                        "gate_reason": gate_decision["reason"],
                    }
                )
                if axis_source:
                    selection_records[-1]["axis_source"] = axis_source
                self._persist_gate_decision(gate_decision)
                self.state.notes.append(
                    "Skipped brainstorm tactic "
                    f"axis={axis} reason={failed_match}"
                )
                continue
            selected = {
                "strategy_axis": axis,
                "family_key": family_key,
                "novelty_lane": self._tactic_novelty_lane(block),
                "spec_task_id": self._explicit_spec_task_id(block),
                "text": block.strip(),
            }
            score, reasons = self._score_brainstorm_tactic(
                block,
                axis,
                family_key,
                family_aliases,
                order,
                explicit_family_key=bool(explicit_family_key),
            )
            record = {
                "axis": axis,
                "declared_axis": declared_axis,
                "family_key": family_key,
                "family_aliases": family_aliases,
                "selected": False,
                "skipped": False,
                "reason": "",
                "score": score,
                "score_reasons": reasons,
            }
            if axis_normalized_from:
                record["axis_normalized_from"] = axis_normalized_from
            if axis_source:
                record["axis_source"] = axis_source
            selection_records.append(record)
            selectable.append((score, order, selected, record))
        if selectable:
            _score, _order, selected, selected_record = max(
                selectable, key=lambda item: (item[0], -item[1])
            )
            selected_record["selected"] = True
            self.state.scratch["brainstorm_selection"] = selection_records
            return selected
        self.state.scratch["brainstorm_selection"] = selection_records
        if selection_records and all(record.get("skipped") for record in selection_records):
            self.state.scratch["brainstorm_all_tactics_failed_loop"] = self.state.loop_count
            self.state.notes.append("All brainstorm tactics matched failed families")
        return None

    def _brainstorm_axis_family_mismatch_reject_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("brainstorm_reject_axis_family_mismatch", True))

    def _extract_tactic_axis(self, text: str, known_axes: set[str]) -> tuple[str, str]:
        return extract_tactic_axis(text, known_axes)

    def _family_key_strategy_axes(self, family_key: str) -> list[str]:
        normalized_axis = self._normalize_strategy_axis(family_key)
        if not normalized_axis:
            return []
        axes = (
            self._strategy_axis_pool()
            if self._strict_strategy_axis_pool_enabled()
            else self._known_strategy_axes()
        )
        if normalized_axis in axes:
            return [normalized_axis]
        return []

    def _canonical_axis_from_family_key(
        self, family_key: str, known_axes: set[str] | None = None
    ) -> str:
        known = known_axes or set(self._known_strategy_axes())
        axes = sorted({axis for axis in self._family_key_strategy_axes(family_key) if axis in known})
        return axes[0] if len(axes) == 1 else ""

    @staticmethod
    def _explicit_tactic_family_key(text: str) -> str:
        return explicit_tactic_family_key(text)

    @staticmethod
    def _explicit_spec_task_id(text: str) -> str:
        match = re.search(
            r"spec[\s_*.-]*task[\s_*.-]*id\s*:\s*`?([A-Za-z0-9_.-]+)`?",
            text,
            flags=re.IGNORECASE,
        )
        return match.group(1).strip() if match else ""

    def _score_brainstorm_tactic(
        self,
        tactic_text: str,
        axis: str,
        family_key: str,
        family_aliases: list[str],
        order: int,
        explicit_family_key: bool = False,
    ) -> tuple[float, list[str]]:
        workflow = self.config.get("workflow", {})
        if not workflow.get("brainstorm_score_tactics", True):
            return float(-order), ["original_order"]
        score = float(-order) * 0.01
        reasons = ["original_order"]
        validated = self._recent_validated_pattern_aliases()
        alias_set = {
            self._normalize_strategy_axis(str(alias))
            for alias in family_aliases
            if self._normalize_strategy_axis(str(alias))
        }
        if axis:
            alias_set.add(axis)
        normalized_family = self._normalize_strategy_axis(family_key)
        if normalized_family:
            alias_set.add(normalized_family)
        if alias_set & validated:
            score += 80.0
            reasons.append("extends_recent_validated_pattern")
        if explicit_family_key:
            score += 5.0
            reasons.append("has_family_key")
        if self._tactic_novelty_lane(tactic_text):
            score += 3.0
            reasons.append("has_novelty_lane")
        if re.search(r"\bhook\s*:", tactic_text, re.IGNORECASE):
            score += 4.0
            reasons.append("has_hook")
        if "`" in tactic_text:
            score += 2.0
            reasons.append("references_concrete_symbol")
        recent_patch_failures = self._recent_patch_failure_aliases()
        if alias_set & recent_patch_failures:
            score -= 10.0
            reasons.append("recent_patch_application_failures")
        return score, reasons

    def _recent_validated_pattern_aliases(self) -> set[str]:
        path = self._workflow_artifact_path(
            "validated_patterns_path", ".local_micro_agent/validated_patterns.jsonl"
        )
        if not path.exists():
            return set()
        limit = int(self.config.get("workflow", {}).get("validated_pattern_score_limit", 6) or 6)
        aliases: set[str] = set()
        for line in path.read_text(errors="replace").splitlines()[-limit:]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for value in (
                record.get("strategy_axis"),
                record.get("family_key"),
            ):
                normalized = self._normalize_strategy_axis(str(value or ""))
                if normalized:
                    aliases.add(normalized)
            last_attempt = record.get("last_attempt")
            if isinstance(last_attempt, dict):
                for raw_axis in last_attempt.get("strategy_axes", []) or []:
                    normalized = self._normalize_strategy_axis(str(raw_axis))
                    if normalized:
                        aliases.add(normalized)
        return aliases

    def _recent_patch_failure_aliases(self) -> set[str]:
        records = self._candidate_history_records(
            limit=int(self.config.get("workflow", {}).get("patch_failure_score_window", 8) or 8)
        )
        aliases: set[str] = set()
        for record in records:
            if not self._is_patch_application_failure_record(record):
                continue
            for value in (
                record.get("strategy_axis"),
                *(record.get("strategy_axes") or []),
                *(record.get("family_aliases") or []),
            ):
                normalized = self._normalize_strategy_axis(str(value or ""))
                if normalized:
                    aliases.add(normalized)
        return aliases

    def _failed_tactic_signatures(self) -> list[set[str]]:
        path = self._workflow_artifact_path(
            "failed_tactics_path", ".local_micro_agent/failed_tactics.jsonl"
        )
        if not path.exists():
            return []
        limit = int(self.config.get("workflow", {}).get("failed_tactic_signature_limit", 16) or 16)
        signatures: list[set[str]] = []
        for line in path.read_text(errors="replace").splitlines()[-limit:]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = "\n".join(
                str(part)
                for part in (
                    record.get("context", ""),
                    (record.get("last_attempt") or {}).get("reason", "")
                    if isinstance(record.get("last_attempt"), dict)
                    else "",
                )
            )
            signature = self._tactic_signature(text)
            if signature:
                signatures.append(signature)
        return signatures

    def _failed_tactic_family_keys(self) -> set[str]:
        path = self._workflow_artifact_path(
            "failed_tactics_path", ".local_micro_agent/failed_tactics.jsonl"
        )
        if not path.exists():
            return set()
        limit = int(
            self.config.get("workflow", {}).get("failed_tactic_family_limit", 24) or 24
        )
        family_keys: set[str] = set()
        for line in path.read_text(errors="replace").splitlines()[-limit:]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            context = "\n".join(
                str(part)
                for part in (
                    record.get("context", ""),
                    (record.get("last_attempt") or {}).get("reason", "")
                    if isinstance(record.get("last_attempt"), dict)
                    else "",
                )
            )
            family_key = str(record.get("family_key", "")).strip()
            if family_key:
                family_keys.add(family_key)
                family_keys.add(self._normalize_strategy_axis(family_key))
            family_keys.update(self._tactic_family_aliases(context))
            axis = self._normalize_strategy_axis(str(record.get("strategy_axis", "")))
            if axis:
                family_keys.add(axis)
            last_attempt = record.get("last_attempt")
            if isinstance(last_attempt, dict):
                last_axis = self._normalize_strategy_axis(
                    str(last_attempt.get("strategy_axis", ""))
                )
                if last_axis:
                    family_keys.add(last_axis)
                for raw_axis in last_attempt.get("strategy_axes", []) or []:
                    normalized_axis = self._normalize_strategy_axis(str(raw_axis))
                    if normalized_axis:
                        family_keys.add(normalized_axis)
        return family_keys

    def _failed_tactic_match_reason(
        self,
        tactic_text: str,
        failed_signatures: list[set[str]],
        failed_family_keys: set[str],
    ) -> str:
        candidate_family_aliases = self._tactic_family_aliases(tactic_text, include_axes=False)
        if not candidate_family_aliases:
            candidate_family_aliases = self._tactic_family_aliases(tactic_text)
        family_matches = sorted(candidate_family_aliases & failed_family_keys)
        if family_matches:
            return "failed_family=" + ",".join(family_matches)
        threshold = float(
            self.config.get("workflow", {}).get("failed_tactic_similarity_threshold", 0.45)
        )
        candidate_signature = self._tactic_signature(tactic_text)
        if not candidate_signature:
            return ""
        for failed_signature in failed_signatures:
            similarity = self._signature_similarity(candidate_signature, failed_signature)
            if similarity >= threshold:
                return f"signature_similarity={similarity:.2f}"
        return ""

    def _adaptive_gate_controller_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("adaptive_gate_controller")) and (
            self._adaptive_search_memory_enabled()
        )

    def _adaptive_gate_decision(
        self,
        gate: str,
        match_reason: str,
        family_aliases: list[str] | set[str],
        tactic_text: str = "",
    ) -> dict[str, Any]:
        aliases = sorted(
            {
                self._normalize_strategy_axis(str(alias))
                for alias in family_aliases
                if self._normalize_strategy_axis(str(alias))
            }
        )
        decision = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "loop": self.state.loop_count,
            "gate": gate,
            "mode": "hard",
            "reason": "legacy_hard_gate",
            "match_reason": match_reason,
            "family_aliases": aliases,
            "all_skipped_streak": self._brainstorm_all_skipped_streak(),
            "family_evidence": self._failed_family_evidence(aliases),
        }
        if tactic_text:
            decision["tactic_signature"] = sorted(self._tactic_signature(tactic_text))[:12]
        if not self._adaptive_gate_controller_enabled():
            return decision

        workflow = self.config.get("workflow", {})
        relax_streak = int(workflow.get("adaptive_gate_all_skipped_relax_streak", 2) or 0)
        if relax_streak > 0 and decision["all_skipped_streak"] >= relax_streak:
            decision["mode"] = "soft"
            decision["reason"] = "opportunity_pressure_all_skipped"
            return decision

        min_attempts = int(workflow.get("adaptive_gate_min_family_attempts_for_hard", 2) or 0)
        family_evidence = decision["family_evidence"]
        max_attempts = max(
            (int(item.get("attempts", 0) or 0) for item in family_evidence.values()),
            default=0,
        )
        if min_attempts > 0 and max_attempts < min_attempts:
            decision["mode"] = "shadow"
            decision["reason"] = "insufficient_failed_family_evidence"
            return decision

        decision["reason"] = "evidence_supported_hard_gate"
        return decision

    def _failed_family_evidence(self, aliases: list[str]) -> dict[str, dict[str, Any]]:
        wanted = {
            self._normalize_strategy_axis(alias)
            for alias in aliases
            if self._normalize_strategy_axis(alias)
        }
        if not wanted:
            return {}
        path = self._workflow_artifact_path(
            "failed_tactics_path", ".local_micro_agent/failed_tactics.jsonl"
        )
        evidence = {
            alias: {"attempts": 0, "last_status": None, "last_metric": None}
            for alias in sorted(wanted)
        }
        if not path.exists():
            return evidence
        for line in path.read_text(errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_aliases = {
                self._normalize_strategy_axis(str(record.get("family_key", ""))),
                self._normalize_strategy_axis(str(record.get("strategy_axis", ""))),
            }
            record_aliases.update(
                self._tactic_family_aliases(
                    "\n".join(
                        str(part)
                        for part in (
                            record.get("context", ""),
                            (record.get("last_attempt") or {}).get("reason", "")
                            if isinstance(record.get("last_attempt"), dict)
                            else "",
                        )
                    )
                )
            )
            attempts = int(record.get("attempts", 0) or 0)
            if attempts <= 0:
                attempts = 1
            for alias in wanted & {item for item in record_aliases if item}:
                item = evidence.setdefault(
                    alias, {"attempts": 0, "last_status": None, "last_metric": None}
                )
                item["attempts"] = int(item.get("attempts", 0) or 0) + attempts
                item["last_status"] = record.get("status")
                last_attempt = record.get("last_attempt")
                if isinstance(last_attempt, dict):
                    item["last_metric"] = last_attempt.get("metric")
        return evidence

    def _persist_gate_decision(self, decision: dict[str, Any]) -> None:
        if not self._adaptive_gate_controller_enabled():
            return
        path = self._workflow_artifact_path(
            "adaptive_gate_decisions_path", ".local_micro_agent/gate_decisions.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _tactic_signature(text: str) -> set[str]:
        return tactic_signature(text)

    @staticmethod
    def _signature_similarity(left: set[str], right: set[str]) -> float:
        return signature_similarity(left, right)

    def _tactic_family_key(self, text: str) -> str:
        return explicit_tactic_family_key(text)

    @staticmethod
    def _tactic_novelty_lane(text: str) -> str:
        return tactic_novelty_lane(text)

    def _tactic_family_aliases(self, text: str, include_axes: bool = True) -> set[str]:
        aliases: set[str] = set()
        family_key = self._tactic_family_key(text)
        if family_key:
            aliases.add(family_key)
            aliases.add(self._normalize_strategy_axis(family_key))
            aliases.update(self._family_key_strategy_axes(family_key))
        if not include_axes:
            return aliases
        for axis in re.findall(
            r"strategy[\s_*.-]*axis[\s*]*:\s*[*\s]*`?([a-zA-Z0-9_-]+)`?",
            text,
            flags=re.IGNORECASE,
        ):
            normalized_axis = self._normalize_strategy_axis(axis)
            if normalized_axis:
                aliases.add(normalized_axis)
        return aliases

    def _persist_brainstorm_tactics(self, brainstorm: str, reject_summary: str) -> None:
        path = self._workflow_artifact_path(
            "brainstorm_tactics_path", ".local_micro_agent/brainstorm_tactics.md"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        record = (
            f"# Brainstorm Tactics\n\n"
            f"- ts: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
            f"- loop: {self.state.loop_count}\n"
            f"- best_metric: {self.state.scratch.get('best_metric', self.state.scratch.get('last_metric'))}\n\n"
            f"## Tactics\n\n{brainstorm}\n\n"
            f"## Recent Reject Summary\n\n```json\n{reject_summary}\n```\n\n"
        )
        with path.open("a") as handle:
            handle.write(record)
        self.state.notes.append(f"Persisted brainstorm tactics: {path}")

    def _persist_brainstorm_selection(self) -> None:
        records = self.state.scratch.get("brainstorm_selection")
        if not isinstance(records, list):
            return
        path = self._workflow_artifact_path(
            "brainstorm_selection_path", ".local_micro_agent/brainstorm_selection.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        selected = next(
            (record for record in records if isinstance(record, dict) and record.get("selected")),
            None,
        )
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "loop": self.state.loop_count,
            "selected": selected,
            "all_skipped": bool(records)
            and all(
                isinstance(record, dict) and bool(record.get("skipped"))
                for record in records
            ),
            "records": records,
        }
        with path.open("a") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        self.state.notes.append(f"Persisted brainstorm selection: {path}")

    def _brainstorm_new_family_required(self) -> bool:
        threshold = int(
            self.config.get("workflow", {}).get("brainstorm_new_family_after_all_skipped", 2)
            or 0
        )
        return threshold > 0 and self._brainstorm_all_skipped_streak() >= threshold

    def _brainstorm_open_novelty_lanes_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("brainstorm_include_open_novelty_lanes", True))

    def _open_novelty_lanes(self) -> list[str]:
        workflow = self.config.get("workflow", {})
        configured = workflow.get("brainstorm_open_novelty_lanes")
        if isinstance(configured, list) and configured:
            lanes = [str(item).strip() for item in configured if str(item).strip()]
            if lanes:
                return lanes
        return [
            "behavior_boundary_probe: test one edge-case boundary or invariant with a narrow local edit",
            "data_flow_simplification: remove redundant transformation, copy, lookup, or conversion work",
            "state_lifecycle_adjustment: change cache/state initialization, reuse, invalidation, or persistence locally",
            "error_recovery_path: improve one concrete failure, timeout, retry, or exception path",
            "api_contract_alignment: make one interface, schema, signature, or caller/callee expectation consistent",
            "performance_hot_path_reduction: reduce repeated work in a measured hot path without a broad rewrite",
            "test_signal_expansion: add or adjust a focused validation signal when tests are allowed",
            "resource_or_concurrency_control: narrow one file, process, async, memory, or lifecycle control issue",
        ]

    def _brainstorm_all_skipped_streak(self) -> int:
        path = self._workflow_artifact_path(
            "brainstorm_selection_path", ".local_micro_agent/brainstorm_selection.jsonl"
        )
        if not path.exists():
            return 0
        streak = 0
        for line in reversed(path.read_text(errors="replace").splitlines()):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break
            if record.get("all_skipped") is True:
                streak += 1
                continue
            break
        return streak

    def _forbidden_tactic_family_aliases(self) -> list[str]:
        aliases = set(self._failed_tactic_family_keys())
        aliases.update(self._skipped_brainstorm_family_aliases())
        return sorted(aliases)

    def _skipped_brainstorm_family_aliases(self) -> set[str]:
        path = self._workflow_artifact_path(
            "brainstorm_selection_path", ".local_micro_agent/brainstorm_selection.jsonl"
        )
        if not path.exists():
            return set()
        limit = int(
            self.config.get("workflow", {}).get("brainstorm_forbidden_selection_limit", 24)
            or 24
        )
        aliases: set[str] = set()
        for line in path.read_text(errors="replace").splitlines()[-limit:]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for item in record.get("records", []) or []:
                if not isinstance(item, dict) or not item.get("skipped"):
                    continue
                for alias in item.get("family_aliases", []) or []:
                    normalized = self._normalize_strategy_axis(str(alias))
                    if normalized:
                        aliases.add(normalized)
        return aliases

    def _create_active_todo_from_selected_tactic(self, selected_tactic: dict[str, str]) -> None:
        axis = str(selected_tactic.get("strategy_axis", "general_edit"))
        todo_id = f"todo-{self.state.loop_count:03d}-{axis}"
        tactic_text = selected_tactic.get("text", "")
        tactic_stage = self._tactic_stage_for_text(tactic_text)
        structural = tactic_stage.startswith("structural_")
        spec_task_id = self._spec_task_id_for_tactic(selected_tactic)
        todo = {
            "todo_id": todo_id,
            "parent_tactic_id": f"brainstorm-loop-{self.state.loop_count}",
            "spec_task_id": spec_task_id,
            "status": "active",
            "strategy_axis": axis,
            "family_key": self._tactic_family_key(tactic_text),
            "tactic_stage": tactic_stage,
            "title": f"Feasibility probe for {axis}",
            "context": tactic_text,
            "micro_goal": (
                "Implement the smallest correctness-preserving structural scaffold/probe "
                "for this tactic. Prefer preserving current behavior over improving the "
                "metric in this step."
                if structural
                else "Implement the smallest correctness-preserving feasibility probe for "
                "this tactic. Do not attempt the full architecture migration in one patch."
            ),
            "implementation_hint": (
                "Start with a guarded or behavior-preserving scaffold. If enabling behavior, "
                "scope it to the smallest safe slice and keep an easy rollback path."
                if structural
                else "Prefer one narrow edit that proves the tactic changes real behavior "
                "before expanding it."
            ),
            "allowed_files": sorted(self._writable_files()),
            "forbidden_patterns": [
                "broad rewrite unrelated to the selected tactic",
                "changing tests or fixtures unless explicitly allowed",
                "mixing multiple independent tactics",
            ],
            "expected_signal": (
                "Tests still pass, configured metrics remain parseable, and the artifact "
                "shows whether this was a scaffold, guarded probe, or expansion. Metric "
                "improvement is optional before the expansion stage."
                if structural
                else "Tests still pass, and any configured metric remains parseable. A metric "
                "improvement is welcome but not required for the first feasibility probe."
            ),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "created_loop": self.state.loop_count,
        }
        self.state.scratch["active_todo"] = todo
        self._persist_todo_plan(todo)

    def _spec_task_id_for_tactic(self, selected_tactic: dict[str, str]) -> str:
        explicit = str(selected_tactic.get("spec_task_id", "") or "").strip()
        if explicit:
            return explicit
        spec = self.state.scratch.get("run_spec")
        if not isinstance(spec, dict):
            return ""
        tasks = spec.get("task_graph")
        if not isinstance(tasks, list):
            return ""
        axis = self._normalize_strategy_axis(str(selected_tactic.get("strategy_axis", "")))
        family = self._normalize_strategy_axis(str(selected_tactic.get("family_key", "")))
        fallback = ""
        for task in tasks:
            if not isinstance(task, dict):
                continue
            if str(task.get("status", "open")) in {"validated", "retired", "failed"}:
                continue
            task_axis = self._normalize_strategy_axis(str(task.get("strategy_axis", "")))
            task_family = self._normalize_strategy_axis(str(task.get("family_key", "")))
            if family and task_family and family == task_family:
                return str(task.get("task_id", ""))
            if axis and task_axis == axis and not fallback:
                fallback = str(task.get("task_id", ""))
        return fallback

    def _tactic_stage_for_text(self, text: str) -> str:
        if not self.config.get("workflow", {}).get("structural_tactic_lifecycle", True):
            return "local_edit"
        normalized = self._normalize_fingerprint_text(text)
        if re.search(r"\b(scaffold|wrapper|shim|adapter)\b", normalized):
            return "structural_scaffold"
        if re.search(r"\b(expand|broaden|roll out|generalize)\b", normalized):
            return "structural_expand"
        structural_patterns = (
            r"\b(rewrite|refactor|scheduler|scheduling|pipeline|lifecycle)\b",
            r"\b(state machine|migration|parser|cache layer|architecture)\b",
            r"\b(vector|unroll|packing|parallel|interleave|tiling)\b",
        )
        if any(re.search(pattern, normalized) for pattern in structural_patterns):
            return "structural_probe"
        return "local_edit"

    def _persist_todo_plan(self, todo: dict[str, Any]) -> None:
        plan_path = self._workflow_artifact_path(
            "todo_plan_path", ".local_micro_agent/todo_plan.json"
        )
        active_path = self._workflow_artifact_path(
            "active_todo_path", ".local_micro_agent/active_todo.json"
        )
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.parent.mkdir(parents=True, exist_ok=True)
        plan = self._load_todo_plan(plan_path)
        previous_active_id = plan.get("active_todo_id")
        todos = plan.setdefault("todos", [])
        if not isinstance(todos, list):
            todos = []
            plan["todos"] = todos
        if previous_active_id and previous_active_id != todo.get("todo_id"):
            for existing in todos:
                if (
                    isinstance(existing, dict)
                    and existing.get("todo_id") == previous_active_id
                    and existing.get("status") in {"active", "attempted"}
                ):
                    existing["status"] = "superseded"
                    existing["superseded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                    break
        replaced = False
        for index, existing in enumerate(todos):
            if isinstance(existing, dict) and existing.get("todo_id") == todo.get("todo_id"):
                todos[index] = {**existing, **todo}
                replaced = True
                break
        if not replaced:
            todos.append(todo)
        plan = {
            **plan,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "active_todo_id": todo.get("todo_id"),
            "todos": todos,
        }
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
        active_path.write_text(json.dumps(todo, ensure_ascii=False, indent=2) + "\n")
        self.state.notes.append(f"Persisted active todo: {active_path}")

    @staticmethod
    def _load_todo_plan(plan_path: Path) -> dict[str, Any]:
        if not plan_path.exists():
            return {"version": 1, "todos": []}
        try:
            plan = json.loads(plan_path.read_text(errors="replace"))
        except json.JSONDecodeError:
            return {"version": 1, "todos": []}
        if not isinstance(plan, dict):
            return {"version": 1, "todos": []}
        plan.setdefault("version", 1)
        plan.setdefault("todos", [])
        return plan

