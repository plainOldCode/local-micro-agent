"""Adaptive search memory: axes, regions, cooldowns, contracts, failure memory.

Extracted from orchestrator.py; mixed into MicroAgent.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from ..decisions import CodeCandidate
from ..state import CodeChange
from ..strategy import (
    DEFAULT_STRATEGY_AXIS_GUIDANCE,
    DEFAULT_STRATEGY_AXIS_KEYWORDS,
    axis_label_matches_text,
    keyword_phrase_matches,
    keyword_token_matches,
    normalize_fingerprint_text,
    normalize_strategy_axis,
    strategy_axes_for_text,
)


class AdaptiveSearchMixin:
    def _candidate_novelty_gate_enabled(self) -> bool:
        return bool(self.config.get("workflow", {}).get("candidate_novelty_gate"))

    def _adaptive_search_memory_enabled(self) -> bool:
        return bool(self.config.get("workflow", {}).get("adaptive_search_memory"))

    def _adaptive_search_reject_cooled_axes_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("adaptive_search_reject_cooled_axes")) and (
            self._adaptive_search_memory_enabled()
        )

    def _adaptive_search_reject_cooled_regions_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return (
            workflow.get("adaptive_search_reject_cooled_regions", True) is not False
            and self._adaptive_search_memory_enabled()
            and self._adaptive_search_reject_cooled_axes_enabled()
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
        if (
            declared
            and (
                not self._strict_strategy_axis_pool_enabled()
                or declared in self._strategy_axis_pool()
            )
            and declared not in axes
        ):
            axes.append(declared)
        if not axes:
            axes = ["general_edit"]
        return sorted(set(axes))

    def _candidate_reason_strategy_axes(self, candidate: CodeCandidate) -> list[str]:
        declared = self._normalize_strategy_axis(candidate.strategy_axis)
        reason_parts = [candidate.reason, *(change.reason for change in candidate.changes)]
        reason_text = self._normalize_fingerprint_text("\n".join(reason_parts))
        axes = self._strategy_axes_for_text(reason_text, self._strategy_axis_keywords())
        if declared and axis_label_matches_text(declared, reason_text) and declared not in axes:
            axes.append(declared)
        return axes or ["general_edit"]

    def _strategy_axis_keywords(self) -> dict[str, tuple[str, ...]]:
        workflow = self.config.get("workflow", {})
        configured = workflow.get("adaptive_search_axis_keywords", {})
        keyword_axes: dict[str, tuple[str, ...]] = {}
        if isinstance(configured, dict):
            for raw_axis, raw_keywords in configured.items():
                axis = self._normalize_strategy_axis(str(raw_axis))
                if not axis:
                    continue
                if isinstance(raw_keywords, str):
                    keywords = [raw_keywords]
                elif isinstance(raw_keywords, list):
                    keywords = [str(keyword) for keyword in raw_keywords if str(keyword)]
                else:
                    keywords = []
                if keywords:
                    keyword_axes[axis] = tuple(keywords)
        configured_pool = workflow.get("adaptive_search_axis_pool")
        if isinstance(configured_pool, list) and configured_pool:
            for raw_axis in configured_pool:
                axis = self._normalize_strategy_axis(str(raw_axis))
                if not axis or axis in keyword_axes:
                    continue
                axis_tokens = [token for token in axis.split("_") if len(token) >= 4]
                label = axis.replace("_", " ")
                keyword_axes[axis] = tuple([label, *axis_tokens])
        if keyword_axes:
            return keyword_axes
        return DEFAULT_STRATEGY_AXIS_KEYWORDS

    @staticmethod
    def _strategy_axes_for_text(
        text: str, keyword_axes: dict[str, tuple[str, ...]]
    ) -> list[str]:
        return strategy_axes_for_text(text, keyword_axes)

    @staticmethod
    def _keyword_phrase_matches(
        tokens: set[str], keyword: str, allow_variants: bool = True
    ) -> bool:
        return keyword_phrase_matches(tokens, keyword, allow_variants=allow_variants)

    @staticmethod
    def _keyword_token_matches(
        tokens: set[str], keyword_token: str, allow_variants: bool = True
    ) -> bool:
        return keyword_token_matches(tokens, keyword_token, allow_variants=allow_variants)

    def _format_axis_contract(self) -> str:
        if not self._axis_contract_enabled():
            self.state.scratch.pop("required_strategy_axis", None)
            return ""
        required_axis = self._select_required_strategy_axis()
        self.state.scratch["required_strategy_axis"] = required_axis
        cooled_axes = self._current_cooled_axes()
        payload = {
            "required_strategy_axis": required_axis,
            "required_family_key": self._selected_tactic_family_for_current_loop(),
            "allowed_strategy_axes": self._allowed_strategy_axes(),
            "cooled_strategy_axes": cooled_axes,
            "known_strategy_axes": self._brainstorm_known_axes(),
            "required_axis_guidance": self._strategy_axis_guidance(required_axis),
            "selected_tactic": self._selected_tactic_for_current_loop(),
            "output_requirement": (
                "Set candidate strategy_axis exactly to required_strategy_axis. "
                "In XML mode include <strategy_axis>axis</strategy_axis> inside each "
                "<candidate>. Candidate reason and change reasons must substantively "
                "target required_strategy_axis. If required_family_key is set, candidate "
                "reason and change reasons must stay on that selected tactic family and "
                "must not re-label a forbidden family under the selected axis. Drift is "
                "rejected before changes are applied."
            ),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _strategy_axis_guidance(self, axis: str) -> dict[str, Any]:
        workflow = self.config.get("workflow", {})
        configured = workflow.get("adaptive_search_axis_guidance", {})
        normalized_axis = self._normalize_strategy_axis(axis)
        if isinstance(configured, dict):
            raw_guidance = configured.get(normalized_axis) or configured.get(axis)
            if isinstance(raw_guidance, dict):
                return raw_guidance
            if isinstance(raw_guidance, str) and raw_guidance.strip():
                return {
                    "focus": raw_guidance.strip(),
                    "try": ["choose one small concrete tactic for this axis"],
                    "avoid_drift": ["renaming another strategy as this axis"],
                }
        return DEFAULT_STRATEGY_AXIS_GUIDANCE.get(
            normalized_axis,
            {
                "focus": f"Make a candidate centered on {normalized_axis or axis}.",
                "try": ["choose one small concrete tactic for this axis"],
                "avoid_drift": ["renaming another strategy as this axis"],
            },
        )

    def _selected_tactic_for_current_loop(self) -> dict[str, Any]:
        selected_tactic = self.state.scratch.get("selected_tactic")
        if not isinstance(selected_tactic, dict):
            return {}
        if self.state.scratch.get("selected_tactic_loop") != self.state.loop_count:
            return {}
        axis = self._normalize_strategy_axis(str(selected_tactic.get("strategy_axis", "")))
        if (
            self._strict_strategy_axis_pool_enabled()
            and axis not in self._strategy_axis_pool()
        ):
            return {}
        return selected_tactic

    def _axis_contract_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("adaptive_search_force_strategy_axis")) and (
            self._adaptive_search_memory_enabled()
        )

    def _strict_strategy_axis_pool_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("adaptive_search_strict_axis_pool", False))

    def _strategy_axis_pool(self) -> list[str]:
        workflow = self.config.get("workflow", {})
        configured = workflow.get("adaptive_search_axis_pool")
        if isinstance(configured, list) and configured:
            return [self._normalize_strategy_axis(str(axis)) for axis in configured if str(axis)]
        axes = list(self._strategy_axis_keywords().keys())
        if "general_edit" not in axes:
            axes.append("general_edit")
        return axes

    def _known_strategy_axes(self) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for axis in [*self._strategy_axis_pool(), *sorted(self._observed_strategy_axes())]:
            normalized = self._normalize_strategy_axis(str(axis))
            if normalized and normalized not in seen:
                ordered.append(normalized)
                seen.add(normalized)
        if "general_edit" not in seen:
            ordered.append("general_edit")
        return ordered

    def _observed_strategy_axes(self) -> set[str]:
        axes: set[str] = set()
        selected_tactic = self.state.scratch.get("selected_tactic")
        if isinstance(selected_tactic, dict):
            axis = self._normalize_strategy_axis(str(selected_tactic.get("strategy_axis", "")))
            if axis:
                axes.add(axis)
        active_todo = self.state.scratch.get("active_todo")
        if isinstance(active_todo, dict):
            axis = self._normalize_strategy_axis(str(active_todo.get("strategy_axis", "")))
            if axis:
                axes.add(axis)
        memory = self.state.scratch.get("adaptive_search_memory")
        if isinstance(memory, dict) and isinstance(memory.get("axes"), dict):
            axes.update(
                self._normalize_strategy_axis(str(axis))
                for axis in memory["axes"]
                if self._normalize_strategy_axis(str(axis))
            )
        return axes

    def _brainstorm_known_axes(self) -> list[str]:
        if self._strict_strategy_axis_pool_enabled():
            return self._strategy_axis_pool()
        return self._known_strategy_axes()

    def _allowed_strategy_axes(self) -> list[str]:
        cooled = set(self._current_cooled_axes())
        pool = (
            self._strategy_axis_pool()
            if self._strict_strategy_axis_pool_enabled()
            else self._known_strategy_axes()
        )
        return [axis for axis in pool if axis not in cooled]

    def _select_required_strategy_axis(self) -> str:
        allowed = self._allowed_strategy_axes()
        selected_tactic = self._selected_tactic_for_current_loop()
        known_axes = self._brainstorm_known_axes()
        if (
            selected_tactic
            and selected_tactic.get("strategy_axis") in known_axes
        ):
            return str(selected_tactic["strategy_axis"])
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
        strict_axis_pool = self._strict_strategy_axis_pool_enabled()
        axis_contract = self._axis_contract_enabled()
        declared = self._normalize_strategy_axis(candidate.strategy_axis)
        if strict_axis_pool and declared and declared not in self._strategy_axis_pool():
            return ("rejected_unknown_axis", f"unknown strategy_axis {declared}")
        if not axis_contract:
            return None
        if not declared:
            return ("rejected_missing_axis", "missing strategy_axis")
        required = self.state.scratch.get("required_strategy_axis")
        if isinstance(required, str) and required and declared != required:
            return (
                "rejected_wrong_axis",
                f"strategy_axis {declared} does not match required {required}",
            )
        if isinstance(required, str) and required:
            candidate_axes = self._candidate_strategy_axes(candidate)
            if required == "general_edit":
                if candidate_axes != ["general_edit"]:
                    return (
                        "rejected_axis_drift",
                        "candidate reason targets "
                        f"{', '.join(candidate_axes)} instead of required general_edit",
                    )
            elif required not in candidate_axes:
                return (
                    "rejected_axis_drift",
                    "candidate does not substantively target required "
                    f"strategy_axis {required}",
                )
        if declared in self._current_cooled_axes():
            if self._selected_tactic_axis_for_current_loop() == declared:
                return None
            return ("rejected_cooled_axis", f"cooled strategy_axis {declared}")
        return None

    def _active_todo_contract_rejection(
        self, candidate: CodeCandidate
    ) -> tuple[str, str] | None:
        workflow = self.config.get("workflow", {})
        if workflow.get("todo_enforce_active_contract", True) is False:
            return None
        if self._todo_contract_soft_now():
            return None
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

        todo_id = str(active_todo.get("todo_id", ""))
        required_axis = self._normalize_strategy_axis(
            str(active_todo.get("strategy_axis", ""))
        )
        declared_axis = self._normalize_strategy_axis(candidate.strategy_axis)
        if required_axis and declared_axis != required_axis:
            return (
                "rejected_todo_axis_drift",
                f"strategy_axis {declared_axis or '<missing>'} does not match "
                f"active todo {todo_id} axis {required_axis}",
            )

        if required_axis:
            candidate_axes = self._candidate_strategy_axes(candidate)
            if required_axis not in candidate_axes:
                return (
                    "rejected_todo_axis_drift",
                    "candidate does not substantively target active todo "
                    f"{todo_id} axis {required_axis}",
                )

        required_family = self._normalize_strategy_axis(
            str(active_todo.get("family_key", ""))
        )
        if required_family:
            candidate_families = self._candidate_reason_family_aliases(candidate)
            if candidate_families and required_family not in candidate_families:
                return (
                    "rejected_todo_family_drift",
                    "candidate reason targets family "
                    f"{', '.join(sorted(candidate_families))} instead of active todo "
                    f"{todo_id} family_key {required_family}",
                )
        scope_rejection = self._active_todo_change_scope_rejection(candidate.changes)
        if scope_rejection is not None:
            return scope_rejection
        return None

    def _active_todo_change_scope_rejection(
        self, changes: list[CodeChange]
    ) -> tuple[str, str] | None:
        workflow = self.config.get("workflow", {})
        if workflow.get("todo_enforce_active_change_scope", True) is False:
            return None
        if self._todo_contract_soft_now():
            return None
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

        symbols = [
            str(item).strip()
            for item in active_todo.get("target_symbols", [])
            if str(item).strip()
        ] if isinstance(active_todo.get("target_symbols"), list) else []
        regions = [
            str(item).strip()
            for item in active_todo.get("target_regions", [])
            if str(item).strip()
        ] if isinstance(active_todo.get("target_regions"), list) else []
        if not symbols and not regions:
            return None

        if self._active_todo_is_structural_probe(active_todo):
            max_changes = int(workflow.get("structural_probe_max_changes", 1) or 1)
            if max_changes > 0 and len(changes) > max_changes:
                return (
                    "rejected_todo_scope_drift",
                    "structural_probe must emit one small change at a time",
                )
            max_lines = int(workflow.get("structural_probe_max_changed_lines", 40) or 40)
            for change in changes:
                if self._change_too_broad_for_structural_probe(change, max_lines):
                    return (
                        "rejected_todo_scope_drift",
                        "structural_probe change is too broad; use a smaller "
                        "reversible probe inside the active target region",
                    )

        allowed_paths = {
            region.split("::", 1)[0]
            for region in regions
            if region.split("::", 1)[0]
        }
        allowed_files = active_todo.get("allowed_files")
        if isinstance(allowed_files, list):
            allowed_paths.update(
                str(path).strip() for path in allowed_files if str(path).strip()
            )

        todo_id = str(active_todo.get("todo_id", "active todo"))
        for change in changes:
            if allowed_paths and change.path not in allowed_paths:
                return (
                    "rejected_todo_scope_drift",
                    f"change path {change.path} is outside active todo {todo_id} "
                    f"target paths {', '.join(sorted(allowed_paths))}",
                )
            if symbols and not self._change_targets_any_symbol(change, symbols):
                return (
                    "rejected_todo_scope_drift",
                    f"change for {change.path} does not target active todo {todo_id} "
                    f"symbols {', '.join(symbols)}",
                )
        return None

    def _active_probe_diff_contract_rejection(
        self,
        previous_snapshot: dict[str, str | None],
        allowed: set[str],
    ) -> tuple[str, str, dict[str, Any]] | None:
        workflow = self.config.get("workflow", {})
        if workflow.get(
            "probe_diff_contract_gate",
            workflow.get("spec_structural_risk_gate", False),
        ) is False:
            return None
        if self._todo_contract_soft_now():
            return None
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
        if not self._active_todo_is_structural_probe(active_todo):
            return None

        contract = self._probe_diff_contract_for_todo(active_todo)
        summary = self._probe_diff_summary(previous_snapshot, allowed)
        violations = self._probe_diff_contract_violations(summary, contract)
        self.state.scratch["last_probe_diff_summary"] = summary
        if not violations:
            return None
        note = "probe diff contract mismatch: " + "; ".join(violations[:5])
        extra = {
            "diff_contract_violations": violations,
            "probe_diff_summary": self._compact_probe_diff_summary(summary),
            "probe_diff_contract": contract,
            "failure_origin": "post_apply_contract",
            "issue_scope": "candidate_delta",
            "repo_valid_after_restore": True,
            "repair_task_eligible": False,
            "memory_use": "avoid_shape",
            "candidate_delta_files_changed": summary.get("files_changed", 0),
        }
        return ("rejected_probe_contract_mismatch", note, extra)

    def _probe_diff_contract_for_todo(self, active_todo: dict[str, Any]) -> dict[str, Any]:
        workflow = self.config.get("workflow", {})
        raw = active_todo.get("probe_diff_contract")
        contract = raw if isinstance(raw, dict) else {}
        target_symbols = self._string_list_from_any(active_todo.get("target_symbols"))
        target_regions = self._string_list_from_any(active_todo.get("target_regions"))
        todo_files = self._string_list_from_any(active_todo.get("allowed_files"))
        region_files = [
            region.split("::", 1)[0]
            for region in target_regions
            if region.split("::", 1)[0]
        ]
        allowed_files = self._string_list_from_any(contract.get("allowed_files"))
        if not allowed_files:
            allowed_files = [*todo_files, *region_files]
        allowed_regions = self._string_list_from_any(contract.get("allowed_regions"))
        if not allowed_regions:
            allowed_regions = [*target_regions, *target_symbols]
        expected_changed_regions = self._string_list_from_any(
            contract.get("expected_changed_regions")
        )
        if not expected_changed_regions:
            expected_changed_regions = allowed_regions
        return {
            "allowed_files": sorted(dict.fromkeys(allowed_files)),
            "allowed_regions": sorted(dict.fromkeys(allowed_regions)),
            "expected_changed_regions": sorted(dict.fromkeys(expected_changed_regions)),
            "target_symbols": sorted(
                dict.fromkeys(
                    self._string_list_from_any(contract.get("target_symbols"))
                    or target_symbols
                )
            ),
            "forbidden_symbols": self._string_list_from_any(
                contract.get("forbidden_symbols")
            ),
            "forbidden_regions": self._string_list_from_any(
                contract.get("forbidden_regions")
            ),
            "required_unchanged_regions": self._string_list_from_any(
                contract.get("required_unchanged_regions")
            ),
            "allowed_change_kinds": self._string_list_from_any(
                contract.get("allowed_change_kinds")
            ),
            "observation": str(contract.get("observation") or "").strip(),
            "max_files_changed": self._positive_int(
                contract.get("max_files_changed"),
                int(workflow.get("structural_probe_max_files_changed", 1) or 1),
            ),
            "max_hunks": self._positive_int(
                contract.get("max_hunks"),
                int(workflow.get("structural_probe_max_hunks", 2) or 2),
            ),
            "max_changed_lines": self._positive_int(
                contract.get("max_changed_lines"),
                int(workflow.get("structural_probe_max_changed_lines", 40) or 40),
            ),
            "max_changed_functions": self._positive_int(
                contract.get("max_changed_functions"),
                int(workflow.get("structural_probe_max_changed_functions", 1) or 1),
            ),
        }

    @staticmethod
    def _positive_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(0, parsed)

    @staticmethod
    def _string_list_from_any(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _probe_diff_summary(
        self,
        previous_snapshot: dict[str, str | None],
        allowed: set[str],
    ) -> dict[str, Any]:
        paths = sorted(set(previous_snapshot) | set(allowed))
        file_summaries: list[dict[str, Any]] = []
        for rel_path in paths:
            old_content = previous_snapshot.get(rel_path)
            current_path = self.state.repo_root / rel_path
            try:
                new_content: str | None = current_path.read_text(errors="replace")
            except OSError:
                new_content = None
            if old_content == new_content:
                continue
            file_summary = self._probe_diff_file_summary(
                rel_path,
                old_content,
                new_content,
            )
            file_summaries.append(file_summary)
        touched_symbols = sorted(
            {
                symbol
                for file_summary in file_summaries
                for symbol in file_summary.get("touched_symbols", [])
            }
        )
        touched_functions = sorted(
            {
                symbol
                for file_summary in file_summaries
                for symbol in file_summary.get("touched_functions", [])
            }
        )
        return {
            "files_changed": len(file_summaries),
            "changed_files": [item["path"] for item in file_summaries],
            "hunks": sum(int(item.get("hunks", 0) or 0) for item in file_summaries),
            "changed_lines": sum(
                int(item.get("changed_lines", 0) or 0) for item in file_summaries
            ),
            "changed_functions": touched_functions,
            "touched_symbols": touched_symbols,
            "files": file_summaries,
        }

    def _probe_diff_file_summary(
        self,
        rel_path: str,
        old_content: str | None,
        new_content: str | None,
    ) -> dict[str, Any]:
        old_lines = (old_content or "").splitlines()
        new_lines = (new_content or "").splitlines()
        ranges: list[dict[str, int]] = []
        changed_lines = 0
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            a=old_lines,
            b=new_lines,
            autojunk=False,
        ).get_opcodes():
            if tag == "equal":
                continue
            changed_lines += max(i2 - i1, j2 - j1)
            ranges.append(
                {
                    "old_start": i1 + 1,
                    "old_end": i2,
                    "new_start": j1 + 1,
                    "new_end": j2,
                }
            )
        touched_symbols: set[str] = set()
        touched_functions: set[str] = set()
        syntax_errors: list[str] = []
        if rel_path.endswith(".py"):
            for content, key in ((old_content, "old"), (new_content, "new")):
                if content is None:
                    continue
                symbol_summary = self._python_symbols_touched_by_ranges(
                    rel_path,
                    content,
                    ranges,
                    range_prefix=key,
                )
                touched_symbols.update(symbol_summary.get("symbols", set()))
                touched_functions.update(symbol_summary.get("functions", set()))
                error = symbol_summary.get("syntax_error")
                if error:
                    syntax_errors.append(str(error))
        return {
            "path": rel_path,
            "hunks": len(ranges),
            "changed_lines": changed_lines,
            "changed_ranges": ranges,
            "touched_symbols": sorted(touched_symbols),
            "touched_functions": sorted(touched_functions),
            "syntax_errors": syntax_errors,
        }

    def _python_symbols_touched_by_ranges(
        self,
        rel_path: str,
        content: str,
        ranges: list[dict[str, int]],
        range_prefix: str,
    ) -> dict[str, Any]:
        try:
            tree = ast.parse(content)
        except SyntaxError as exc:
            return {
                "symbols": set(),
                "functions": set(),
                "syntax_error": f"{rel_path}:{exc.lineno or 0}: {exc.msg}",
            }
        symbol_ranges = self._python_symbol_ranges(tree, rel_path)
        touched_symbols: set[str] = set()
        touched_functions: set[str] = set()
        start_key = f"{range_prefix}_start"
        end_key = f"{range_prefix}_end"
        for changed_range in ranges:
            start = int(changed_range.get(start_key, 0) or 0)
            end = int(changed_range.get(end_key, 0) or 0)
            if end <= 0:
                end = start
            if start <= 0:
                start = 1
            for item in symbol_ranges:
                if not self._line_ranges_overlap(
                    start,
                    end,
                    int(item["start"]),
                    int(item["end"]),
                ):
                    continue
                names = item["names"]
                touched_symbols.update(names)
                if item["kind"] == "function":
                    touched_functions.add(str(item["canonical"]))
        return {"symbols": touched_symbols, "functions": touched_functions}

    @staticmethod
    def _line_ranges_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
        return max(start_a, start_b) <= min(end_a, end_b)

    def _python_symbol_ranges(self, tree: ast.AST, rel_path: str) -> list[dict[str, Any]]:
        ranges: list[dict[str, Any]] = []
        stack: list[str] = []

        def add_symbol(node: ast.AST, name: str, kind: str) -> None:
            if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
                return
            qualified = ".".join([*stack, name]) if stack else name
            names = {
                name,
                qualified,
                qualified.replace(".", "::"),
                f"{rel_path}::{qualified}",
            }
            ranges.append(
                {
                    "kind": kind,
                    "start": int(node.lineno),
                    "end": int(node.end_lineno),
                    "canonical": qualified,
                    "names": sorted(names),
                }
            )

        def assignment_names(node: ast.AST) -> list[str]:
            if isinstance(node, ast.Name):
                return [node.id]
            if isinstance(node, ast.Attribute):
                return [node.attr]
            if isinstance(node, (ast.Tuple, ast.List)):
                names: list[str] = []
                for element in node.elts:
                    names.extend(assignment_names(element))
                return names
            return []

        def visit(node: ast.AST) -> None:
            if isinstance(node, ast.ClassDef):
                add_symbol(node, node.name, "class")
                stack.append(node.name)
                for child in node.body:
                    visit(child)
                stack.pop()
                return
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add_symbol(node, node.name, "function")
                stack.append(node.name)
                for child in node.body:
                    visit(child)
                stack.pop()
                return
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                else:
                    targets = [node.target]
                for target in targets:
                    for name in assignment_names(target):
                        add_symbol(node, name, "binding")
            for child in ast.iter_child_nodes(node):
                visit(child)

        visit(tree)
        return ranges

    def _probe_diff_contract_violations(
        self,
        summary: dict[str, Any],
        contract: dict[str, Any],
    ) -> list[str]:
        violations: list[str] = []
        changed_files = self._string_list_from_any(summary.get("changed_files"))
        allowed_files = set(self._string_list_from_any(contract.get("allowed_files")))
        if allowed_files:
            outside_files = sorted(path for path in changed_files if path not in allowed_files)
            if outside_files:
                violations.append(
                    "changed files outside probe contract: " + ", ".join(outside_files)
                )
        max_files = int(contract.get("max_files_changed", 0) or 0)
        if max_files and len(changed_files) > max_files:
            violations.append(
                f"changed {len(changed_files)} files, max_files_changed={max_files}"
            )
        max_hunks = int(contract.get("max_hunks", 0) or 0)
        hunks = int(summary.get("hunks", 0) or 0)
        if max_hunks and hunks > max_hunks:
            violations.append(f"changed {hunks} hunks, max_hunks={max_hunks}")
        max_lines = int(contract.get("max_changed_lines", 0) or 0)
        changed_lines = int(summary.get("changed_lines", 0) or 0)
        if max_lines and changed_lines > max_lines:
            violations.append(
                f"changed {changed_lines} lines, max_changed_lines={max_lines}"
            )
        changed_functions = self._string_list_from_any(summary.get("changed_functions"))
        max_functions = int(contract.get("max_changed_functions", 0) or 0)
        if max_functions and len(self._canonical_regions(changed_functions)) > max_functions:
            violations.append(
                "changed "
                f"{len(self._canonical_regions(changed_functions))} functions, "
                f"max_changed_functions={max_functions}"
            )
        for file_summary in summary.get("files", []):
            if not isinstance(file_summary, dict):
                continue
            syntax_errors = self._string_list_from_any(file_summary.get("syntax_errors"))
            if syntax_errors:
                violations.append(
                    "python diff region mapping failed: " + "; ".join(syntax_errors[:2])
                )
        touched_symbols = self._string_list_from_any(summary.get("touched_symbols"))
        allowed_regions = self._string_list_from_any(contract.get("allowed_regions"))
        if allowed_regions and touched_symbols:
            outside = sorted(
                symbol
                for symbol in touched_symbols
                if not self._region_matches_any(symbol, allowed_regions)
            )
            if outside:
                violations.append(
                    "changed symbols outside allowed_regions: "
                    + ", ".join(outside[:6])
                )
        expected_regions = self._string_list_from_any(
            contract.get("expected_changed_regions")
        )
        if (
            expected_regions
            and changed_files
            and self._summary_has_python_changes(summary)
            and not any(
                self._region_matches_any(symbol, expected_regions)
                for symbol in touched_symbols
            )
        ):
            violations.append(
                "diff did not touch expected_changed_regions: "
                + ", ".join(expected_regions[:6])
            )
        forbidden_regions = [
            *self._string_list_from_any(contract.get("forbidden_symbols")),
            *self._string_list_from_any(contract.get("forbidden_regions")),
            *self._string_list_from_any(contract.get("required_unchanged_regions")),
        ]
        touched_forbidden = sorted(
            symbol
            for symbol in touched_symbols
            if self._region_matches_any(symbol, forbidden_regions)
        )
        if touched_forbidden:
            violations.append(
                "changed forbidden or required-unchanged regions: "
                + ", ".join(touched_forbidden[:6])
            )
        return violations

    @staticmethod
    def _summary_has_python_changes(summary: dict[str, Any]) -> bool:
        return any(
            isinstance(item, dict) and str(item.get("path", "")).endswith(".py")
            for item in summary.get("files", [])
        )

    @classmethod
    def _region_matches_any(cls, touched: str, expected: list[str]) -> bool:
        touched_aliases = cls._region_aliases(touched)
        for item in expected:
            expected_aliases = cls._region_aliases(item)
            if touched_aliases & expected_aliases:
                return True
            for touched_alias in touched_aliases:
                for expected_alias in expected_aliases:
                    if (
                        touched_alias.endswith("." + expected_alias)
                        or expected_alias.endswith("." + touched_alias)
                        or touched_alias.endswith("::" + expected_alias)
                        or expected_alias.endswith("::" + touched_alias)
                    ):
                        return True
        return False

    @staticmethod
    def _region_aliases(value: str) -> set[str]:
        text = str(value).strip()
        if not text:
            return set()
        normalized = text.replace("()", "")
        basename = normalized.rsplit("/", 1)[-1]
        region = normalized.rsplit("::", 1)[-1]
        dotted = region.replace("::", ".")
        aliases = {
            normalized,
            normalized.replace("::", "."),
            basename,
            basename.replace("::", "."),
            region,
            dotted,
            dotted.rsplit(".", 1)[-1],
        }
        return {alias.lower() for alias in aliases if alias}

    @classmethod
    def _canonical_regions(cls, values: list[str]) -> set[str]:
        canonical = set()
        for value in values:
            aliases = cls._region_aliases(value)
            if aliases:
                canonical.add(sorted(aliases, key=len)[-1])
        return canonical

    @staticmethod
    def _compact_probe_diff_summary(summary: dict[str, Any]) -> dict[str, Any]:
        compact_files = []
        for file_summary in summary.get("files", []):
            if not isinstance(file_summary, dict):
                continue
            compact_files.append(
                {
                    key: file_summary.get(key)
                    for key in (
                        "path",
                        "hunks",
                        "changed_lines",
                        "touched_symbols",
                        "touched_functions",
                        "syntax_errors",
                    )
                    if file_summary.get(key) not in (None, "", [], {})
                }
            )
        return {
            "files_changed": summary.get("files_changed", 0),
            "changed_files": summary.get("changed_files", []),
            "hunks": summary.get("hunks", 0),
            "changed_lines": summary.get("changed_lines", 0),
            "changed_functions": summary.get("changed_functions", []),
            "touched_symbols": summary.get("touched_symbols", []),
            "files": compact_files,
        }

    @staticmethod
    def _active_todo_is_structural_probe(active_todo: dict[str, Any]) -> bool:
        stage = str(active_todo.get("tactic_stage", "")).strip().lower()
        return stage == "structural_probe"

    def _change_too_broad_for_structural_probe(
        self, change: CodeChange, max_lines: int
    ) -> bool:
        text_parts = [
            change.target or "",
            change.replacement or "",
            change.patch or "",
            change.reason or "",
        ]
        line_count = max(
            self._non_empty_line_count(change.target or ""),
            self._non_empty_line_count(change.replacement or ""),
            self._non_empty_line_count(change.patch or ""),
        )
        if max_lines > 0 and line_count > max_lines:
            return True
        target = (change.target or "").lstrip()
        if target.startswith(("def ", "async def ", "class ")):
            return True
        joined = "\n".join(text_parts).lower()
        broad_patterns = (
            r"\b(rewrite|replace|refactor|restructure)\b.*\b(function|class|method|module|loop|algorithm|pipeline|hot path)\b",
            r"\b(entire|whole|all)\b.*\b(function|class|method|module|loop|algorithm|pipeline|hot path)\b",
            r"\bchange\b.*\b(data\s*flow|control\s*flow|state lifecycle|ordering)\b",
        )
        return any(re.search(pattern, joined) for pattern in broad_patterns)

    @staticmethod
    def _non_empty_line_count(text: str) -> int:
        return sum(1 for line in text.splitlines() if line.strip())

    def _change_targets_any_symbol(
        self, change: CodeChange, symbols: list[str]
    ) -> bool:
        target = change.target if isinstance(change.target, str) else ""
        if target:
            try:
                content = (self.state.repo_root / change.path).read_text(errors="replace")
            except OSError:
                content = ""
            if content and target in content:
                return self._target_text_inside_python_symbols(content, target, symbols)

        evidence_text = " ".join(
            item
            for item in (
                change.reason,
                change.target or "",
                change.replacement or "",
                change.patch or "",
            )
            if isinstance(item, str)
        ).lower()
        return any(self._symbol_mentioned(symbol, evidence_text) for symbol in symbols)

    @classmethod
    def _target_text_inside_python_symbols(
        cls, content: str, target: str, symbols: list[str]
    ) -> bool:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return False
        lines = content.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))

        ranges: list[tuple[int, int]] = []
        for symbol in symbols:
            node = cls._find_symbol_node(tree, symbol)
            if (
                node is None
                or not hasattr(node, "lineno")
                or not hasattr(node, "end_lineno")
            ):
                continue
            start_line = max(1, int(node.lineno))
            end_line = min(len(lines), int(node.end_lineno))
            ranges.append((offsets[start_line - 1], offsets[end_line]))
        if not ranges:
            return False

        start = content.find(target)
        while start >= 0:
            end = start + len(target)
            if any(
                range_start <= start and end <= range_end
                for range_start, range_end in ranges
            ):
                return True
            start = content.find(target, start + 1)
        return False

    @staticmethod
    def _symbol_mentioned(symbol: str, evidence_text: str) -> bool:
        symbol = symbol.strip()
        if not symbol:
            return False
        variants = {symbol, symbol.replace("::", "."), symbol.split(".")[-1]}
        if "::" in symbol:
            variants.add(symbol.rsplit("::", 1)[-1])
        return any(variant and variant.lower() in evidence_text for variant in variants)

    def _active_todo_duplicate_variant_rejection(
        self, candidate: CodeCandidate
    ) -> tuple[str, str] | None:
        workflow = self.config.get("workflow", {})
        if workflow.get("todo_reject_duplicate_variants", True) is False:
            return None
        if self._todo_contract_soft_now():
            return None
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
        todo_id = str(active_todo.get("todo_id", ""))
        if not todo_id:
            return None
        candidate_signature = self._todo_variant_signature_for_candidate(candidate)
        if not candidate_signature:
            return None
        threshold = float(
            workflow.get("todo_duplicate_variant_similarity_threshold", 0.92) or 0.92
        )
        for attempt in reversed(self._recent_todo_attempts(todo_id)):
            status = str(attempt.get("status", ""))
            if not status.startswith("rejected"):
                continue
            attempt_signature = self._todo_variant_signature_for_attempt(attempt)
            if not attempt_signature:
                continue
            similarity = self._signature_similarity(
                candidate_signature, attempt_signature
            )
            if similarity >= threshold:
                loop = attempt.get("loop", "?")
                return (
                    "rejected_todo_duplicate_variant",
                    f"active todo {todo_id} repeats rejected variant from loop {loop} "
                    f"(similarity={similarity:.2f})",
                )
        return None

    def _recent_todo_attempts(self, todo_id: str) -> list[dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        active_todo = self.state.scratch.get("active_todo")
        if isinstance(active_todo, dict):
            last_attempt = active_todo.get("last_attempt")
            if isinstance(last_attempt, dict) and last_attempt.get("todo_id") == todo_id:
                attempts.append(last_attempt)
        path = self._workflow_artifact_path(
            "todo_attempts_path", ".local_micro_agent/todo_attempts.jsonl"
        )
        if path.exists():
            limit = int(
                self.config.get("workflow", {}).get("todo_duplicate_variant_window", 6)
                or 6
            )
            for line in path.read_text(errors="replace").splitlines()[-limit:]:
                try:
                    attempt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(attempt, dict) and attempt.get("todo_id") == todo_id:
                    attempts.append(attempt)
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[Any, Any, Any]] = set()
        for attempt in attempts:
            key = (
                attempt.get("loop"),
                attempt.get("candidate_id"),
                attempt.get("reason"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(attempt)
        return deduped

    def _todo_variant_signature_for_candidate(
        self, candidate: CodeCandidate
    ) -> set[str]:
        return self._tactic_signature(
            "\n".join(
                [
                    candidate.reason,
                    *(change.reason for change in candidate.changes),
                ]
            )
        )

    def _todo_variant_signature_for_attempt(self, attempt: dict[str, Any]) -> set[str]:
        return self._tactic_signature(str(attempt.get("reason", "")))

    def _candidate_family_contract_rejection(
        self, candidate: CodeCandidate
    ) -> tuple[str, str] | None:
        selected_family = self._selected_tactic_family_for_current_loop()
        if not selected_family:
            return None
        candidate_families = self._candidate_reason_family_aliases(candidate)
        if selected_family in candidate_families:
            return None
        failed_families = self._failed_tactic_family_keys() | set(
            self._skipped_brainstorm_family_aliases()
        )
        drift_matches = sorted(candidate_families & failed_families)
        if not drift_matches:
            return None
        gate_decision = self._adaptive_gate_decision(
            gate="candidate_family_drift",
            match_reason="failed_family=" + ",".join(drift_matches),
            family_aliases=drift_matches,
            tactic_text="\n".join(
                [candidate.reason, *(change.reason for change in candidate.changes)]
            ),
        )
        self._persist_gate_decision(gate_decision)
        if gate_decision["mode"] != "hard":
            self.state.notes.append(
                "Adaptive gate allowed candidate family drift "
                f"mode={gate_decision['mode']} reason={gate_decision['reason']}"
            )
            return None
        return (
            "rejected_family_drift",
            "candidate reason targets failed family "
            f"{', '.join(drift_matches)} instead of selected family_key {selected_family}",
        )

    def _candidate_reason_family_aliases(self, candidate: CodeCandidate) -> set[str]:
        reason_parts = [candidate.reason, *(change.reason for change in candidate.changes)]
        reason_text = "\n".join(reason_parts)
        return {
            self._normalize_strategy_axis(alias)
            for alias in self._tactic_family_aliases(reason_text, include_axes=False)
            if self._normalize_strategy_axis(alias)
        }

    @staticmethod
    def _normalize_strategy_axis(axis: str) -> str:
        return normalize_strategy_axis(axis)

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
        if self._candidate_matches_selected_tactic(candidate):
            return []
        selected_axis = self._selected_tactic_axis_for_current_loop()
        cooled = []
        for axis in self._candidate_strategy_axes(candidate):
            if axis == selected_axis:
                continue
            axis_state = axes_state.get(axis)
            if not isinstance(axis_state, dict):
                continue
            cooldown_until = axis_state.get("cooldown_until_loop")
            if isinstance(cooldown_until, int) and cooldown_until > current_loop:
                cooled.append(axis)
        return cooled

    def _cooled_candidate_regions(self, candidate: CodeCandidate) -> list[str]:
        if not self._adaptive_search_reject_cooled_regions_enabled():
            return []
        if self._candidate_matches_selected_tactic(candidate):
            return []
        memory = self.state.scratch.get("adaptive_search_memory")
        if not isinstance(memory, dict):
            memory = self._adaptive_search_memory_from_history()
            if memory:
                self.state.scratch["adaptive_search_memory"] = memory
        if not isinstance(memory, dict):
            return []
        regions_state = memory.get("regions")
        if not isinstance(regions_state, dict):
            return []
        current_loop = self.state.loop_count
        cooled: list[str] = []
        for region_key in self._candidate_region_keys(candidate):
            region_state = regions_state.get(region_key)
            if not isinstance(region_state, dict):
                continue
            cooldown_until = region_state.get("cooldown_until_loop")
            if isinstance(cooldown_until, int) and cooldown_until > current_loop:
                cooled.append(region_key)
        return sorted(cooled)

    def _candidate_region_keys(self, candidate: CodeCandidate) -> list[str]:
        axes = self._candidate_strategy_axes(candidate)
        families = sorted(self._candidate_reason_family_aliases(candidate))
        family = families[0] if families else ""
        keys: list[str] = []
        for change in candidate.changes:
            region = self._change_region_key(change)
            for axis in axes:
                suffix = axis if not family else f"{axis}+{family}"
                keys.append(f"{region}::{suffix}")
        return sorted(set(keys))

    def _change_region_key(self, change: CodeChange) -> str:
        path = change.path
        content = self._current_source_for_region(path)
        if not content:
            return f"{path}::file"
        anchor = change.target or change.patch or change.content or change.reason
        line_no = self._best_anchor_line_number(content, anchor)
        symbol = self._python_symbol_at_line(content, line_no)
        if symbol:
            return f"{path}::{symbol}"
        bucket = max(1, ((max(line_no, 1) - 1) // 50) + 1)
        return f"{path}::lines_{bucket * 50 - 49}_{bucket * 50}"

    def _current_source_for_region(self, path: str) -> str:
        snapshot = self.state.scratch.get("pre_code_snapshot")
        if isinstance(snapshot, dict) and isinstance(snapshot.get(path), str):
            return str(snapshot[path])
        abs_path = self.state.repo_root / path
        try:
            return abs_path.read_text(errors="replace")
        except OSError:
            return ""

    def _best_anchor_line_number(self, content: str, anchor: str) -> int:
        if not anchor:
            return 1
        index = content.find(anchor)
        if index >= 0:
            return content[:index].count("\n") + 1
        lines = content.splitlines()
        tokens = self._anchor_tokens(anchor)
        best_index = 0
        best_score = -1
        if tokens:
            for line_index, line in enumerate(lines):
                score = len(tokens & self._anchor_tokens(line))
                if score > best_score:
                    best_index = line_index
                    best_score = score
        return best_index + 1

    @staticmethod
    def _python_symbol_at_line(content: str, line_no: int) -> str:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return ""
        best: tuple[int, str] | None = None
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            if start <= line_no <= end:
                span = end - start
                name = node.name
                if best is None or span < best[0]:
                    best = (span, name)
        return best[1] if best else ""

    def _candidate_matches_selected_tactic(self, candidate: CodeCandidate) -> bool:
        selected_axis = self._selected_tactic_axis_for_current_loop()
        if not selected_axis:
            return False
        declared = self._normalize_strategy_axis(candidate.strategy_axis)
        if declared != selected_axis:
            return False
        return selected_axis in self._candidate_strategy_axes(candidate)

    def _selected_tactic_axis_for_current_loop(self) -> str | None:
        selected_tactic = self._selected_tactic_for_current_loop()
        if not selected_tactic:
            return None
        axis = self._normalize_strategy_axis(str(selected_tactic.get("strategy_axis", "")))
        return axis or None

    def _selected_tactic_family_for_current_loop(self) -> str | None:
        selected_tactic = self._selected_tactic_for_current_loop()
        if not selected_tactic:
            return None
        family_key = self._normalize_strategy_axis(str(selected_tactic.get("family_key", "")))
        return family_key or None

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
        regions_state = memory.setdefault("regions", {})
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
        region_keys = self._candidate_region_keys(candidate)
        for region_key in region_keys:
            region_state = regions_state.setdefault(
                region_key,
                {
                    "attempts": 0,
                    "failures": 0,
                    "successes": 0,
                    "cooldown_until_loop": None,
                    "last_status": None,
                    "last_metric": None,
                },
            )
            region_state["attempts"] = int(region_state.get("attempts", 0)) + 1
            region_state["last_status"] = status
            region_state["last_metric"] = metric
            if improved:
                region_state["successes"] = int(region_state.get("successes", 0)) + 1
                region_state["cooldown_until_loop"] = None
            else:
                region_state["failures"] = int(region_state.get("failures", 0)) + 1
                if self._region_should_cool_down(region_key, status):
                    cooldown = int(
                        self.config.get("workflow", {}).get(
                            "adaptive_search_region_cooldown_loops",
                            self.config.get("workflow", {}).get(
                                "adaptive_search_axis_cooldown_loops", 3
                            ),
                        )
                    )
                    region_state["cooldown_until_loop"] = self.state.loop_count + cooldown
        recent.append(
            {
                "loop": self.state.loop_count,
                "candidate_id": candidate.candidate_id,
                "axes": axes,
                "regions": region_keys,
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
        failure_statuses = self._adaptive_search_failure_statuses()
        if status not in failure_statuses:
            return False
        recent_failures = 1
        for record in reversed(recent[-window:]):
            if axis not in record.get("axes", []):
                continue
            if record.get("status") in failure_statuses or record.get("failed") is True:
                recent_failures += 1
        return recent_failures >= threshold

    def _region_should_cool_down(self, region_key: str, status: str) -> bool:
        memory = self.state.scratch.get("adaptive_search_memory")
        if not isinstance(memory, dict):
            return False
        recent = memory.get("recent")
        if not isinstance(recent, list):
            return False
        window = int(
            self.config.get("workflow", {}).get(
                "adaptive_search_region_window",
                self.config.get("workflow", {}).get("adaptive_search_axis_window", 8),
            )
        )
        threshold = int(
            self.config.get("workflow", {}).get(
                "adaptive_search_region_failure_threshold",
                self.config.get("workflow", {}).get("adaptive_search_axis_failure_threshold", 3),
            )
        )
        failure_statuses = self._adaptive_search_failure_statuses()
        if status not in failure_statuses:
            return False
        recent_failures = 1
        for record in reversed(recent[-window:]):
            if region_key not in record.get("regions", []):
                continue
            if record.get("status") in failure_statuses or record.get("failed") is True:
                recent_failures += 1
        return recent_failures >= threshold

    @staticmethod
    def _adaptive_search_failure_statuses() -> set[str]:
        return {
            "rejected",
            "rejected_cooled_axis",
            "rejected_cooled_region",
            "rejected_missing_axis",
            "rejected_family_drift",
            "rejected_no_changes",
            "rejected_repeated_pattern",
            "rejected_todo_axis_drift",
            "rejected_todo_family_drift",
            "rejected_unknown_axis",
            "rejected_wrong_axis",
            "rejected_no_metric",
        }

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
                "failure_classes": raw_state.get("failure_classes", {}),
                "last_status": raw_state.get("last_status"),
                "last_metric": raw_state.get("last_metric"),
                "best_metric": raw_state.get("best_metric"),
            }
            if is_cooled:
                item["cooldown_until_loop"] = cooldown_until
                cooled_down.append(axis)
            axes.append(item)
        regions_state = memory.get("regions")
        regions = []
        cooled_regions = []
        if isinstance(regions_state, dict):
            for region_key, raw_state in sorted(regions_state.items()):
                if not isinstance(raw_state, dict):
                    continue
                cooldown_until = raw_state.get("cooldown_until_loop")
                is_cooled = isinstance(cooldown_until, int) and cooldown_until > current_loop
                item = {
                    "region": region_key,
                    "attempts": raw_state.get("attempts", 0),
                    "failures": raw_state.get("failures", 0),
                    "successes": raw_state.get("successes", 0),
                    "failure_classes": raw_state.get("failure_classes", {}),
                    "last_status": raw_state.get("last_status"),
                    "last_metric": raw_state.get("last_metric"),
                }
                if is_cooled:
                    item["cooldown_until_loop"] = cooldown_until
                    cooled_regions.append(region_key)
                regions.append(item)
        recent = memory.get("recent") if isinstance(memory.get("recent"), list) else []
        payload = {
            "current_loop": current_loop,
            "cooled_down_axes": cooled_down,
            "cooled_down_regions": cooled_regions,
            "gate_controller": self._adaptive_gate_controller_summary(),
            "axes": axes,
            "regions": regions[-8:],
            "recent": recent[-5:],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _adaptive_gate_controller_summary(self) -> dict[str, Any]:
        return {
            "enabled": self._adaptive_gate_controller_enabled(),
            "all_skipped_streak": self._brainstorm_all_skipped_streak(),
            "relax_streak": int(
                self.config.get("workflow", {}).get(
                    "adaptive_gate_all_skipped_relax_streak", 2
                )
                or 0
            ),
            "min_family_attempts_for_hard": int(
                self.config.get("workflow", {}).get(
                    "adaptive_gate_min_family_attempts_for_hard", 2
                )
                or 0
            ),
        }

    def _format_adaptive_gate_memory(self) -> str:
        if not self._adaptive_gate_controller_enabled():
            return ""
        path = self._workflow_artifact_path(
            "adaptive_gate_decisions_path", ".local_micro_agent/gate_decisions.jsonl"
        )
        summary = self._adaptive_gate_controller_summary()
        if not path.exists():
            return json.dumps(
                {"summary": summary, "recent_gate_decisions": []},
                ensure_ascii=False,
                indent=2,
            )
        limit = int(self.config.get("workflow", {}).get("adaptive_gate_recent_limit", 8) or 8)
        records = []
        for line in path.read_text(errors="replace").splitlines()[-limit:]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(
                {
                    "loop": record.get("loop"),
                    "gate": record.get("gate"),
                    "mode": record.get("mode"),
                    "reason": record.get("reason"),
                    "match_reason": record.get("match_reason"),
                    "family_aliases": record.get("family_aliases", []),
                    "all_skipped_streak": record.get("all_skipped_streak"),
                }
            )
        return json.dumps(
            {"summary": summary, "recent_gate_decisions": records},
            ensure_ascii=False,
            indent=2,
        )

    def _adaptive_search_memory_from_history(self) -> dict[str, Any] | None:
        path = self._candidate_history_path()
        if path is None or not path.exists():
            return None
        limit = int(self.config.get("workflow", {}).get("candidate_history_limit", 20))
        lines = path.read_text(errors="replace").splitlines()[-limit:]
        memory: dict[str, Any] = {"axes": {}, "recent": []}
        failure_statuses = self._adaptive_search_failure_statuses()
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
            failure_class = str(record.get("failure_class") or "")
            recent_record = {
                "loop": record.get("loop"),
                "candidate_id": record.get("candidate_id"),
                "axes": [str(axis) for axis in axes],
                "regions": [
                    str(region)
                    for region in record.get("region_keys", [])
                    if str(region)
                ],
                "family_aliases": [
                    str(alias)
                    for alias in record.get("family_aliases", [])
                    if str(alias)
                ],
                "status": status,
                "failure_class": failure_class,
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
                        "failure_classes": {},
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
                    if failure_class:
                        classes = axis_state.setdefault("failure_classes", {})
                        classes[failure_class] = int(classes.get(failure_class, 0)) + 1
            for region_key in recent_record["regions"]:
                region_state = memory.setdefault("regions", {}).setdefault(
                    region_key,
                    {
                        "attempts": 0,
                        "failures": 0,
                        "successes": 0,
                        "cooldown_until_loop": None,
                        "last_status": None,
                        "last_metric": None,
                        "failure_classes": {},
                    },
                )
                region_state["attempts"] += 1
                region_state["last_status"] = status
                region_state["last_metric"] = metric
                if status in {"improved", "accepted"} and not failed:
                    region_state["successes"] += 1
                elif status in failure_statuses or failed:
                    region_state["failures"] += 1
                    if failure_class:
                        classes = region_state.setdefault("failure_classes", {})
                        classes[failure_class] = int(classes.get(failure_class, 0)) + 1
        self._apply_history_cooldowns(memory, failure_statuses)
        return memory if memory["axes"] or memory.get("regions") else None

    def _failure_memory_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(workflow.get("adaptive_search_memory")) and workflow.get(
            "failure_memory", True
        ) is not False

    def _failure_memory_path(self) -> Path:
        return self._workflow_artifact_path(
            "failure_memory_path", ".local_micro_agent/failure_memory.jsonl"
        )

    def _format_failure_memory(self) -> str:
        if not self._failure_memory_enabled():
            return ""
        path = self._failure_memory_path()
        if not path.exists():
            return ""
        limit = int(self.config.get("workflow", {}).get("failure_memory_recent_limit", 8) or 8)
        payload = self._scoped_failure_memory_payload(limit)
        if not payload["current_repo_issues"] and not payload["rejected_candidate_lessons"]:
            return ""
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _scoped_failure_memory_payload(self, limit: int) -> dict[str, Any]:
        path = self._failure_memory_path()
        records: list[dict[str, Any]] = []
        for line in path.read_text(errors="replace").splitlines()[-limit:]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(self._compact_failure_memory_record(record))
        current_repo_issues = [
            record
            for record in records
            if record.get("issue_scope") == "current_repo"
            or record.get("repair_task_eligible") is True
        ]
        rejected_candidate_lessons = [
            record
            for record in records
            if record not in current_repo_issues
        ]
        return {
            "failure_memory_policy": (
                "Only current_repo_issues may be converted into repair tasks. "
                "Rejected candidate lessons describe transient candidate deltas, "
                "patch misses, contract rejects, metric gates, or invalid design "
                "shapes; use them as negative evidence, not as proof that the "
                "current repository is broken."
            ),
            "current_repo_issues": current_repo_issues,
            "rejected_candidate_lessons": rejected_candidate_lessons,
        }

    def _compact_failure_memory_record(self, record: dict[str, Any]) -> dict[str, Any]:
        issue_scope = str(record.get("issue_scope") or "")
        repair_task_eligible = record.get("repair_task_eligible")
        if not issue_scope:
            issue_scope = (
                "current_repo"
                if repair_task_eligible is True
                else "candidate_delta"
            )
        compact_keys = (
            "loop",
            "strategy_axis",
            "failure_class",
            "failure_signature",
            "observed_signal",
            "why_invalid",
            "next_rule",
            "repair_hint",
            "failure_origin",
            "issue_scope",
            "repo_valid_after_restore",
            "repair_task_eligible",
            "memory_use",
        )
        compact = {
            key: record.get(key)
            for key in compact_keys
            if record.get(key) not in (None, "", [], {})
        }
        compact["issue_scope"] = issue_scope
        if repair_task_eligible is not None:
            compact["repair_task_eligible"] = bool(repair_task_eligible)
        return compact

    def _spec_candidate_failure_scope_context(self) -> str:
        if not self._failure_memory_enabled():
            return ""
        path = self._failure_memory_path()
        if not path.exists():
            return ""
        limit = int(
            self.config.get("workflow", {}).get(
                "spec_candidate_failure_memory_limit",
                self.config.get("workflow", {}).get("failure_memory_recent_limit", 8),
            )
            or 8
        )
        payload = self._scoped_failure_memory_payload(limit)
        if not payload["current_repo_issues"] and not payload["rejected_candidate_lessons"]:
            return ""
        return (
            "Recent candidate failure memory by issue scope follows. Current repo "
            "issues are the only entries eligible for repair/syntax-fix tasks. "
            "Rejected candidate lessons came from discarded candidate deltas, "
            "pre-apply contracts, patch application, or metric gates; do not turn "
            "their SyntaxError/test/patch-miss text into a new current-code repair "
            "task. Use those lessons only to avoid, retarget, or shrink future "
            "candidate shapes.\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def _remember_failure_lesson(
        self,
        candidate: CodeCandidate,
        status: str,
        metric: int | None,
        applied: int,
        failed: bool,
        failure_class: str,
        failure_detail: str,
        no_change_reason: str,
        diagnostic_results: list[dict[str, Any]],
        recovery_hint: str,
        failure_scope: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._failure_memory_enabled():
            return {}
        if status in {"improved", "accepted"} and not failed:
            return {}
        detail = no_change_reason or failure_detail
        scope = failure_scope or self._candidate_failure_scope(
            status=status,
            applied=applied,
            failed=failed,
            failure_class=failure_class,
            failure_detail=failure_detail,
            no_change_reason=no_change_reason,
            results=[],
        )
        diagnostic_summary = self._diagnostic_summary(diagnostic_results, limit=500)
        observed_signal = []
        if metric is not None:
            observed_signal.append(f"metric={metric}")
        if diagnostic_summary:
            observed_signal.append(f"diagnostics={diagnostic_summary}")
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "loop": self.state.loop_count,
            "candidate_id": candidate.candidate_id,
            "status": status,
            "strategy_axis": candidate.strategy_axis,
            "strategy_axes": self._candidate_strategy_axes(candidate),
            "failure_class": failure_class,
            "failure_signature": sorted(
                self._tactic_signature(
                    "\n".join(
                        [candidate.reason, *(change.reason for change in candidate.changes)]
                    )
                )
            )[:16],
            "observed_signal": self._truncate_text(" | ".join(observed_signal), 700),
            "why_invalid": self._truncate_text(detail or status, 700),
            "next_rule": self._failure_memory_next_rule(failure_class, detail),
            "repair_hint": self._truncate_text(recovery_hint, 500),
        }
        record.update(scope)
        path = self._failure_memory_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return {
            "failure_memory_path": self._repo_relative_path(path),
            "failure_memory_next_rule": record["next_rule"],
        }

    def _candidate_failure_scope(
        self,
        status: str,
        applied: int,
        failed: bool,
        failure_class: str,
        failure_detail: str = "",
        no_change_reason: str = "",
        results: list[Any] | None = None,
    ) -> dict[str, Any]:
        detail = self._normalize_fingerprint_text(
            " ".join([status, failure_class, failure_detail, no_change_reason])
        )
        result_list = results or []
        result_failed = any(getattr(result, "exit_code", 0) != 0 for result in result_list)
        origin = self._candidate_failure_origin(
            status=status,
            applied=applied,
            failed=failed,
            failure_class=failure_class,
            detail=detail,
            result_failed=result_failed,
        )
        issue_scope = self._candidate_issue_scope(
            origin=origin,
            applied=applied,
            result_failed=result_failed,
        )
        repo_valid_after_restore: bool | None
        if issue_scope == "current_repo":
            repo_valid_after_restore = False
        elif issue_scope == "candidate_delta":
            repo_valid_after_restore = (
                True
                if isinstance(self.state.scratch.get("pre_code_snapshot"), dict)
                else None
            )
        else:
            repo_valid_after_restore = None
        repair_task_eligible = issue_scope == "current_repo"
        return {
            "failure_origin": origin,
            "issue_scope": issue_scope,
            "repo_valid_after_restore": repo_valid_after_restore,
            "repair_task_eligible": repair_task_eligible,
            "memory_use": self._candidate_memory_use(origin, issue_scope),
        }

    @staticmethod
    def _candidate_failure_origin(
        status: str,
        applied: int,
        failed: bool,
        failure_class: str,
        detail: str,
        result_failed: bool,
    ) -> str:
        if "post restore" in detail or "post-restore" in detail or "after restore" in detail:
            return "post_restore_validation"
        if (
            status == "rejected_probe_contract_mismatch"
            or failure_class == "probe_contract_mismatch"
        ):
            return "post_apply_contract"
        if "design contract" in detail or status in {"design_rejected", "failed_design"}:
            return "design_contract"
        if status.startswith("rejected_todo") or failure_class in {
            "contract_mismatch",
            "axis_mismatch",
            "family_mismatch",
        }:
            return "pre_apply_contract"
        if failure_class == "patch_miss":
            return "patch_apply"
        if failure_class in {
            "no_improvement",
            "probe_no_signal",
            "scaffold_validated",
            "metric_missing",
        }:
            return "metric_gate"
        if applied > 0 or result_failed or failed:
            return "candidate_validation"
        return "unknown"

    @staticmethod
    def _candidate_issue_scope(
        origin: str,
        applied: int,
        result_failed: bool,
    ) -> str:
        if origin == "post_restore_validation":
            return "current_repo"
        if origin == "design_contract":
            return "design_shape"
        if origin == "candidate_validation" and applied <= 0 and result_failed:
            return "current_repo"
        if origin in {
            "patch_apply",
            "pre_apply_contract",
            "post_apply_contract",
            "candidate_validation",
            "metric_gate",
        }:
            return "candidate_delta"
        return "unknown"

    @staticmethod
    def _candidate_memory_use(origin: str, issue_scope: str) -> str:
        if issue_scope == "current_repo":
            return "create_repair_task"
        if origin == "patch_apply":
            return "retry_with_fresh_context"
        if issue_scope == "design_shape":
            return "avoid_shape"
        if issue_scope == "candidate_delta":
            return "avoid_shape"
        return "ignore_for_spec"

    @staticmethod
    def _failure_memory_next_rule(failure_class: str, detail: str) -> str:
        normalized = normalize_fingerprint_text(detail)
        if failure_class in {"patch_miss", "duplicate_variant", "axis_mismatch", "family_mismatch"}:
            return "avoid"
        if failure_class == "probe_contract_mismatch":
            return "repair_with_constraint"
        if "out of scratch space" in normalized or "out of memory" in normalized:
            return "repair_with_constraint"
        if failure_class in {"correctness_failure", "invariant_broken", "scope_too_broad"}:
            return "repair_with_constraint"
        if failure_class in {"no_improvement", "probe_no_signal"}:
            return "avoid_same_variant"
        return "avoid_or_repair_with_new_evidence"

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
        regions_state = memory.get("regions")
        if not isinstance(regions_state, dict):
            return
        region_threshold = int(
            self.config.get("workflow", {}).get(
                "adaptive_search_region_failure_threshold", threshold
            )
        )
        region_cooldown = int(
            self.config.get("workflow", {}).get(
                "adaptive_search_region_cooldown_loops", cooldown
            )
        )
        for region_key, region_state in regions_state.items():
            recent_failures = 0
            for record in reversed(recent[-window:]):
                if region_key not in record.get("regions", []):
                    continue
                if record.get("status") in failure_statuses or record.get("failed") is True:
                    recent_failures += 1
            if recent_failures >= region_threshold:
                region_state["cooldown_until_loop"] = (
                    self.state.loop_count + region_cooldown
                )

    @staticmethod
    def _normalize_fingerprint_text(text: str) -> str:
        return normalize_fingerprint_text(text)
