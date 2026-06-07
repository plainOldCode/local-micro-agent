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

from .decisions import CodeCandidate, CodeDecision, ReadDecision, TestDecision
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
from .strategy import (
    DEFAULT_STRATEGY_AXIS_GUIDANCE,
    DEFAULT_STRATEGY_AXIS_KEYWORDS,
    axis_label_matches_text,
    explicit_tactic_family_key,
    extract_tactic_axis,
    keyword_phrase_matches,
    keyword_token_matches,
    normalize_fingerprint_text,
    normalize_strategy_axis,
    signature_similarity,
    strategy_axes_for_text,
    tactic_novelty_lane,
    tactic_signature,
)
from .validators import (
    JsonValidationError,
    parse_json_object,
    parse_xml_candidates,
    require_keys,
    retry_repair_prompt,
)


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

    def _profile_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return bool(
            workflow.get("profile_agent")
            or workflow.get("debug_profile_agent")
            or workflow.get("profile_agent_debug")
        )

    def _profile_span_start(self) -> dict[str, float]:
        return {"wall": time.time(), "perf": time.perf_counter()}

    def _record_profile_span(
        self,
        event_type: str,
        start: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self._profile_enabled():
            return
        now_wall = time.time()
        elapsed_ms = (time.perf_counter() - start["perf"]) * 1000
        record: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "event_type": event_type,
            "loop": self.state.loop_count,
            "state": str(self.state.current),
            "elapsed_ms": round(elapsed_ms, 3),
            "started_at_epoch": round(start["wall"], 6),
            "ended_at_epoch": round(now_wall, 6),
        }
        if extra:
            record.update(
                {
                    key: value
                    for key, value in extra.items()
                    if value not in (None, "", [], {})
                }
            )
        path = self._workflow_artifact_path(
            "profile_events_path", ".local_micro_agent/profile_events.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    async def _profiled_phase(self, phase: str, func) -> None:
        start = self._profile_span_start()
        start_loop = self.state.loop_count
        start_state = str(self.state.current)
        try:
            await func()
        except Exception as exc:
            self._record_profile_span(
                "phase",
                start,
                {
                    "phase": phase,
                    "start_loop": start_loop,
                    "end_loop": self.state.loop_count,
                    "start_state": start_state,
                    "end_state": str(self.state.current),
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
        self._record_profile_span(
            "phase",
            start,
            {
                "phase": phase,
                "start_loop": start_loop,
                "end_loop": self.state.loop_count,
                "start_state": start_state,
                "end_state": str(self.state.current),
                "success": True,
            },
        )

    async def _model_chat(
        self, role: str, messages: list[dict[str, str]], call_site: str = ""
    ) -> str:
        start = self._profile_span_start()
        prompt_chars = sum(len(str(message.get("content", ""))) for message in messages)
        model_name = self.config.get("models", {}).get(role) or self.config.get(
            "models", {}
        ).get("default")
        provider = self.config.get("providers", {}).get(str(model_name), {})
        model = self.models.get(role)
        stream_callback, stream_stats = self._profile_model_stream_callback(
            model=model,
            role=role,
            call_site=call_site,
            model_name=str(model_name or ""),
            provider=provider,
        )
        try:
            if stream_callback is not None:
                output = await model.chat(messages, stream_callback=stream_callback)
            else:
                output = await model.chat(messages)
        except Exception as exc:
            self._record_profile_span(
                "model_call",
                start,
                {
                    "role": role,
                    "call_site": call_site,
                    "model_name": model_name,
                    "provider_kind": provider.get("kind"),
                    "provider_model": provider.get("model"),
                    "message_count": len(messages),
                    "prompt_chars": prompt_chars,
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    **stream_stats,
                },
            )
            raise
        self._record_profile_span(
            "model_call",
            start,
            {
                "role": role,
                "call_site": call_site,
                "model_name": model_name,
                "provider_kind": provider.get("kind"),
                "provider_model": provider.get("model"),
                "message_count": len(messages),
                "prompt_chars": prompt_chars,
                "output_chars": len(output),
                "success": True,
                **stream_stats,
            },
        )
        return output

    def _profile_model_stream_enabled(self) -> bool:
        workflow = self.config.get("workflow", {})
        return self._profile_enabled() and bool(workflow.get("profile_model_stream", True))

    def _profile_model_stream_callback(
        self,
        model: Any,
        role: str,
        call_site: str,
        model_name: str,
        provider: dict[str, Any],
    ) -> tuple[Any | None, dict[str, Any]]:
        if not self._profile_model_stream_enabled():
            return None, {}
        if not bool(getattr(model, "supports_streaming", False)):
            return None, {}
        workflow = self.config.get("workflow", {})
        seq = int(self.state.scratch.get("_profile_model_stream_seq", 0) or 0) + 1
        self.state.scratch["_profile_model_stream_seq"] = seq
        label_parts = [
            f"{seq:04d}",
            f"loop-{self.state.loop_count:03d}",
            self._safe_stream_label(str(self.state.current)),
            self._safe_stream_label(role),
            self._safe_stream_label(call_site or "chat"),
        ]
        stream_dir = self._workflow_artifact_path(
            "model_stream_dir", ".local_micro_agent/model_streams"
        )
        stream_path = stream_dir / ("-".join(label_parts) + ".txt")
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        stream_path.write_text("")
        interval = int(workflow.get("profile_model_stream_log_interval_chars", 2000) or 0)
        stats: dict[str, Any] = {
            "streaming": True,
            "stream_path": self._repo_relative_path(stream_path),
            "stream_chunks": 0,
            "stream_chars": 0,
        }
        next_log_at = {"value": interval}
        self._log(
            "STREAM start "
            f"role={role} call_site={call_site or 'chat'} model={model_name} "
            f"provider={provider.get('kind', '')} path={stats['stream_path']}"
        )

        def on_chunk(chunk: str) -> None:
            if not chunk:
                return
            with stream_path.open("a") as handle:
                handle.write(chunk)
            stats["stream_chunks"] = int(stats.get("stream_chunks", 0)) + 1
            stats["stream_chars"] = int(stats.get("stream_chars", 0)) + len(chunk)
            if interval <= 0:
                return
            if int(stats["stream_chars"]) < next_log_at["value"]:
                return
            self._log(
                "STREAM progress "
                f"role={role} call_site={call_site or 'chat'} "
                f"chars={stats['stream_chars']} chunks={stats['stream_chunks']} "
                f"path={stats['stream_path']}"
            )
            while next_log_at["value"] <= int(stats["stream_chars"]):
                next_log_at["value"] += interval

        return on_chunk, stats

    @staticmethod
    def _safe_stream_label(value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip().lower())
        return cleaned.strip("._-") or "item"

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
        seeded_plan = self.config.get("workflow", {}).get("plan_markdown")
        if seeded_plan:
            self.state.plan_markdown = seeded_plan.strip()
            self.state.current = AgentStateName.READ
            return

        project_context = await self._load_project_context()
        workflow_context = self._workflow_plan_context()
        if workflow_context:
            project_context = "\n\n".join(part for part in [project_context, workflow_context] if part)
        output = await self._model_chat(
            "planner",
            plan_prompt(self.state, project_context),
            call_site="plan",
        )
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
        todo = {
            "todo_id": todo_id,
            "parent_tactic_id": f"brainstorm-loop-{self.state.loop_count}",
            "status": "active",
            "strategy_axis": axis,
            "family_key": self._tactic_family_key(tactic_text),
            "title": f"Feasibility probe for {axis}",
            "context": tactic_text,
            "micro_goal": (
                "Implement the smallest correctness-preserving feasibility probe for this "
                "tactic. Do not attempt the full architecture migration in one patch."
            ),
            "implementation_hint": (
                "Prefer one narrow edit that proves the tactic changes real behavior "
                "before expanding it."
            ),
            "allowed_files": sorted(self._writable_files()),
            "forbidden_patterns": [
                "broad rewrite unrelated to the selected tactic",
                "changing tests or fixtures unless explicitly allowed",
                "mixing multiple independent tactics",
            ],
            "expected_signal": (
                "Tests still pass, and any configured metric remains parseable. A metric "
                "improvement is welcome but not required for the first feasibility probe."
            ),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "created_loop": self.state.loop_count,
        }
        self.state.scratch["active_todo"] = todo
        self._persist_todo_plan(todo)

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
            )
            self._append_candidate_history(
                candidate,
                status=status,
                metric=metric,
                applied=applied,
                failed=failed,
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
            "selected_tactic": self._selected_tactic_for_axis_contract(),
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

    def _selected_tactic_for_axis_contract(self) -> dict[str, Any]:
        selected_tactic = self.state.scratch.get("selected_tactic")
        if not isinstance(selected_tactic, dict):
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
        selected_tactic = self.state.scratch.get("selected_tactic")
        selected_loop = self.state.scratch.get("selected_tactic_loop")
        known_axes = self._brainstorm_known_axes()
        if (
            isinstance(selected_tactic, dict)
            and selected_loop == self.state.loop_count
            and (
                not self._strict_strategy_axis_pool_enabled()
                or selected_tactic.get("strategy_axis") in known_axes
            )
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
            reason_axes = self._candidate_reason_strategy_axes(candidate)
            if required_axis not in reason_axes:
                return (
                    "rejected_todo_axis_drift",
                    "candidate reason does not substantively target active todo "
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
        return None

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

    def _candidate_matches_selected_tactic(self, candidate: CodeCandidate) -> bool:
        selected_axis = self._selected_tactic_axis_for_current_loop()
        if not selected_axis:
            return False
        declared = self._normalize_strategy_axis(candidate.strategy_axis)
        if declared != selected_axis:
            return False
        return selected_axis in self._candidate_reason_strategy_axes(candidate)

    def _selected_tactic_axis_for_current_loop(self) -> str | None:
        selected_tactic = self.state.scratch.get("selected_tactic")
        if not isinstance(selected_tactic, dict):
            return None
        if self.state.scratch.get("selected_tactic_loop") != self.state.loop_count:
            return None
        axis = self._normalize_strategy_axis(str(selected_tactic.get("strategy_axis", "")))
        if axis and (
            not self._strict_strategy_axis_pool_enabled()
            or axis in self._strategy_axis_pool()
        ):
            return axis
        return None

    def _selected_tactic_family_for_current_loop(self) -> str | None:
        selected_tactic = self.state.scratch.get("selected_tactic")
        if not isinstance(selected_tactic, dict):
            return None
        if self.state.scratch.get("selected_tactic_loop") != self.state.loop_count:
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
            "rejected_family_drift",
            "rejected_no_changes",
            "rejected_repeated_pattern",
            "rejected_todo_axis_drift",
            "rejected_todo_family_drift",
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
            "gate_controller": self._adaptive_gate_controller_summary(),
            "axes": axes,
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
        return normalize_fingerprint_text(text)

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
                self._create_validated_pattern_followup_todo()
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

    def _has_current_run_improvement(self) -> bool:
        if self.state.scratch.get("metric_improved") is True:
            return True
        return bool(
            self._latest_candidate_record_with_status("improved")
            or self._latest_candidate_record_with_status("accepted")
        )

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
        if self._has_active_todo_budget():
            return False
        records = self._candidate_history_records(limit=max(threshold, 1))
        if len(records) < threshold:
            return False
        return all(str(record.get("status", "")).startswith("rejected") for record in records)

    def _has_active_todo_budget(self) -> bool:
        if self._todo_contract_soft_now():
            return False
        active_todo = self.state.scratch.get("active_todo")
        if not isinstance(active_todo, dict):
            active_todo = self._load_active_todo()
            if active_todo:
                self.state.scratch["active_todo"] = active_todo
        if not isinstance(active_todo, dict) or not active_todo:
            return False
        if active_todo.get("status") not in {"active", "attempted"}:
            return False
        return not self._todo_attempt_budget_exhausted(active_todo)

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
            summary.append(
                {
                    "todo_id": todo.get("todo_id"),
                    "status": todo.get("status"),
                    "strategy_axis": todo.get("strategy_axis"),
                    "attempts": todo.get("attempts", 0),
                    "context": self._truncate_text(str(todo.get("context", "")), 280),
                    "last_status": (
                        last_attempt.get("status") if isinstance(last_attempt, dict) else None
                    ),
                    "last_metric": (
                        last_attempt.get("metric") if isinstance(last_attempt, dict) else None
                    ),
                    "last_reason": self._truncate_text(
                        str(last_attempt.get("reason", "")), 220
                    )
                    if isinstance(last_attempt, dict)
                    else "",
                    "last_failure_detail": self._truncate_text(
                        str(last_failure_detail), 260
                    )
                    if last_failure_detail
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
            if failure_detail:
                item["failure_detail"] = self._truncate_text(str(failure_detail), 260)
            summary.append(item)
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
            "tactic_library": self.state.scratch.get("tactic_library", ""),
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
        formatted = []
        for record in records:
            item = {
                "status": record.get("status"),
                "metric": record.get("metric"),
                "failed": record.get("failed"),
                "strategy_axis": record.get("strategy_axis", ""),
                "strategy_axes": record.get("strategy_axes", []),
                "changes": record.get("changes", []),
            }
            for key in (
                "no_change_reason",
                "failure_detail",
                "repair_parent_id",
                "artifact_id",
                "patch_path",
                "test_output_path",
            ):
                value = record.get(key)
                if value:
                    item[key] = (
                        self._truncate_text(str(value), 500)
                        if key in {"no_change_reason", "failure_detail"}
                        else value
                    )
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
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if failure_detail:
            extra["failure_detail"] = self._truncate_text(failure_detail, 2000)
        if no_change_reason:
            extra["no_change_reason"] = self._truncate_text(no_change_reason, 1000)
        if repair_parent_id:
            extra["repair_parent_id"] = repair_parent_id
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
            )
        )
        return extra

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
        messages = await self._target_not_found_repair_prompt(
            candidate,
            failure_detail=failure_detail,
            allowed=allowed,
        )
        try:
            decision = await self._target_not_found_repair_call(candidate, messages)
        except JsonValidationError as exc:
            self.state.notes.append(
                f"Candidate {candidate.candidate_id} target-not-found repair rejected: {exc}"
            )
            return None
        if not decision.candidates:
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
            self.state.notes.append(
                f"Candidate {candidate.candidate_id} target-not-found repair rejected: {note}"
            )
            return None
        return repaired

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
                "current source below and must match exactly."
            )
        else:
            system = (
                "Repair one failed CODE candidate. The previous candidate was rejected "
                "because a target string did not match the current source. Output strict "
                "JSON with a top-level candidates array containing exactly one candidate "
                "and one change. Do not invent a new tactic. Preserve the strategy_axis "
                "and todo context. The new target must be copied verbatim from the "
                "current source below and must match exactly."
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
            blocks.append(
                f"### {change.path}\n```text\n{self._slice_text(content, limit)}\n```"
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
            "changes": self._summarize_changes(candidate.changes),
            "todo_id": self._active_todo_id(),
        }
        if extra:
            record.update(
                {
                    key: value
                    for key, value in extra.items()
                    if value not in (None, "", [], {})
                }
            )
        if self._is_patch_application_failure_record(record):
            record["budget_counted"] = False
        with path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._append_todo_attempt(record)

    def _active_todo_id(self) -> str:
        if self._todo_contract_soft_now():
            return ""
        active_todo = self.state.scratch.get("active_todo")
        if isinstance(active_todo, dict) and active_todo.get("status") in {"active", "attempted"}:
            if self._todo_attempt_budget_exhausted(active_todo):
                return ""
            return str(active_todo.get("todo_id", ""))
        return ""

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
            "reason": candidate_record.get("reason"),
        }
        if candidate_record.get("budget_counted") is False:
            attempt["budget_counted"] = False
        for key in (
            "failure_detail",
            "no_change_reason",
            "artifact_id",
            "artifact_path",
            "repair_parent_id",
            "patch_path",
            "test_output_path",
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
        if status in {"improved", "accepted"}:
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

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

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
            output = await self._model_chat(role, messages, call_site="json_call")
        except Exception as exc:
            raise JsonValidationError(
                f"{role} model call failed: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            return self._parse_decision(output, schema)
        except JsonValidationError as exc:
            self._record_raw_model_output(role, "initial", output, exc)
            try:
                repaired = await self._model_chat(
                    role,
                    retry_repair_prompt(output, exc),
                    call_site="json_repair",
                )
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
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S%z')}] [local-micro-agent] {message}",
            flush=True,
        )


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
