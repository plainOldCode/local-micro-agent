"""Profiling spans, model stream artifacts, and structured run logging.

Extracted from orchestrator.py; mixed into MicroAgent.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any



class TelemetryMixin:
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

    @staticmethod
    def _profile_model_usage_fields(
        usage: dict[str, Any], elapsed_seconds: float
    ) -> dict[str, Any]:
        if not usage:
            return {}
        fields: dict[str, Any] = {}

        def numeric(key: str) -> int | float | None:
            value = usage.get(key)
            if isinstance(value, (int, float)) and value >= 0:
                return value
            return None

        prompt_tokens = numeric("prompt_tokens")
        completion_tokens = numeric("completion_tokens")
        total_tokens = numeric("total_tokens")
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        for key, value in (
            ("prompt_tokens", prompt_tokens),
            ("completion_tokens", completion_tokens),
            ("total_tokens", total_tokens),
        ):
            if value is not None:
                fields[key] = int(value)

        duration_fields = {
            "provider_prompt_eval_duration_ns": numeric("provider_prompt_eval_duration_ns"),
            "provider_eval_duration_ns": numeric("provider_eval_duration_ns"),
            "provider_total_duration_ns": numeric("provider_total_duration_ns"),
        }
        for key, value in duration_fields.items():
            if value is not None:
                fields[key] = int(value)
                fields[key.removesuffix("_ns") + "_ms"] = round(value / 1_000_000, 3)

        for key in ("provider_prompt_eval_count", "provider_eval_count"):
            value = numeric(key)
            if value is not None:
                fields[key] = int(value)

        reasoning_chars = numeric("reasoning_content_chars")
        if reasoning_chars is not None:
            fields["reasoning_content_chars"] = int(reasoning_chars)
        reasoning_only = usage.get("reasoning_only_response")
        if isinstance(reasoning_only, bool):
            fields["reasoning_only_response"] = reasoning_only

        prompt_duration = duration_fields["provider_prompt_eval_duration_ns"]
        completion_duration = duration_fields["provider_eval_duration_ns"]
        total_duration = duration_fields["provider_total_duration_ns"]
        if prompt_tokens is not None and prompt_duration and prompt_duration > 0:
            fields["prompt_tokens_per_second"] = round(
                prompt_tokens / (prompt_duration / 1_000_000_000), 3
            )
        if completion_tokens is not None and completion_duration and completion_duration > 0:
            fields["completion_tokens_per_second"] = round(
                completion_tokens / (completion_duration / 1_000_000_000), 3
            )
        if total_tokens is not None and total_duration and total_duration > 0:
            fields["total_tokens_per_second"] = round(
                total_tokens / (total_duration / 1_000_000_000), 3
            )
        if total_tokens is not None and elapsed_seconds > 0:
            fields["wall_tokens_per_second"] = round(total_tokens / elapsed_seconds, 3)
        return fields

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
        reasoning_stream_path = stream_dir / ("-".join(label_parts) + ".reasoning.txt")
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        stream_path.write_text("")
        interval = int(workflow.get("profile_model_stream_log_interval_chars", 2000) or 0)
        stats: dict[str, Any] = {
            "streaming": True,
            "stream_path": self._repo_relative_path(stream_path),
            "stream_chunks": 0,
            "stream_chars": 0,
            "reasoning_stream_path": self._repo_relative_path(reasoning_stream_path),
            "reasoning_stream_chunks": 0,
            "reasoning_stream_chars": 0,
        }
        next_log_at = {"value": interval}
        self._log(
            "STREAM start "
            f"role={role} call_site={call_site or 'chat'} model={model_name} "
            f"provider={provider.get('kind', '')} path={stats['stream_path']}"
        )

        def on_chunk(chunk: Any) -> None:
            chunk_kind = "content"
            if isinstance(chunk, dict):
                chunk_kind = str(chunk.get("kind") or "content")
                chunk = str(chunk.get("content") or "")
            if not chunk:
                return
            if chunk_kind == "reasoning":
                with reasoning_stream_path.open("a") as handle:
                    handle.write(chunk)
                stats["reasoning_stream_chunks"] = (
                    int(stats.get("reasoning_stream_chunks", 0)) + 1
                )
                stats["reasoning_stream_chars"] = (
                    int(stats.get("reasoning_stream_chars", 0)) + len(chunk)
                )
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

    @staticmethod
    def _log(message: str) -> None:
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S%z')}] [local-micro-agent] {message}",
            flush=True,
        )
