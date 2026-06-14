"""Model call layer: chat calls, JSON repair, token budgets, response normalization.

Extracted from orchestrator.py; mixed into MicroAgent.
"""
from __future__ import annotations

import time
from typing import Any

from ..candidate_repair import candidate_repair_call_site, candidate_repair_prompt
from ..decisions import CodeCandidate, CodeDecision, ReadDecision, TestDecision
from ..models import ModelResponse, ModelTextParts
from ..state import CodeChange
from ..validators import (
    JsonValidationError,
    parse_json_object,
    parse_xml_candidates,
    require_keys,
)


class ModelRuntimeMixin:
    async def _model_chat(
        self, role: str, messages: list[dict[str, str]], call_site: str = ""
    ) -> str:
        start = self._profile_span_start()
        requested_role = role
        role = self._model_role_for_call_site(role, call_site)
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
                response = await model.chat(messages, stream_callback=stream_callback)
            else:
                response = await model.chat(messages)
        except Exception as exc:
            self._record_profile_span(
                "model_call",
                start,
                {
                    "role": role,
                    **({"requested_role": requested_role} if requested_role != role else {}),
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
        output, usage = self._normalize_model_response(response)
        elapsed_seconds = max(time.perf_counter() - start["perf"], 0.0)
        usage_fields = self._profile_model_usage_fields(usage, elapsed_seconds)
        budget_fields = self._model_token_budget_fields(
            provider,
            usage,
            prompt_chars=prompt_chars,
            role=role,
            call_site=call_site,
        )
        rejected_reasoning_only = self._reject_reasoning_only_response(
            output, usage, role=role, call_site=call_site
        )
        self._record_profile_span(
            "model_call",
            start,
            {
                "role": role,
                **({"requested_role": requested_role} if requested_role != role else {}),
                "call_site": call_site,
                "model_name": model_name,
                "provider_kind": provider.get("kind"),
                "provider_model": provider.get("model"),
                "message_count": len(messages),
                "prompt_chars": prompt_chars,
                "output_chars": len(output),
                "success": not rejected_reasoning_only,
                **(
                    {
                        "rejected_reasoning_only": True,
                        "error": "Model returned reasoning-only response with empty final content",
                    }
                    if rejected_reasoning_only
                    else {}
                ),
                **stream_stats,
                **usage_fields,
                **budget_fields,
            },
        )
        if rejected_reasoning_only:
            raise RuntimeError(
                "Model returned reasoning-only response with empty final content"
            )
        return output

    async def _model_thinking_brief(
        self,
        role: str,
        messages: list[dict[str, str]],
        call_site: str = "spec_think_brief",
    ) -> ModelTextParts:
        start = self._profile_span_start()
        requested_role = role
        role = self._model_role_for_call_site(role, call_site)
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
                response = await model.chat(messages, stream_callback=stream_callback)
            else:
                response = await model.chat(messages)
        except Exception as exc:
            self._record_profile_span(
                "model_call",
                start,
                {
                    "role": role,
                    **({"requested_role": requested_role} if requested_role != role else {}),
                    "call_site": call_site,
                    "model_name": model_name,
                    "provider_kind": provider.get("kind"),
                    "provider_model": provider.get("model"),
                    "message_count": len(messages),
                    "prompt_chars": prompt_chars,
                    "success": False,
                    "thinking_brief": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    **stream_stats,
                },
            )
            raise
        parts = self._normalize_model_text_parts(response)
        selected = self._model_text_selected_source(parts)
        elapsed_seconds = max(time.perf_counter() - start["perf"], 0.0)
        usage_fields = self._profile_model_usage_fields(parts.usage, elapsed_seconds)
        budget_fields = self._model_token_budget_fields(
            provider,
            parts.usage,
            prompt_chars=prompt_chars,
            role=role,
            call_site=call_site,
        )
        selected_text = parts.content if selected == "content" else parts.reasoning
        self._record_profile_span(
            "model_call",
            start,
            {
                "role": role,
                **({"requested_role": requested_role} if requested_role != role else {}),
                "call_site": call_site,
                "model_name": model_name,
                "provider_kind": provider.get("kind"),
                "provider_model": provider.get("model"),
                "message_count": len(messages),
                "prompt_chars": prompt_chars,
                "output_chars": len(parts.content),
                "thinking_brief": True,
                "thinking_brief_selected_source": selected,
                "thinking_brief_chars": len(selected_text),
                "reasoning_only_accepted": selected == "reasoning",
                "success": True,
                **stream_stats,
                **usage_fields,
                **budget_fields,
            },
        )
        usage = dict(parts.usage)
        usage.update(
            {
                "thinking_brief_selected_source": selected,
                "thinking_brief_chars": len(selected_text),
                "provider_kind": provider.get("kind"),
                "provider_model": provider.get("model"),
                "model_name": model_name,
                "role": role,
                "call_site": call_site,
            }
        )
        return ModelTextParts(
            content=parts.content,
            reasoning=parts.reasoning,
            usage=usage,
            source=selected,
        )

    def _reject_reasoning_only_response(
        self,
        output: str,
        usage: dict[str, Any],
        *,
        role: str,
        call_site: str,
    ) -> bool:
        if output.strip():
            return False
        if usage.get("reasoning_only_response") is not True:
            return False
        workflow = self.config.get("workflow", {})
        if workflow.get("allow_reasoning_only_response"):
            return False
        allowed = workflow.get("reasoning_only_allowed_call_sites", [])
        if isinstance(allowed, str):
            allowed = [allowed]
        if call_site in {str(item) for item in allowed if str(item).strip()}:
            return False
        self.state.notes.append(
            "Rejected reasoning-only model response with empty content: "
            f"role={role} call_site={call_site}"
        )
        return True

    def _model_role_for_call_site(self, role: str, call_site: str) -> str:
        workflow = self.config.get("workflow", {})
        deep_role = self._deep_reasoning_role_for_call_site(role, call_site)
        if deep_role is not None:
            return deep_role
        override_role = self._model_role_override_for_call_site(call_site)
        if override_role is not None:
            return override_role
        if not workflow.get("reasoning_lane_enabled", False):
            return role
        call_sites = workflow.get(
            "reasoning_lane_call_sites", ["plan", "semantic_analysis", "reflect"]
        )
        if not isinstance(call_sites, list):
            call_sites = []
        if call_site not in {str(item) for item in call_sites}:
            return role
        excluded_roles = workflow.get(
            "reasoning_lane_excluded_roles", ["coder", "brainstorm", "tester"]
        )
        if not isinstance(excluded_roles, list):
            excluded_roles = []
        if role in {str(item) for item in excluded_roles}:
            return role
        reasoning_role = str(workflow.get("reasoning_lane_model_role", "reasoner") or "")
        if not reasoning_role or reasoning_role == role:
            return role
        if reasoning_role not in self.config.get("models", {}):
            return role
        return reasoning_role

    def _model_role_override_for_call_site(self, call_site: str) -> str | None:
        if not call_site:
            return None
        workflow = self.config.get("workflow", {})
        overrides = workflow.get("model_role_overrides_by_call_site", {})
        if not isinstance(overrides, dict):
            return None
        override = str(overrides.get(call_site) or "").strip()
        if not override or override not in self.config.get("models", {}):
            return None
        return override

    def _deep_reasoning_role_for_call_site(
        self, role: str, call_site: str
    ) -> str | None:
        workflow = self.config.get("workflow", {})
        if not workflow.get("deep_reasoning_enabled", False):
            return None
        call_sites = workflow.get("deep_reasoning_call_sites", ["reflect"])
        if isinstance(call_sites, str):
            call_sites = [call_sites]
        if call_site not in {str(item) for item in call_sites if str(item).strip()}:
            return None
        excluded_roles = workflow.get(
            "deep_reasoning_excluded_roles", ["coder", "brainstorm", "tester"]
        )
        if isinstance(excluded_roles, str):
            excluded_roles = [excluded_roles]
        if role in {str(item) for item in excluded_roles if str(item).strip()}:
            return None
        deep_role = str(
            workflow.get("deep_reasoning_model_role")
            or workflow.get("reasoning_lane_model_role")
            or "reasoner"
        ).strip()
        if not deep_role or deep_role not in self.config.get("models", {}):
            return None
        if not self._deep_reasoning_triggered(call_site):
            return None
        self.state.notes.append(
            f"Escalating model call to deep reasoning: call_site={call_site} role={deep_role}"
        )
        return deep_role

    def _deep_reasoning_triggered(self, call_site: str) -> bool:
        workflow = self.config.get("workflow", {})
        if bool(workflow.get("deep_reasoning_always")):
            return True
        failure_threshold = int(
            workflow.get("deep_reasoning_after_same_failure_class", 0) or 0
        )
        if failure_threshold > 0:
            counts = self.state.scratch.get("retry_failure_class_counts")
            if isinstance(counts, dict):
                for value in counts.values():
                    try:
                        if int(value) >= failure_threshold:
                            return True
                    except (TypeError, ValueError):
                        continue
        no_improvement_threshold = int(
            workflow.get("deep_reasoning_after_no_improvement_loops", 0) or 0
        )
        invariant_threshold = int(
            workflow.get("deep_reasoning_after_invariant_failures", 0) or 0
        )
        if no_improvement_threshold <= 0 and invariant_threshold <= 0:
            return False
        records_method = getattr(self, "_candidate_history_records", None)
        if not callable(records_method):
            return False
        limit = max(no_improvement_threshold, invariant_threshold, 1)
        records = records_method(limit=limit)
        if no_improvement_threshold > 0 and len(records) >= no_improvement_threshold:
            baseline = self.config.get("workflow", {}).get("baseline_metric")
            improved = False
            for record in records[-no_improvement_threshold:]:
                metric = record.get("metric")
                if isinstance(metric, (int, float)) and isinstance(baseline, (int, float)):
                    improved = metric < baseline
                elif str(record.get("status", "")) in {"accepted", "improved"}:
                    improved = True
                if improved:
                    break
            if not improved:
                return True
        if invariant_threshold > 0 and len(records) >= invariant_threshold:
            recent = records[-invariant_threshold:]
            hard_failures = {"correctness_failure", "invariant_broken"}
            if all(str(record.get("failure_class", "")) in hard_failures for record in recent):
                return True
        return False

    def _provider_for_role(self, role: str) -> dict[str, Any]:
        model_name = self.config.get("models", {}).get(role) or self.config.get(
            "models", {}
        ).get("default")
        provider = self.config.get("providers", {}).get(str(model_name), {})
        return provider if isinstance(provider, dict) else {}

    def _input_token_budget(self, provider: dict[str, Any]) -> int | None:
        num_ctx = provider.get("num_ctx")
        max_tokens = provider.get("max_tokens")
        if not isinstance(num_ctx, int) or num_ctx <= 0:
            return None
        if not isinstance(max_tokens, int) or max_tokens < 0:
            max_tokens = 0
        return max(num_ctx - max_tokens, 1)

    def _prompt_chars_per_token(self) -> float:
        return float(
            self.config.get("workflow", {}).get("prompt_chars_per_token_estimate", 3.5)
            or 3.5
        )

    def _model_token_budget_fields(
        self,
        provider: dict[str, Any],
        usage: dict[str, Any],
        *,
        prompt_chars: int,
        role: str,
        call_site: str,
    ) -> dict[str, Any]:
        input_budget = self._input_token_budget(provider)
        if input_budget is None:
            return {}
        prompt_tokens = usage.get("prompt_tokens")
        estimated = False
        if not isinstance(prompt_tokens, (int, float)) or prompt_tokens < 0:
            prompt_tokens = prompt_chars / self._prompt_chars_per_token()
            estimated = True
        ratio = float(prompt_tokens) / float(input_budget)
        fields: dict[str, Any] = {
            "input_token_budget": input_budget,
            "input_token_budget_used_ratio": round(ratio, 4),
        }
        if estimated:
            fields["prompt_tokens_estimated"] = int(prompt_tokens)
        warn_ratio = float(
            self.config.get("workflow", {}).get("prompt_token_budget_warn_ratio", 0.9)
            or 0.9
        )
        if ratio >= warn_ratio:
            fields["input_token_budget_warning"] = True
            note = (
                "Prompt token budget pressure: "
                f"role={role} call_site={call_site or 'chat'} "
                f"prompt_tokens={'~' if estimated else ''}{int(prompt_tokens)} "
                f"budget={input_budget} ratio={ratio:.2f}"
            )
            self._record_prompt_budget_warning("pressure", note)
            self._log(note)
        return fields

    def _record_prompt_budget_warning(self, kind: str, note: str) -> None:
        warnings = self.state.scratch.setdefault("prompt_token_budget_warnings", [])
        if isinstance(warnings, list):
            warnings.append(
                {
                    "loop": self.state.loop_count,
                    "kind": kind,
                    "note": note,
                }
            )
        else:
            self.state.scratch["prompt_token_budget_warnings"] = [
                {
                    "loop": self.state.loop_count,
                    "kind": kind,
                    "note": note,
                }
            ]
        note_keys = self.state.scratch.setdefault("prompt_token_budget_note_keys", [])
        if not isinstance(note_keys, list):
            note_keys = []
            self.state.scratch["prompt_token_budget_note_keys"] = note_keys
        key = f"{self.state.loop_count}:{kind}"
        if key not in note_keys:
            self.state.notes.append(note)
            note_keys.append(key)

    def _shrink_dynamic_suffix_blocks(
        self,
        stable_messages: list[dict[str, str]],
        dynamic_blocks: list[str],
        *,
        role: str,
    ) -> list[str]:
        workflow = self.config.get("workflow", {})
        if workflow.get("auto_shrink_dynamic_context", True) is False:
            return dynamic_blocks
        if not dynamic_blocks:
            return dynamic_blocks
        provider = self._provider_for_role(role)
        input_budget = self._input_token_budget(provider)
        if input_budget is None:
            return dynamic_blocks
        target_ratio = float(workflow.get("prompt_token_budget_target_ratio", 0.85) or 0.85)
        char_budget = int(input_budget * self._prompt_chars_per_token() * target_ratio)
        stable_chars = sum(len(str(message.get("content", ""))) for message in stable_messages)
        dynamic_budget = char_budget - stable_chars
        dynamic_chars = sum(len(block) for block in dynamic_blocks)
        if dynamic_budget <= 0 or dynamic_chars <= dynamic_budget:
            return dynamic_blocks
        per_block = max(500, dynamic_budget // max(len(dynamic_blocks), 1))
        self._record_prompt_budget_warning(
            "shrink",
            "Shrank dynamic CODE context for prompt budget: "
            f"dynamic_chars={dynamic_chars} budget={dynamic_budget}",
        )
        return [self._slice_text(block, per_block) for block in dynamic_blocks]

    @staticmethod
    def _normalize_model_response(response: Any) -> tuple[str, dict[str, Any]]:
        if isinstance(response, ModelResponse):
            return response.content, dict(response.usage)
        return str(response), {}

    @staticmethod
    def _normalize_model_text_parts(response: Any) -> ModelTextParts:
        if isinstance(response, ModelTextParts):
            return response
        if isinstance(response, ModelResponse):
            content = response.content
            reasoning = response.reasoning
            if content.strip() and reasoning.strip():
                source = "mixed"
            elif content.strip():
                source = "content"
            elif reasoning.strip():
                source = "reasoning"
            else:
                source = "empty"
            return ModelTextParts(
                content=content,
                reasoning=reasoning,
                usage=dict(response.usage),
                source=source,
            )
        text = str(response)
        return ModelTextParts(
            content=text,
            reasoning="",
            usage={},
            source="content" if text.strip() else "empty",
        )

    @staticmethod
    def _model_text_selected_source(parts: ModelTextParts) -> str:
        if parts.content.strip():
            return "content"
        if parts.reasoning.strip():
            return "reasoning"
        return "empty"

    async def _json_call(
        self,
        role: str,
        messages: list[dict[str, str]],
        schema: type,
        output_format: str = "json",
    ):
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
            repair_base = "candidate_output" if schema is CodeDecision else "json"
            try:
                repaired = await self._model_chat(
                    role,
                    candidate_repair_prompt(output_format, output, exc),
                    call_site=candidate_repair_call_site(output_format, repair_base),
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
