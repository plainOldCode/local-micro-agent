"""Model call layer: chat calls, JSON repair, token budgets, response normalization.

Extracted from orchestrator.py; mixed into MicroAgent.
"""
from __future__ import annotations

import time
from typing import Any

from .decisions import CodeCandidate, CodeDecision, ReadDecision, TestDecision
from .models import ModelResponse
from .state import CodeChange
from .validators import (
    JsonValidationError,
    parse_json_object,
    parse_xml_candidates,
    require_keys,
    retry_repair_prompt,
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
                "success": True,
                **stream_stats,
                **usage_fields,
                **budget_fields,
            },
        )
        if self._reject_reasoning_only_response(output, usage, role=role, call_site=call_site):
            raise RuntimeError(
                "Model returned reasoning-only response with empty final content"
            )
        return output

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

