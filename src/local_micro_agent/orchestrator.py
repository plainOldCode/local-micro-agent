from __future__ import annotations

import argparse
import asyncio
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from .mcp_client import McpServerSpec, McpToolClient
from .models import ModelManager
from .prompts import PROMPT_MARKDOWN, code_prompt, plan_prompt, read_prompt, test_prompt
from .state import AgentState, AgentStateName, CodeChange, FileSnapshot, TestResult
from .validators import JsonValidationError, parse_json_object, require_keys, retry_repair_prompt


class ReadDecision:
    def __init__(self, files: list[str], reason: str = ""):
        self.files = files
        self.reason = reason


class CodeDecision:
    def __init__(self, changes: list[CodeChange], candidates: list["CodeCandidate"] | None = None):
        self.changes = changes
        self.candidates = candidates or [CodeCandidate("1", changes, "single candidate")]


class CodeCandidate:
    def __init__(self, candidate_id: str, changes: list[CodeChange], reason: str = ""):
        self.candidate_id = candidate_id
        self.changes = changes
        self.reason = reason


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

        output = await self.models.get("planner").chat(plan_prompt(self.state))
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
            self.state.file_context.append(FileSnapshot(path=rel_path, content=content))
        self.state.current = AgentStateName.CODE

    async def code(self) -> None:
        seeded_changes = self.config.get("workflow", {}).get("seed_changes")
        if seeded_changes:
            decision = CodeDecision(changes=[CodeChange.from_dict(c) for c in seeded_changes])
        else:
            try:
                messages = code_prompt(self.state)
                if self.config.get("workflow", {}).get("candidate_queue"):
                    messages = [
                        *messages,
                        {
                            "role": "system",
                            "content": (
                                "Candidate queue mode is enabled. Output strict JSON with a top-level "
                                '"candidates" array, not a top-level "changes" array. Example: '
                                '{"candidates":[{"id":"1","reason":"short","changes":[{"path":"file.py",'
                                '"target":"exact text","replacement":"new text","reason":"short"}]}]}. '
                                "Each candidate must be independent and safe to apply from the same baseline."
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
            await self._restore_snapshot(baseline_snapshot)
            applied = await self._apply_changes(candidate.changes, allowed)
            if applied == 0:
                self.state.notes.append(f"Candidate {candidate.candidate_id} rejected: no changes applied")
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
            if failed or not improved:
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
        self.state.notes.append(f"Candidate queue accepted metric={iteration_best_metric}")

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
        if self.config.get("workflow", {}).get("deterministic_test_decision"):
            if failed and self._should_retry_rejected_candidate():
                self.state.loop_count += 1
                self.state.current = AgentStateName.CODE
                return
            self.state.current = AgentStateName.FAILED if failed else AgentStateName.DONE
            return

        decision = await self._json_call("tester", test_prompt(self.state), TestDecision)
        if not failed and decision.status == "pass":
            self.state.current = AgentStateName.DONE
            return

        self.state.loop_count += 1
        if self.state.loop_count >= self.state.max_loops or decision.status == "fail":
            self.state.current = AgentStateName.FAILED
            return

        self.state.notes.append(f"Retry focus: {decision.next_focus or decision.reason}")
        self.state.current = AgentStateName.CODE

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
        output = await self.models.get(role).chat(messages)
        try:
            return self._parse_decision(output, schema)
        except JsonValidationError as exc:
            repaired = await self.models.get(role).chat(retry_repair_prompt(output, exc))
            return self._parse_decision(repaired, schema)

    @staticmethod
    def _parse_decision(output: str, schema: type):
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
        original = await self.mcp.read_file(str(abs_path))
        if target not in original:
            self.state.notes.append(f"Replacement target not found: {path}")
            return False
        if original.count(target) != 1:
            self.state.notes.append(f"Replacement target is ambiguous: {path}")
            return False
        await self.mcp.write_file(str(abs_path), original.replace(target, replacement, 1))
        return True

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
