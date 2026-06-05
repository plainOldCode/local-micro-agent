from __future__ import annotations

import argparse
import asyncio
import json
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
    def __init__(self, changes: list[CodeChange]):
        self.changes = changes


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
        if seeded_files:
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
            decision = await self._json_call("coder", code_prompt(self.state), CodeDecision)
        self.state.proposed_changes = decision.changes
        self.state.scratch["applied_changes"] = 0
        allowed = set(
            self.config.get("workflow", {}).get("writable_files") or self.state.planned_files
        )
        for change in decision.changes:
            if change.path not in allowed:
                self.state.notes.append(f"Rejected out-of-plan change: {change.path}")
                continue
            if change.target is not None and change.replacement is not None:
                if await self._apply_replacement(change.path, change.target, change.replacement):
                    self.state.scratch["applied_changes"] += 1
                continue
            if change.patch:
                if await self._apply_patch(change.patch):
                    self.state.scratch["applied_changes"] += 1
                continue
            if change.content is not None:
                await self.mcp.write_file(str(self.state.repo_root / change.path), change.content)
                self.state.scratch["applied_changes"] += 1
                continue
            self.state.notes.append(f"Skipped empty change: {change.path}")
        self.state.current = AgentStateName.TEST

    async def test(self) -> None:
        commands = self.config.get("workflow", {}).get("test_commands", [])
        self.state.test_results = []
        failed = False
        workflow = self.config.get("workflow", {})
        for command in commands:
            result = await self.mcp.run_command(
                command,
                cwd=str(self.state.repo_root),
                timeout_seconds=workflow.get("command_timeout_seconds", 120),
                output_limit=workflow.get("command_output_limit", 200_000),
            )
            self.state.test_results.append(TestResult(**result))
            failed = failed or result["exit_code"] != 0
        if self.state.scratch.get("applied_changes", 0) == 0:
            failed = True
            self.state.notes.append("No code changes were applied")
        if self.config.get("workflow", {}).get("deterministic_test_decision"):
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
