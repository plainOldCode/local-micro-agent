from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from .mcp_client import McpServerSpec, McpToolClient
from .models import ModelManager
from .prompts import PROMPT_MARKDOWN, code_prompt, plan_prompt, read_prompt, test_prompt
from .state import AgentState, AgentStateName, CodeChange, FileSnapshot, TestResult
from .validators import JsonValidationError, parse_model_json, retry_repair_prompt


class ReadDecision(BaseModel):
    files: list[str]
    reason: str


class CodeDecision(BaseModel):
    changes: list[CodeChange]


class TestDecision(BaseModel):
    status: str
    reason: str
    next_focus: str = ""


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
                    await self.plan()
                elif self.state.current == AgentStateName.READ:
                    await self.read()
                elif self.state.current == AgentStateName.CODE:
                    await self.code()
                elif self.state.current == AgentStateName.TEST:
                    await self.test()
                else:
                    self.state.current = AgentStateName.FAILED
        finally:
            await self.mcp.close()
        return self.state

    async def plan(self) -> None:
        output = await self.models.get("planner").chat(plan_prompt(self.state))
        self.state.plan_markdown = output.strip()
        self.state.current = AgentStateName.READ

    async def read(self) -> None:
        decision = await self._json_call("planner", read_prompt(self.state), ReadDecision)
        self.state.planned_files = decision.files
        self.state.file_context = []
        for rel_path in decision.files:
            abs_path = self.state.repo_root / rel_path
            content = await self.mcp.read_file(str(abs_path))
            self.state.file_context.append(FileSnapshot(path=rel_path, content=content))
        self.state.current = AgentStateName.CODE

    async def code(self) -> None:
        decision = await self._json_call("coder", code_prompt(self.state), CodeDecision)
        self.state.proposed_changes = decision.changes
        allowed = set(self.state.planned_files)
        for change in decision.changes:
            if change.path not in allowed:
                self.state.notes.append(f"Rejected out-of-plan change: {change.path}")
                continue
            if change.content is None:
                self.state.notes.append(f"Skipped non-content patch skeleton: {change.path}")
                continue
            await self.mcp.write_file(str(self.state.repo_root / change.path), change.content)
        self.state.current = AgentStateName.TEST

    async def test(self) -> None:
        commands = self.config.get("workflow", {}).get("test_commands", [])
        self.state.test_results = []
        failed = False
        for command in commands:
            result = await self.mcp.run_command(command, cwd=str(self.state.repo_root))
            self.state.test_results.append(TestResult(**result))
            failed = failed or result["exit_code"] != 0

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

    async def _json_call(self, role: str, messages: list[dict[str, str]], schema: type[BaseModel]):
        output = await self.models.get(role).chat(messages)
        try:
            return parse_model_json(output, schema)
        except JsonValidationError as exc:
            repaired = await self.models.get(role).chat(retry_repair_prompt(output, exc))
            return parse_model_json(repaired, schema)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


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
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
