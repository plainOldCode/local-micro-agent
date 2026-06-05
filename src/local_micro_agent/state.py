from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AgentStateName(StrEnum):
    PLAN = "plan"
    READ = "read"
    CODE = "code"
    TEST = "test"
    DONE = "done"
    FAILED = "failed"


class FileSnapshot(BaseModel):
    path: str
    content: str


class CodeChange(BaseModel):
    path: str
    content: str | None = None
    patch: str | None = None
    reason: str


class TestResult(BaseModel):
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""


class AgentState(BaseModel):
    """The single state bag carried through the FSM."""

    repo_root: Path
    user_request: str
    current: AgentStateName = AgentStateName.PLAN
    loop_count: int = 0
    max_loops: int = 3

    plan_markdown: str = ""
    planned_files: list[str] = Field(default_factory=list)
    file_context: list[FileSnapshot] = Field(default_factory=list)
    proposed_changes: list[CodeChange] = Field(default_factory=list)
    test_results: list[TestResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    scratch: dict[str, Any] = Field(default_factory=dict)

    def latest_test_summary(self) -> str:
        if not self.test_results:
            return "No tests have run yet."
        last = self.test_results[-1]
        return (
            f"command={last.command!r}\n"
            f"exit_code={last.exit_code}\n"
            f"stdout_tail={last.stdout[-2000:]}\n"
            f"stderr_tail={last.stderr[-2000:]}"
        )
