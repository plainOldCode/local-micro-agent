from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class AgentStateName(StrEnum):
    PLAN = "plan"
    READ = "read"
    REFLECT = "reflect"
    CODE = "code"
    TEST = "test"
    DONE = "done"
    FAILED = "failed"


@dataclass
class FileSnapshot:
    path: str
    content: str


@dataclass
class ExternalContext:
    kind: str
    source: str
    title: str
    content: str
    sha256: str
    trust: str = "advisory"
    fetched_at: str | None = None


@dataclass
class CodeChange:
    path: str
    reason: str
    content: str | None = None
    patch: str | None = None
    target: str | None = None
    replacement: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeChange":
        return cls(
            path=str(data["path"]),
            reason=str(data.get("reason", "")),
            content=data.get("content"),
            patch=data.get("patch"),
            target=data.get("target"),
            replacement=data.get("replacement"),
        )


@dataclass
class TestResult:
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""


@dataclass
class AgentState:
    """The single state bag carried through the FSM."""

    repo_root: Path
    user_request: str
    current: AgentStateName = AgentStateName.PLAN
    loop_count: int = 0
    max_loops: int = 3
    plan_markdown: str = ""
    planned_files: list[str] = field(default_factory=list)
    file_context: list[FileSnapshot] = field(default_factory=list)
    external_context: list[ExternalContext] = field(default_factory=list)
    proposed_changes: list[CodeChange] = field(default_factory=list)
    test_results: list[TestResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    scratch: dict[str, Any] = field(default_factory=dict)

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

    def recent_notes_summary(self, limit: int = 12) -> str:
        if not self.notes:
            return "No agent feedback yet."
        return "\n".join(f"- {note}" for note in self.notes[-limit:])

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["repo_root"] = str(self.repo_root)
        data["current"] = str(self.current)
        return data
