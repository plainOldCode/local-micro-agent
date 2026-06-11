from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class AgentStateName(StrEnum):
    PLAN = "plan"
    SPEC_SYNTH = "spec_synth"
    SCHEDULE = "schedule"
    TASK_READ = "task_read"
    ACCEPT_SYNTH = "accept_synth"
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
    target_region: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    anchor_before: str | None = None
    anchor_after: str | None = None
    target_hash: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeChange":
        return cls(
            path=str(data["path"]),
            reason=str(data.get("reason", "")),
            content=data.get("content"),
            patch=data.get("patch"),
            target=data.get("target"),
            replacement=data.get("replacement"),
            target_region=data.get("target_region"),
            start_line=cls._optional_int(data.get("start_line")),
            end_line=cls._optional_int(data.get("end_line")),
            anchor_before=data.get("anchor_before"),
            anchor_after=data.get("anchor_after"),
            target_hash=data.get("target_hash"),
        )

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


@dataclass
class TestResult:
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""


DEFAULT_MAX_LOOPS = 3


@dataclass
class AgentState:
    """The single state bag carried through the FSM."""

    repo_root: Path
    user_request: str
    current: AgentStateName = AgentStateName.PLAN
    loop_count: int = 0
    fsm_step_count: int = 0
    max_loops: int | None = None
    max_loops_defaulted: bool = field(default=True, init=False, repr=False)
    plan_markdown: str = ""
    planned_files: list[str] = field(default_factory=list)
    file_context: list[FileSnapshot] = field(default_factory=list)
    external_context: list[ExternalContext] = field(default_factory=list)
    proposed_changes: list[CodeChange] = field(default_factory=list)
    test_results: list[TestResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    scratch: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_loops is None:
            self.max_loops = DEFAULT_MAX_LOOPS
            self.max_loops_defaulted = True
        else:
            self.max_loops_defaulted = False

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
        data.pop("max_loops_defaulted", None)
        return data
