from __future__ import annotations

from .state import AgentState

DEFAULT_CHAR_LIMIT = 15000


PLAN_SYSTEM = """You are the PLAN node in a local coding-agent FSM.
Output only concise Markdown with:
1. Files to read or modify
2. Ordered implementation steps
3. Test commands
Do not write code. Do not include unrelated architecture discussion."""

READ_SYSTEM = """You are the READ node in a local coding-agent FSM.
Select the minimum source files needed for the plan.
Output strict JSON:
{"files":["relative/path.py"],"reason":"short reason"}
Do not include markdown or prose outside JSON."""

CODE_SYSTEM = """You are the CODE node in a local coding-agent FSM.
Use only the supplied plan, source files, and latest test failure.
Output strict JSON:
{"changes":[{"path":"relative/path.py","target":"exact existing text","replacement":"new text","reason":"why"}]}
Rules:
- Modify only listed files.
- Prefer exact target/replacement snippets.
- Use "patch" only if target/replacement is impossible.
- Use full-file "content" only for very small files.
- Preserve existing public behavior unless the plan says otherwise.
- No markdown fences, no commentary outside JSON."""

TEST_SYSTEM = """You are the TEST node in a local coding-agent FSM.
Given test output, decide whether the work is complete or needs another CODE loop.
Output strict JSON:
{"status":"pass|retry|fail","reason":"short reason","next_focus":"specific fix target"}
Do not include markdown or prose outside JSON."""


def plan_prompt(state: AgentState, project_context: str = "") -> list[dict[str, str]]:
    user_content = state.user_request
    if project_context:
        user_content = f"{state.user_request}\n\nProject context:\n{project_context}"
    return [
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def read_prompt(state: AgentState) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": READ_SYSTEM},
        {"role": "user", "content": f"Plan:\n{state.plan_markdown}"},
    ]


def code_prompt(state: AgentState) -> list[dict[str, str]]:
    source_blocks = "\n\n".join(
        f"### {snap.path}\n```text\n{slice_text(snap.content)}\n```" for snap in state.file_context
    )
    return [
        {"role": "system", "content": CODE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"User request:\n{state.user_request}\n\n"
                f"Plan:\n{state.plan_markdown}\n\n"
                f"Latest test summary:\n{state.latest_test_summary()}\n\n"
                f"Source files:\n{source_blocks}"
            ),
        },
    ]


def test_prompt(state: AgentState) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": TEST_SYSTEM},
        {"role": "user", "content": state.latest_test_summary()},
    ]


PROMPT_MARKDOWN = {
    "PLAN": PLAN_SYSTEM,
    "READ": READ_SYSTEM,
    "CODE": CODE_SYSTEM,
    "TEST": TEST_SYSTEM,
}


def slice_text(text: str, limit: int = DEFAULT_CHAR_LIMIT) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + "\n[...truncated...]\n" + text[-tail:]
