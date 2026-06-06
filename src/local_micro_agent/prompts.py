from __future__ import annotations

from .state import AgentState

DEFAULT_CHAR_LIMIT = 15000


PLAN_SYSTEM = """You are the PLAN node in a local coding-agent FSM.
Output only concise Markdown with:
1. Files to read or modify
2. Ordered implementation steps
3. Test commands
Do not write code. Do not include unrelated architecture discussion.
Respect project instructions and workflow constraints before giving generic
advice. If writable files are constrained, do not plan modifications outside
that set. Do not modify tests unless the user explicitly asks for test changes.
Prefer reading source entrypoints named by the README or task text before
choosing implementation changes."""

READ_SYSTEM = """You are the READ node in a local coding-agent FSM.
Select the minimum source files needed for the plan.
Output strict JSON:
{"files":["relative/path.py"],"reason":"short reason"}
Do not include markdown or prose outside JSON."""

REFLECT_SYSTEM = """You are the REFLECT node in a local coding-agent FSM.
Do not write code. Analyze only the latest rejected attempt and feedback.
Output exactly 1-3 concise Markdown bullets:
- why the previous attempt failed
- what must change in the next CODE attempt
- what pattern must not be repeated"""

BRAINSTORM_SYSTEM = """You are the BRAINSTORM node in a local coding-agent FSM.
The search is stuck in a local minimum. Do not write code.
Output exactly 3 numbered tactics in Markdown.
Each tactic must:
- be a different algorithmic or architectural paradigm
- avoid repeating the rejected patterns
- name the most relevant strategy_axis
- include one concrete implementation hook in the supplied source
Keep each tactic to 2 short sentences."""

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
- Do not output comment-only, formatting-only, or explanatory placeholder changes.
- No markdown fences, no commentary outside JSON."""

CODE_XML_SYSTEM = """You are the CODE node in a local coding-agent FSM.
Use only the supplied plan, source files, and latest test failure.
Do not output JSON. Output exactly one small candidate in this XML-like format:
<candidates>
<candidate id="1">
<strategy_axis>one_known_axis</strategy_axis>
<reason>one short sentence</reason>
<change>
<path>relative/path.py</path>
<search>
exact existing code, copied verbatim; 1-40 lines only
</search>
<replace>
new code, copied verbatim; 1-40 lines only
</replace>
</change>
</candidate>
</candidates>
Rules:
- Modify only listed files.
- Emit exactly one <candidate> and exactly one <change>.
- Include exactly one <strategy_axis> tag inside <candidate>.
- Keep <reason> to one sentence.
- Keep <search> and <replace> under 40 lines each.
- Never replace an entire function or class.
- Prefer a tiny local edit around the immediate bottleneck.
- The <search> block must match existing code exactly, including whitespace.
- Put raw code inside <search> and <replace>; do not JSON-escape quotes or newlines.
- Do not add markdown fences or prose outside <candidates>.
- Do not output comment-only, formatting-only, or explanatory placeholder changes."""

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


def reflect_prompt(state: AgentState, feedback_notes_limit: int = 12) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REFLECT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"User request:\n{state.user_request}\n\n"
                f"Plan:\n{state.plan_markdown}\n\n"
                f"Latest test summary:\n{state.latest_test_summary()}\n\n"
                f"Recent agent feedback:\n{state.recent_notes_summary(feedback_notes_limit)}"
            ),
        },
    ]


def brainstorm_prompt(
    state: AgentState, reject_summary: str, cooled_axes: list[str], feedback_notes_limit: int = 8
) -> list[dict[str, str]]:
    source_blocks = "\n\n".join(
        f"### {snap.path}\n```text\n{slice_text(snap.content)}\n```" for snap in state.file_context
    )
    return [
        {"role": "system", "content": BRAINSTORM_SYSTEM},
        {
            "role": "user",
            "content": (
                f"User request:\n{state.user_request}\n\n"
                f"Plan:\n{state.plan_markdown}\n\n"
                f"Source files:\n{source_blocks}\n\n"
                f"Current best/test summary:\n{state.latest_test_summary()}\n\n"
                f"Cooled axes:\n{', '.join(cooled_axes) if cooled_axes else 'none'}\n\n"
                f"Recent reject summary:\n{reject_summary}\n\n"
                f"Recent agent feedback:\n{state.recent_notes_summary(feedback_notes_limit)}"
            ),
        },
    ]


def code_prompt(
    state: AgentState, feedback_notes_limit: int = 12, output_format: str = "json"
) -> list[dict[str, str]]:
    source_blocks = "\n\n".join(
        f"### {snap.path}\n```text\n{slice_text(snap.content)}\n```" for snap in state.file_context
    )
    reflection = state.scratch.get("reflection")
    reflection_block = (
        f"\n\nRetry reflection:\n{reflection}"
        if isinstance(reflection, str) and reflection.strip()
        else ""
    )
    return [
        {
            "role": "system",
            "content": CODE_XML_SYSTEM if output_format == "xml" else CODE_SYSTEM,
        },
        {
            "role": "user",
            "content": (
                f"User request:\n{state.user_request}\n\n"
                f"Plan:\n{state.plan_markdown}\n\n"
                f"Source files:\n{source_blocks}"
                f"\n\nLatest test summary:\n{state.latest_test_summary()}\n\n"
                f"Recent agent feedback:\n{state.recent_notes_summary(feedback_notes_limit)}"
                f"{reflection_block}"
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
    "REFLECT": REFLECT_SYSTEM,
    "BRAINSTORM": BRAINSTORM_SYSTEM,
    "CODE": CODE_SYSTEM,
    "CODE_XML": CODE_XML_SYSTEM,
    "TEST": TEST_SYSTEM,
}


def slice_text(text: str, limit: int = DEFAULT_CHAR_LIMIT) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + "\n[...truncated...]\n" + text[-tail:]
