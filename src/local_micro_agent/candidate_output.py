from __future__ import annotations

from typing import Literal

CandidateOutputMode = Literal["single", "queue"]


def candidate_output_message(
    output_format: str, mode: CandidateOutputMode = "single"
) -> dict[str, str]:
    """Return a format reminder for CODE-style candidate output."""
    if output_format == "xml":
        return _xml_candidate_output_message(mode)
    return _json_candidate_output_message(mode)


def _xml_candidate_output_message(mode: CandidateOutputMode) -> dict[str, str]:
    if mode == "queue":
        content = (
            "Candidate queue mode is enabled. Output one or more <candidate> "
            "blocks inside a single <candidates> root. Each candidate must be "
            "independent and safe to apply from the same baseline. If a strategy "
            "axis contract is present, include <strategy_axis>axis</strategy_axis> "
            "inside each <candidate>."
        )
    else:
        content = (
            "Output exactly one <candidate> block inside a single <candidates> "
            "root. The candidate must contain exactly one <change>. If a strategy "
            "axis contract is present, include <strategy_axis>axis</strategy_axis> "
            "inside the <candidate>."
        )
    return {"role": "system", "content": content}


def _json_candidate_output_message(mode: CandidateOutputMode) -> dict[str, str]:
    if mode == "queue":
        content = (
            "Candidate queue mode is enabled. Output strict JSON with a top-level "
            '"candidates" array, not a top-level "changes" array. Example: '
            '{"candidates":[{"id":"1","strategy_axis":"general_edit",'
            '"reason":"short","changes":[{"path":"file.py",'
            '"target":"exact text","replacement":"new text","reason":"short",'
            '"start_line":12,"end_line":14,"anchor_before":"exact nearby text"}]}]}. '
            "Each candidate must be independent and safe to apply from the same baseline."
        )
    else:
        content = (
            "Output strict JSON with a top-level candidates array containing exactly "
            "one candidate and one change. Example: "
            '{"candidates":[{"id":"1","strategy_axis":"general_edit",'
            '"reason":"short","changes":[{"path":"file.py",'
            '"target":"exact text","replacement":"new text","reason":"short",'
            '"start_line":12,"end_line":14,"anchor_before":"exact nearby text"}]}]}.'
        )
    return {"role": "system", "content": content}
