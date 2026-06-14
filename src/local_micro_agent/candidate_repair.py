from __future__ import annotations


def candidate_repair_prompt(
    output_format: str,
    bad_output: str,
    error: Exception,
) -> list[dict[str, str]]:
    """Return a repair prompt that preserves the requested candidate format."""
    if output_format == "xml":
        return _xml_candidate_repair_prompt(bad_output, error)
    return _json_candidate_repair_prompt(bad_output, error)


def candidate_repair_call_site(output_format: str, base: str) -> str:
    """Name repair call sites by output format for profiling and smoke diagnosis."""
    if output_format == "xml":
        return f"{base}_xml_repair"
    return f"{base}_json_repair"


def _xml_candidate_repair_prompt(
    bad_output: str,
    error: Exception,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Repair the CODE output into the XML-like candidate format only. "
                "Do not convert it to JSON. Return a single <candidates> root "
                "containing one or more complete <candidate> blocks. Each candidate "
                "must contain <reason> when available, optional <strategy_axis>, and "
                "one or more <change> blocks with <path>, <search>, <replace>, and "
                "optional <reason>. Raw code belongs inside <search> and <replace>; "
                "do not wrap the answer in Markdown or prose."
            ),
        },
        {
            "role": "user",
            "content": f"Validation error:\n{error}\n\nBad output:\n{bad_output}",
        },
    ]


def _json_candidate_repair_prompt(
    bad_output: str,
    error: Exception,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Repair the output into valid JSON only. No prose.",
        },
        {
            "role": "user",
            "content": f"Validation error:\n{error}\n\nBad output:\n{bad_output}",
        },
    ]
