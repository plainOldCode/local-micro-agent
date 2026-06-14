from __future__ import annotations

import json
import re
from typing import Any


class JsonValidationError(ValueError):
    pass


class XmlValidationError(JsonValidationError):
    pass


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from models that sometimes add accidental prose."""
    stripped = text.strip()
    try:
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise JsonValidationError("No JSON object found in model output")
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JsonValidationError(f"Invalid JSON object: {exc}") from exc


def parse_xml_candidates(text: str) -> dict[str, Any]:
    """Parse CODE output that uses XML-like raw text blocks.

    This intentionally avoids a real XML parser because raw Python snippets may
    contain `<`, `>`, quotes, and other characters that are valid code but not
    escaped XML text.
    """
    stripped = text.strip()
    start = stripped.find("<candidates")
    end = stripped.rfind("</candidates>")
    if start == -1 or end == -1:
        raise XmlValidationError("No <candidates> block found in model output")

    block = stripped[start : end + len("</candidates>")]
    candidate_blocks = re.findall(
        r"<candidate(?:\s+id=\"([^\"]*)\")?\s*>(.*?)</candidate>",
        block,
        re.DOTALL,
    )

    candidates = []
    for index, (candidate_id, candidate_content) in enumerate(candidate_blocks, start=1):
        changes = []
        reason = _tag_text(candidate_content, "reason")
        strategy_axis = _tag_text(candidate_content, "strategy_axis")
        for change_content in re.findall(r"<change>(.*?)</change>", candidate_content, re.DOTALL):
            path = _tag_text(change_content, "path")
            search = _tag_text(change_content, "search", strip=False)
            replace = _tag_text(change_content, "replace", strip=False)
            change_reason = _tag_text(change_content, "reason") or reason
            if not path or not search or not replace:
                continue
            change = {
                "path": path,
                "target": _trim_xml_block(search),
                "replacement": _trim_xml_block(replace),
                "reason": change_reason,
            }
            start_line = _optional_int(_tag_text(change_content, "start_line"))
            end_line = _optional_int(_tag_text(change_content, "end_line"))
            anchor_before = _trim_xml_block(
                _tag_text(change_content, "anchor_before", strip=False)
            )
            anchor_after = _trim_xml_block(
                _tag_text(change_content, "anchor_after", strip=False)
            )
            if start_line is not None:
                change["start_line"] = start_line
            if end_line is not None:
                change["end_line"] = end_line
            if anchor_before:
                change["anchor_before"] = anchor_before
            if anchor_after:
                change["anchor_after"] = anchor_after
            changes.append(change)
        if not changes:
            continue
        candidates.append(
            {
                "id": candidate_id or str(index),
                "reason": reason,
                "strategy_axis": strategy_axis,
                "changes": changes,
            }
        )

    if not candidates:
        raise XmlValidationError("No valid <candidate> entries found in XML output")
    return {"candidates": candidates}


def _tag_text(text: str, tag: str, strip: bool = True) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    if not match:
        return ""
    value = match.group(1)
    return value.strip() if strip else value


def _trim_xml_block(text: str) -> str:
    lines = text.splitlines()
    if lines and not lines[0].strip():
        lines = lines[1:]
    if lines and not lines[-1].strip():
        lines = lines[:-1]
    return "\n".join(lines)


def _optional_int(text: str) -> int | None:
    value = text.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def require_keys(data: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise JsonValidationError(f"Missing required keys: {missing}")


def retry_repair_prompt(bad_output: str, error: Exception) -> list[dict[str, str]]:
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
