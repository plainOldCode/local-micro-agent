from __future__ import annotations

import json
import xml.etree.ElementTree as ET
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
    """Parse CODE output that uses raw XML blocks instead of JSON strings."""
    stripped = text.strip()
    start = stripped.find("<candidates")
    end = stripped.rfind("</candidates>")
    if start == -1 or end == -1:
        raise XmlValidationError("No <candidates> block found in model output")

    xml_text = stripped[start : end + len("</candidates>")]
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise XmlValidationError(f"Invalid XML candidates: {exc}") from exc

    candidates = []
    for index, candidate_el in enumerate(root.findall("candidate"), start=1):
        changes = []
        for change_el in candidate_el.findall("change"):
            path = _xml_child_text(change_el, "path")
            search = _xml_child_text(change_el, "search")
            replace = _xml_child_text(change_el, "replace")
            changes.append(
                {
                    "path": path,
                    "target": _trim_xml_block(search),
                    "replacement": _trim_xml_block(replace),
                    "reason": _xml_child_text(change_el, "reason")
                    or _xml_child_text(candidate_el, "reason"),
                }
            )
        candidates.append(
            {
                "id": candidate_el.attrib.get("id", str(index)),
                "reason": _xml_child_text(candidate_el, "reason"),
                "changes": changes,
            }
        )

    if not candidates:
        raise XmlValidationError("No <candidate> entries found in XML output")
    return {"candidates": candidates}


def _xml_child_text(element: ET.Element, tag: str) -> str:
    child = element.find(tag)
    if child is None or child.text is None:
        return ""
    return child.text


def _trim_xml_block(text: str) -> str:
    lines = text.splitlines()
    if lines and not lines[0].strip():
        lines = lines[1:]
    if lines and not lines[-1].strip():
        lines = lines[:-1]
    return "\n".join(lines)


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
