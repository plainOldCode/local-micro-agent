from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class JsonValidationError(ValueError):
    pass


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from models that sometimes add accidental prose."""
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise JsonValidationError("No JSON object found in model output")
    return json.loads(stripped[start : end + 1])


def parse_model_json(text: str, schema: type[T]) -> T:
    try:
        return schema.model_validate(parse_json_object(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise JsonValidationError(str(exc)) from exc


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
