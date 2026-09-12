"""Parse model JSON and return actionable validation errors."""

from __future__ import annotations

import json
import re


def parse_json_response(raw: str) -> dict:
    stripped = raw.strip()
    if not stripped:
        raise ValueError("empty model response")
    candidates = [stripped]
    code_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if code_match:
        candidates.append(code_match.group(1))
    brace_match = re.search(r"(\{.*\})", stripped, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(1))

    errors: list[str] = []
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if not isinstance(value, dict):
            raise ValueError("model response JSON must be an object")
        return value
    raise ValueError(f"cannot parse model response as JSON: {errors[-1] if errors else 'unknown error'}")

