#!/usr/bin/env python3
"""Normalize local c-SEER resume keys to match current classifier inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()

    input_path = args.model_dir / "llm_classification_input.jsonl"
    result_path = args.model_dir / "results" / "classification_results.rubric.jsonl"
    input_rows = [json.loads(line) for line in input_path.open(encoding="utf-8") if line.strip()]
    types = {
        (str(row.get("id", "")), str(row.get("focus_id", ""))): type(row.get("focus_id"))
        for row in input_rows
    }
    rows = [json.loads(line) for line in result_path.open(encoding="utf-8") if line.strip()]
    seen: set[tuple[object, object, object]] = set()
    normalized = []
    for row in reversed(rows):
        key = (str(row.get("id", "")), str(row.get("focus_id", "")))
        focus_type = types.get(key)
        if focus_type is int:
            row["focus_id"] = int(row["focus_id"])
        row["classifier"] = "rubric"
        resume_key = (row.get("_repair_model"), row.get("id"), row.get("focus_id"))
        if resume_key in seen:
            continue
        seen.add(resume_key)
        normalized.append(row)
    normalized.reverse()
    result_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in normalized),
        encoding="utf-8",
    )
    print(json.dumps({"input_rows": len(input_rows), "result_rows": len(normalized)}))


if __name__ == "__main__":
    main()
