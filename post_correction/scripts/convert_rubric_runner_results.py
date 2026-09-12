#!/usr/bin/env python3
"""Convert rubric runner output to data-prepare's classification schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("inference_status") != "SUCCESS":
        raise ValueError(f"cannot convert unsuccessful sample {row.get('sample_id')}")
    aggregated = row.get("aggregated_result") or {}
    label = str(aggregated.get("final_label") or "")
    if label not in {"CONSEQUENTIAL", "INCONSEQUENTIAL"}:
        raise ValueError(f"unexpected final_label for {row.get('sample_id')}: {label}")
    judgment = row.get("rubric_judgment") or {}
    impact_id = "Q3"
    impact = (judgment.get("items") or {}).get(impact_id) or {}
    reason = str(impact.get("reason_code") or aggregated.get("aggregation_rule") or label)
    source_fields = {
        key: value
        for key, value in row.items()
        if key not in {
            "aggregated_result", "attempts", "inference_status", "judge_model",
            "latency_seconds", "prompt_version", "raw_response", "rubric_judgment",
            "rubric_version", "sample_id", "usage",
        }
    }
    return {
        **source_fields,
        "classifier": "rubric",
        "classification": {
            "is_semantic_error": label == "CONSEQUENTIAL",
            "reason": reason,
        },
        "llm_response": str(row.get("raw_response") or ""),
        "classification_details": {
            "raw_response": str(row.get("raw_response") or ""),
            "rubric_version": row.get("rubric_version"),
            "rubric_judgment": judgment,
            "aggregated_result": aggregated,
            "attempts": row.get("attempts") or [],
            "judge_model": row.get("judge_model"),
            "prompt_version": row.get("prompt_version"),
            "usage": row.get("usage") or {},
            "latency_seconds": row.get("latency_seconds"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    latest: dict[str, dict[str, Any]] = {}
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            latest[str(row.get("sample_id") or "")] = row
    failures = sum(row.get("inference_status") != "SUCCESS" for row in latest.values())
    if failures:
        raise ValueError(f"rubric runner still has {failures} failed samples")
    converted = [convert_row(row) for row in latest.values()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"converted": len(converted), "failures_skipped": failures}))


if __name__ == "__main__":
    main()
