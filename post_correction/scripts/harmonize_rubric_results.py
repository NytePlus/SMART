#!/usr/bin/env python3
"""Reuse one rubric judgment for byte-equivalent c-SEER classification inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FINGERPRINT_FIELDS = (
    "id",
    "focus_id",
    "target",
    "pred",
    "reference_segment",
    "hypothesis_segment",
    "op_segment",
)
SCORABLE_LABELS = {"CONSEQUENTIAL", "INCONSEQUENTIAL"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fingerprint(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in FINGERPRINT_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    judgments: dict[tuple[str, ...], dict[str, Any]] = {}
    skipped_unscorable = 0
    for path in args.canonical:
        for row in read_jsonl(path):
            label = str((row.get("aggregated_result") or {}).get("final_label", ""))
            if row.get("inference_status") != "SUCCESS" or label not in SCORABLE_LABELS:
                skipped_unscorable += 1
                continue
            judgments.setdefault(fingerprint(row), row)

    output_rows = []
    missing = []
    for input_row in read_jsonl(args.input):
        source = judgments.get(fingerprint(input_row))
        if source is None:
            missing.append(str(input_row.get("sample_id", "")))
            continue
        output_rows.append({**source, **input_row})
    if missing:
        raise ValueError(
            f"missing scorable canonical judgments for {len(missing)} inputs: {missing[:5]}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "inputs": len(output_rows),
        "canonical_fingerprints": len(judgments),
        "skipped_unscorable": skipped_unscorable,
    }))


if __name__ == "__main__":
    main()
