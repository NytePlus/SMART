#!/usr/bin/env python3
"""Validate rubric experiment input JSONL."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rubric_eval.io_utils import read_jsonl
from rubric_eval.schema import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    issues = validate_dataset(records)
    if issues:
        for issue in issues[:20]:
            print(issue)
        raise SystemExit(f"Validation failed: {len(issues)} invalid records")
    datasets = Counter(str(record["dataset"]) for record in records)
    labels = Counter(
        str(record.get("human_annotation", {}).get("label_name", "missing"))
        for record in records
    )
    print(f"Valid records: {len(records)}")
    print(f"Datasets: {dict(datasets)}")
    print(f"Human labels: {dict(labels)}")


if __name__ == "__main__":
    main()

