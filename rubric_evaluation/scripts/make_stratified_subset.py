#!/usr/bin/env python3
"""Create a deterministic dataset/label-stratified smoke-test subset."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rubric_eval.io_utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-stratum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude",
        action="append",
        type=Path,
        default=[],
        help="JSONL whose sample IDs must not be selected; may be repeated",
    )
    args = parser.parse_args()
    if args.per_stratum <= 0:
        raise SystemExit("--per-stratum must be positive")

    excluded_ids = {
        str(record["sample_id"])
        for path in args.exclude
        for record in read_jsonl(path)
    }
    groups: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for record in read_jsonl(args.input):
        if str(record["sample_id"]) in excluded_ids:
            continue
        human = record.get("human_annotation", {}).get("is_semantic_error")
        if not isinstance(human, bool):
            raise ValueError(f"{record.get('sample_id')}: missing boolean human label")
        groups[(str(record["dataset"]), human)].append(record)

    rng = random.Random(args.seed)
    selected: list[dict] = []
    for key in sorted(groups):
        candidates = list(groups[key])
        rng.shuffle(candidates)
        selected.extend(candidates[: args.per_stratum])
    selected.sort(key=lambda record: str(record["sample_id"]))
    count = write_jsonl(args.output, selected)
    print(
        f"Wrote {count} records across {len(groups)} strata to {args.output}; "
        f"excluded IDs={len(excluded_ids)}"
    )


if __name__ == "__main__":
    main()
