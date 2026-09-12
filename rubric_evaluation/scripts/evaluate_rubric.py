#!/usr/bin/env python3
"""Evaluate deterministic rubric labels against existing human labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rubric_eval.evaluation import disagreement_records, evaluate_records, render_markdown
from rubric_eval.io_utils import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    report = evaluate_records(records)
    disagreements = disagreement_records(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metrics.json", report)
    write_jsonl(args.output_dir / "disagreements.jsonl", disagreements)
    (args.output_dir / "summary.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Accuracy (scorable): {report['scorable_metrics']['accuracy']:.4f}")
    print(f"Macro-F1 (scorable): {report['scorable_metrics']['macro_f1']:.4f}")
    print(f"Consequential recall: {report['scorable_metrics']['per_label']['CONSEQUENTIAL']['recall']:.4f}")
    print(f"Scorable coverage: {report['scorable_coverage']:.4f}")
    print(f"Disagreements: {len(disagreements)}")
    print(f"Reports: {args.output_dir}")


if __name__ == "__main__":
    main()
