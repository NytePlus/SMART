#!/usr/bin/env python3
"""Run segment mapping, rubric inference, and SIER aggregation end to end."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rubric", type=Path, default=ROOT / "rubric-based/config/rubric_v1.json")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default="https://apifusion.aispeech.com.cn/v1")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping = args.output_dir / "segment_mapping.jsonl"
    rubric_results = args.output_dir / "rubric_results.jsonl"
    metrics = args.output_dir / "sier_metrics.json"
    subprocess.run([
        sys.executable, str(ROOT / "run_segment_mapping.py"),
        "--input", str(args.input), "--output", str(mapping),
    ], check=True)
    subprocess.run([
        sys.executable, str(ROOT / "run_rubric_sier.py"),
        "--input", str(mapping), "--rubric", str(args.rubric),
        "--segment-output", str(rubric_results), "--metric-output", str(metrics),
        "--model", args.model, "--base-url", args.base_url, "--workers", str(args.workers),
    ], check=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "rubric": str(args.rubric.resolve()),
        "model": args.model,
        "outputs": {
            "segment_mapping": str(mapping),
            "rubric_results": str(rubric_results),
            "sier_metrics": str(metrics),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"evaluation complete -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
