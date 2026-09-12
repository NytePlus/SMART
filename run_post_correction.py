#!/usr/bin/env python3
"""Correct hypotheses using history context without exposing current references."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "post_correction"))

from asr_post_correction.client import CachedJsonClient  # noqa: E402
from asr_post_correction.core import CorrectionEngine  # noqa: E402
from asr_post_correction.prompts import load_prompt_bundle  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--mode", choices=("single", "multi"), default="single")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default="https://apifusion.aispeech.com.cn/v1")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/post_correction")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=4)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    for index, row in enumerate(rows, 1):
        if "reference" in row or "target" in row:
            raise ValueError(f"current reference is forbidden in post-correction input (line {index})")
        for field in ("id", "history_context", "hypothesis"):
            if field not in row:
                raise ValueError(f"missing {field!r} on line {index}")
    prompts = load_prompt_bundle(args.prompt)
    client = CachedJsonClient(
        model=args.model, base_url=args.base_url, cache_dir=args.cache_dir,
        max_attempts=args.max_attempts,
    )
    engine = CorrectionEngine(client, prompts)

    def process(row: dict) -> dict:
        try:
            hypothesis = str(row["hypothesis"])
            context = {"history": str(row["history_context"]), "metadata": row.get("metadata", {})}
            result = (engine.single_round(hypothesis, context) if args.mode == "single"
                      else engine.multi_round(hypothesis, context, args.max_rounds))
            return {"id": str(row["id"]), "status": "ok", **result.to_dict()}
        except Exception as exc:
            return {"id": str(row.get("id", "")), "status": "error", "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(process, rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    failures = sum(row["status"] != "ok" for row in results)
    print(f"corrected={len(results) - failures}, failed={failures} -> {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
