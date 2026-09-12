#!/usr/bin/env python3
"""Run rubric-based semantic consequence inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rubric_eval.config import load_rubric
from rubric_eval.io_utils import read_jsonl
from rubric_eval.llm_runner import load_env_file, resolve_api_settings, run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--rubric",
        type=Path,
        default=ROOT / "config" / "rubric_v1.json",
    )
    parser.add_argument("--env-file", type=Path, default=ROOT.parent / ".env")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=3200)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.max_attempts <= 0:
        raise SystemExit("--workers and --max-attempts must be positive")

    load_env_file(args.env_file)
    model, base_url, api_key = resolve_api_settings(args.model, args.base_url, args.api_key)
    records = read_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    config = load_rubric(args.rubric)
    manifest = run_inference(
        records=records,
        output_path=args.output,
        config=config,
        model=model,
        base_url=base_url,
        api_key=api_key,
        workers=args.workers,
        temperature=args.temperature,
        seed=args.seed,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
        retry_failures=args.retry_failures,
    )
    print(
        f"Finished: successes={manifest['new_successes']}, "
        f"failures={manifest['new_failures']}, output={args.output}"
    )


if __name__ == "__main__":
    main()
