#!/usr/bin/env python3
"""Run rubric inference for mapped errors and aggregate authoritative SER variants."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "rubric-based"))
sys.path.insert(0, str(ROOT / "sier_metrics"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def rubric_inputs(mapped_rows: list[dict]) -> list[dict]:
    result = []
    for row in mapped_rows:
        for segment in row.get("error_segments", []):
            index = int(segment["segment_index"])
            result.append({
                "sample_id": f"{row['id']}:{index}",
                "id": str(row["id"]),
                "focus_id": str(index),
                "dataset": str(row.get("dataset", "unknown")),
                "target": str(row["reference"]),
                "pred": str(row["hypothesis"]),
                "reference_segment": str(segment["reference_segment"]),
                "hypothesis_segment": str(segment["hypothesis_segment"]),
                "op_segment": str(segment["op_segment"]),
                "history": str(row.get("history_context", "")),
                "suffix": str(row.get("suffix_context", "")),
            })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--rubric", type=Path, default=ROOT / "rubric-based/config/rubric_v1.json")
    parser.add_argument("--segment-output", required=True, type=Path)
    parser.add_argument("--metric-output", required=True, type=Path)
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://apifusion.aispeech.com.cn/v1"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=2)
    args = parser.parse_args()

    from rubric_eval.config import load_rubric
    from rubric_eval.llm_runner import run_inference
    from _ser_metrics import compute_ser_variants

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("set LLM_API_KEY or OPENAI_API_KEY")
    mapped = read_jsonl(args.input)
    inputs = rubric_inputs(mapped)
    raw_output = args.segment_output.with_name(args.segment_output.stem + ".raw.jsonl")
    config = load_rubric(args.rubric)
    run_inference(
        records=inputs, output_path=raw_output, config=config, model=args.model,
        base_url=args.base_url, api_key=api_key, workers=args.workers,
        max_attempts=args.max_attempts,
    )
    raw_results = read_jsonl(raw_output)
    normalized = []
    classification_rows = []
    unscorable = 0
    for row in raw_results:
        aggregated = row.get("aggregated_result") or {}
        label = aggregated.get("final_label", "UNSCORABLE")
        if row.get("inference_status") != "SUCCESS":
            label = "UNSCORABLE"
        normalized.append({
            "id": str(row.get("id", "")),
            "segment_index": int(row.get("focus_id", 0)),
            "reference_segment": row.get("reference_segment", ""),
            "hypothesis_segment": row.get("hypothesis_segment", ""),
            "op_segment": row.get("op_segment", ""),
            "label": label,
            "is_semantic_error": label == "CONSEQUENTIAL",
            "rubric_version": row.get("rubric_version"),
            "items": (row.get("rubric_judgment") or {}).get("items", {}),
            "change_types": (row.get("rubric_judgment") or {}).get("change_types", []),
        })
        if label == "UNSCORABLE":
            unscorable += 1
            continue
        classification_rows.append({
            **{key: row.get(key, "") for key in (
                "id", "focus_id", "target", "pred", "reference_segment",
                "hypothesis_segment", "op_segment")},
            "classification": {"is_semantic_error": label == "CONSEQUENTIAL"},
        })
    write_jsonl(args.segment_output, normalized)

    model_dir = args.metric_output.parent / ".sier_work"
    write_jsonl(model_dir / "pred.raw.jsonl", [
        {"id": str(row["id"]), "target": row["reference"], "pred": row["hypothesis"]}
        for row in mapped
    ])
    scorable_ids = {
        str(result.get("sample_id"))
        for result in raw_results
        if result.get("inference_status") == "SUCCESS"
        and (result.get("aggregated_result") or {}).get("final_label")
        in {"CONSEQUENTIAL", "INCONSEQUENTIAL"}
    }
    write_jsonl(model_dir / "llm_classification_input.jsonl", [
        row for row in inputs if str(row["sample_id"]) in scorable_ids
    ])
    write_jsonl(model_dir / "results/classification_results.jsonl", classification_rows)
    metrics = compute_ser_variants(model_dir)
    output = {
        **metrics,
        "sier": metrics.get("s_ser"),
        "variants": {key: metrics.get(key) for key in ("s_ser", "c_ser", "c_ser2", "w_ser", "w_mer2")},
        "unscorable_segments": unscorable,
    }
    args.metric_output.parent.mkdir(parents=True, exist_ok=True)
    args.metric_output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"rubric segments={len(normalized)}, unscorable={unscorable} -> {args.metric_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
