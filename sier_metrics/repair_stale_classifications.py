#!/usr/bin/env python3
"""Prepare and merge repairs for stale benchmark classification rows."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING_ROOT = ROOT / "segments-mapping" / "benchmark"
DEFAULT_REPAIR_DIR = (
    ROOT
    / "benchmark"
    / "testing"
    / "reports"
    / "asr_full"
    / "metrics"
    / "analysis"
    / "classification_repair"
)
MODELS: tuple[str, ...] = (
    "qwen3-asr",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "sensevoice",
    "paraformer",
    "qwen-asr-flash-filetrans",
    "doubao",
)
FINGERPRINT_FIELDS: tuple[str, ...] = (
    "target",
    "pred",
    "reference_segment",
    "hypothesis_segment",
    "op_segment",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def key(row: dict[str, Any]) -> tuple[str, str]:
    focus_id = row.get("focus_id")
    return (str(row.get("id") or ""), "" if focus_id is None else str(focus_id))


def fingerprint(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field) or "") for field in FINGERPRINT_FIELDS)


def dedupe(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {key(row): row for row in rows}


def prepare(mapping_root: Path, repair_dir: Path) -> dict[str, Any]:
    queue: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for model in MODELS:
        model_dir = mapping_root / model
        inputs = dedupe(read_jsonl(model_dir / "llm_classification_input.jsonl"))
        results = dedupe(read_jsonl(model_dir / "results" / "classification_results.jsonl"))
        stale: list[dict[str, Any]] = []
        for row_key, input_row in inputs.items():
            result_row = results.get(row_key)
            if result_row is None or fingerprint(result_row) != fingerprint(input_row):
                stale.append({**input_row, "_repair_model": model})
        counts[model] = len(stale)
        queue.extend(stale)

    queue_path = repair_dir / "llm_classification_input.jsonl"
    results_dir = repair_dir / "results"
    if results_dir.exists() and any(results_dir.iterdir()):
        raise FileExistsError(
            f"repair results already exist under {results_dir}; move them before preparing a new run"
        )
    write_jsonl(queue_path, queue)
    manifest = {
        "mapping_root": str(mapping_root),
        "queue_path": str(queue_path),
        "total": len(queue),
        "by_model": counts,
    }
    (repair_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_repair_result(
    repair_row: dict[str, Any],
    input_row: dict[str, Any],
    *,
    model: str,
) -> None:
    if fingerprint(repair_row) != fingerprint(input_row):
        raise ValueError(f"repair fingerprint mismatch for {model}/{key(input_row)}")
    classification = repair_row.get("classification")
    if not isinstance(classification, dict) or not isinstance(
        classification.get("is_semantic_error"), bool
    ):
        raise ValueError(f"invalid repair classification for {model}/{key(input_row)}")


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(file_descriptor)
    temp_path = Path(temp_name)
    try:
        write_jsonl(temp_path, rows)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def merge(mapping_root: Path, repair_dir: Path) -> dict[str, Any]:
    repair_results_path = repair_dir / "results" / "classification_results.jsonl"
    repair_rows = read_jsonl(repair_results_path)
    repair_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    duplicate_keys: Counter[tuple[str, str, str]] = Counter()
    for row in repair_rows:
        model = str(row.get("_repair_model") or "")
        repair_key = (model, *key(row))
        duplicate_keys[repair_key] += 1
        repair_index[repair_key] = row
    duplicates = [repair_key for repair_key, count in duplicate_keys.items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate repair results: {duplicates[:3]}")

    failures = read_jsonl(repair_dir / "results" / "classification_failures.jsonl")
    if failures:
        raise ValueError(f"repair has {len(failures)} failed classifications")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary: dict[str, Any] = {"repair_results": len(repair_rows), "models": {}}
    expected_repairs = 0
    for model in MODELS:
        model_dir = mapping_root / model
        input_rows = read_jsonl(model_dir / "llm_classification_input.jsonl")
        existing_path = model_dir / "results" / "classification_results.jsonl"
        existing = dedupe(read_jsonl(existing_path))
        merged: list[dict[str, Any]] = []
        repaired = 0
        for input_row in input_rows:
            row_key = key(input_row)
            existing_row = existing.get(row_key)
            if existing_row is not None and fingerprint(existing_row) == fingerprint(input_row):
                merged.append(existing_row)
                continue
            expected_repairs += 1
            repair_row = repair_index.get((model, *row_key))
            if repair_row is None:
                raise ValueError(f"missing repair result for {model}/{row_key}")
            validate_repair_result(repair_row, input_row, model=model)
            merged.append({field: value for field, value in repair_row.items() if field != "_repair_model"})
            repaired += 1

        if len(merged) != len(input_rows):
            raise ValueError(f"merged row count mismatch for {model}")
        for input_row, merged_row in zip(input_rows, merged, strict=True):
            if key(input_row) != key(merged_row) or fingerprint(input_row) != fingerprint(merged_row):
                raise ValueError(f"post-merge validation failed for {model}/{key(input_row)}")

        backup = existing_path.with_name(f"{existing_path.name}.bak-stale-repair-{timestamp}")
        shutil.copy2(existing_path, backup)
        atomic_write_jsonl(existing_path, merged)
        summary["models"][model] = {
            "rows": len(merged),
            "repaired": repaired,
            "backup": str(backup),
        }

    if expected_repairs != len(repair_rows):
        raise ValueError(
            f"repair result count mismatch: expected={expected_repairs}, actual={len(repair_rows)}"
        )
    summary["total_repaired"] = expected_repairs
    summary_path = repair_dir / "merge_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "merge"))
    parser.add_argument("--mapping-root", type=Path, default=DEFAULT_MAPPING_ROOT)
    parser.add_argument("--repair-dir", type=Path, default=DEFAULT_REPAIR_DIR)
    args = parser.parse_args()

    if args.action == "prepare":
        summary = prepare(args.mapping_root, args.repair_dir)
    else:
        summary = merge(args.mapping_root, args.repair_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
