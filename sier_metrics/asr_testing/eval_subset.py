"""Eval subset helpers: keep full benchmark data, subsample AISHELL-4 for SER/X/Z."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from asr_testing.audio_samples import available_datasets, read_jsonl
from asr_testing.config import TESTING_ROOT
from asr_testing.wer import load_benchmark_utterance_source_map

DEFAULT_BENCHMARK_ROOT = TESTING_ROOT.parent / "benchmark"
DEFAULT_AISHELL_LIMIT = 100
DEFAULT_SEED = 42
DEFAULT_SUBSET_NAME = "eval_subset_aishell100.json"


def make_composite_id(dataset: str, sample_id: str) -> str:
    return f"{dataset}:{sample_id}"


def load_composite_source_map(
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT,
) -> dict[str, tuple[str, str]]:
    """Map composite utterance id -> (source_dataset, source_segment_id)."""
    composite_map: dict[str, tuple[str, str]] = {}
    for (bucket, utterance_id), source_key in load_benchmark_utterance_source_map(
        benchmark_root
    ).items():
        composite_map[make_composite_id(bucket, utterance_id)] = source_key
    return composite_map


def parse_composite_id(composite_id: str) -> tuple[str, str] | None:
    if ":" not in composite_id:
        return None
    bucket, utterance_id = composite_id.split(":", 1)
    if not bucket or not utterance_id:
        return None
    return bucket, utterance_id


def resolve_source_segment_key(
    composite_id: str,
    composite_source_map: dict[str, tuple[str, str]],
) -> tuple[str, str] | tuple[str, str, str]:
    """Return (source_dataset, source_segment_id) or fallback (composite_id, '')."""
    source = composite_source_map.get(composite_id)
    if source and source[0] and source[1]:
        return source
    return (composite_id, "")


def source_error_dedupe_key(
    row: dict[str, Any],
    composite_source_map: dict[str, tuple[str, str]],
) -> tuple[str, ...]:
    composite_id = str(row.get("id") or "")
    focus_id = str(row.get("focus_id") or "")
    source_dataset, source_segment_id = resolve_source_segment_key(
        composite_id,
        composite_source_map,
    )
    if source_segment_id:
        return (source_dataset, source_segment_id, focus_id)
    return (composite_id, focus_id)


def dedupe_rows_by_source_error_key(
    rows: list[dict[str, Any]],
    composite_source_map: dict[str, tuple[str, str]],
) -> tuple[list[dict[str, Any]], int]:
    deduped: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        deduped[source_error_dedupe_key(row, composite_source_map)] = row
    unique_rows = list(deduped.values())
    return unique_rows, len(rows) - len(unique_rows)


def dedupe_pred_rows_by_source(
    rows: list[dict[str, Any]],
    composite_source_map: dict[str, tuple[str, str]],
) -> tuple[list[dict[str, Any]], int]:
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        composite_id = str(row.get("id") or "")
        source_dataset, source_segment_id = resolve_source_segment_key(
            composite_id,
            composite_source_map,
        )
        key = (
            (source_dataset, source_segment_id)
            if source_segment_id
            else (composite_id, "")
        )
        deduped[key] = row
    unique_rows = list(deduped.values())
    return unique_rows, len(rows) - len(unique_rows)


def dedupe_benchmark_text_map_by_source(
    items: dict[tuple[str, str], str],
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT,
) -> tuple[dict[tuple[str, str], str], dict[str, int]]:
    utterance_sources = load_benchmark_utterance_source_map(benchmark_root)
    deduped: dict[tuple[str, str], str] = {}
    seen_sources: set[tuple[str, str]] = set()
    removed = 0
    for key in sorted(items):
        source = utterance_sources.get(key)
        source_key = source if source and source[0] and source[1] else key
        if source_key in seen_sources:
            removed += 1
            continue
        seen_sources.add(source_key)
        deduped[key] = items[key]
    meta = {
        "input_utterances": len(items),
        "unique_source_utterances": len(deduped),
        "duplicate_source_utterances_removed": removed,
    }
    return deduped, meta


def load_source_dataset_map(benchmark_root: Path = DEFAULT_BENCHMARK_ROOT) -> dict[str, str]:
    source_map: dict[str, str] = {}
    for dataset in available_datasets(benchmark_root):
        semantic_path = benchmark_root / dataset / "processed" / "semantic_xyz.jsonl"
        if not semantic_path.exists():
            continue
        for row in read_jsonl(semantic_path):
            composite_id = make_composite_id(dataset, str(row["id"]))
            source_map[composite_id] = str(row.get("source_dataset") or "")
    return source_map


def build_eval_utterance_ids(
    source_map: dict[str, str],
    *,
    aishell_limit: int = DEFAULT_AISHELL_LIMIT,
    seed: int = DEFAULT_SEED,
) -> tuple[set[str], dict[str, Any]]:
    aishell_ids = sorted(cid for cid, src in source_map.items() if src == "AISHELL-4")
    non_aishell_ids = sorted(cid for cid, src in source_map.items() if src != "AISHELL-4")
    selected_aishell = set(aishell_ids[:aishell_limit])
    allowed = set(non_aishell_ids) | selected_aishell
    meta = {
        "aishell_limit": aishell_limit,
        "aishell_total": len(aishell_ids),
        "aishell_selected": len(selected_aishell),
        "non_aishell_total": len(non_aishell_ids),
        "eval_utterances": len(allowed),
        "selection_policy": "sorted_first_n",
        "seed": seed,
        "selected_aishell_ids": sorted(selected_aishell),
        "non_aishell_ids": non_aishell_ids,
    }
    return allowed, meta


def load_or_build_eval_subset(
    *,
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT,
    subset_path: Path | None = None,
    aishell_limit: int = DEFAULT_AISHELL_LIMIT,
    seed: int = DEFAULT_SEED,
    rebuild: bool = False,
) -> tuple[set[str], dict[str, Any]]:
    if subset_path is None:
        subset_path = TESTING_ROOT.parents[1] / "segments-mapping" / "benchmark" / DEFAULT_SUBSET_NAME
    if subset_path.exists() and not rebuild:
        payload = json.loads(subset_path.read_text(encoding="utf-8"))
        allowed = set(payload.get("eval_utterance_ids") or [])
        meta = payload.get("meta") or {}
        if allowed:
            return allowed, meta

    source_map = load_source_dataset_map(benchmark_root)
    allowed, meta = build_eval_utterance_ids(source_map, aishell_limit=aishell_limit, seed=seed)
    payload = {
        "benchmark_root": str(benchmark_root),
        "eval_utterance_ids": sorted(allowed),
        "meta": meta,
    }
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    subset_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return allowed, meta


def filter_rows_by_utterance(rows: list[dict[str, Any]], allowed: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("id") or "") in allowed]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def apply_eval_subset_to_model_dir(
    model_dir: Path,
    allowed: set[str],
    *,
    backup_full: bool = True,
) -> dict[str, Any]:
    llm_input = model_dir / "llm_classification_input.jsonl"
    llm_full = model_dir / "llm_classification_input.full.jsonl"
    if not llm_input.exists() and not llm_full.exists():
        raise FileNotFoundError(f"missing llm input under {model_dir}")

    if backup_full and llm_full.exists():
        full_rows = read_jsonl(llm_full)
    else:
        full_rows = read_jsonl(llm_input)
        if backup_full and full_rows:
            write_jsonl(llm_full, full_rows)

    filtered_rows = filter_rows_by_utterance(full_rows, allowed)
    write_jsonl(llm_input, filtered_rows)

    results_path = model_dir / "results" / "classification_results.jsonl"
    filtered_results_path = model_dir / "results" / "classification_results.full.jsonl"
    result_rows = read_jsonl(results_path)
    if result_rows:
        if backup_full and not filtered_results_path.exists():
            write_jsonl(filtered_results_path, result_rows)
        filtered_results = filter_rows_by_utterance(result_rows, allowed)
        write_jsonl(results_path, filtered_results)
    else:
        filtered_results = []

    pred_rows = read_jsonl(model_dir / "pred.raw.jsonl")
    filtered_pred_rows = filter_rows_by_utterance(
        [{"id": row["id"], **row} for row in pred_rows],
        allowed,
    )
    return {
        "model_dir": str(model_dir),
        "llm_input_full": len(full_rows),
        "llm_input_eval": len(filtered_rows),
        "classification_full": len(result_rows),
        "classification_eval": len(filtered_results),
        "pred_eval_utterances": len(filtered_pred_rows),
    }
