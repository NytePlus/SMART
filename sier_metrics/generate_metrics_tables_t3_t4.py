#!/usr/bin/env python3
"""Generate benchmark tables T3, T4, and example E6 with source-level deduplication."""

from __future__ import annotations

import _path_setup  # noqa: F401

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from _ser_metrics import (
    classification_key,
    compute_ser_variants,
    dedupe_classification_rows,
    load_input_keys,
    read_jsonl,
)
from _text_units import text_chunk_count
from generate_metrics_examples import ExampleSpec, build_example, render_example

from asr_testing.config import ASR_FULL_METRICS, ASR_FULL_PREDICTIONS, TESTING_ROOT
from asr_testing.eval_subset import (
    dedupe_pred_rows_by_source,
    dedupe_rows_by_source_error_key,
    load_composite_source_map,
    make_composite_id,
    dedupe_benchmark_text_map_by_source,
)
from asr_testing.wer import (
    compute_wer,
    load_benchmark_ground_truth,
    load_model_predictions,
)

ROOT = TESTING_ROOT.parents[1]
DEFAULT_BENCHMARK_ROOT = ROOT / "benchmark" / "benchmark"
DEFAULT_MAPPING_ROOT = ROOT / "segments-mapping" / "benchmark"
DEFAULT_PREDICTION_ROOT = ASR_FULL_PREDICTIONS
DEFAULT_OUTPUT = ASR_FULL_METRICS / "tables" / "t3_t4" / "metrics_tables_t3_t4.md"

AISHELL_DATASET = "AISHELL-4"

MODELS: tuple[tuple[str, str], ...] = (
    ("qwen3-asr", "Qwen3-ASR"),
    ("whisper-large-v3", "Whisper-large-v3"),
    ("whisper-large-v3-turbo", "Whisper-large-v3-turbo"),
    ("sensevoice", "SenseVoice"),
    ("paraformer", "Paraformer"),
    ("qwen-asr-flash-filetrans", "Qwen-Flash-Filetrans"),
    ("doubao", "Doubao Filetrans"),
)


def compute_model_ser(
    model_dir: Path,
    *,
    allowed_composites: set[str],
    composite_source_map: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    return compute_ser_variants(
        model_dir,
        allowed_composites=allowed_composites,
        composite_source_map=composite_source_map,
    )


def filter_non_aishell_text_map(
    items: dict[tuple[str, str], str],
    composite_source_map: dict[str, tuple[str, str]],
) -> dict[tuple[str, str], str]:
    filtered: dict[tuple[str, str], str] = {}
    for (bucket, utterance_id), text in items.items():
        composite_id = make_composite_id(bucket, utterance_id)
        source = composite_source_map.get(composite_id)
        if source and source[0] == AISHELL_DATASET:
            continue
        filtered[(bucket, utterance_id)] = text
    return filtered


def compute_model_wer(
    model_key: str,
    *,
    benchmark_root: Path,
    prediction_root: Path,
    composite_source_map: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    ground_truth = load_benchmark_ground_truth(benchmark_root)
    ground_truth = filter_non_aishell_text_map(ground_truth, composite_source_map)
    ground_truth, gt_meta = dedupe_benchmark_text_map_by_source(ground_truth, benchmark_root)

    predictions = load_model_predictions(
        prediction_root,
        model_key,
        benchmark_root=benchmark_root,
    )
    predictions = filter_non_aishell_text_map(predictions, composite_source_map)
    predictions, _ = dedupe_benchmark_text_map_by_source(predictions, benchmark_root)
    predictions = {key: predictions[key] for key in ground_truth if key in predictions}

    stats = compute_wer(predictions, ground_truth)
    return {
        "utterances": len(ground_truth),
        "dedupe_meta": gt_meta,
        "substitutions": stats.substitutions,
        "insertions": stats.insertions,
        "deletions": stats.deletions,
        "ref_chars": stats.ref_chars,
        "errors": stats.errors,
        "wer": stats.wer,
    }


def build_non_aishell_deduped_composites(
    composite_source_map: dict[str, tuple[str, str]],
) -> set[str]:
    allowed: set[str] = set()
    seen_sources: set[tuple[str, str]] = set()
    for composite_id in sorted(composite_source_map):
        source_dataset, source_segment_id = composite_source_map[composite_id]
        if source_dataset == AISHELL_DATASET:
            continue
        source_key = (source_dataset, source_segment_id)
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        allowed.add(composite_id)
    return allowed


def filter_rows_by_composites(
    rows: list[dict[str, Any]],
    allowed: set[str],
) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("id") or "") in allowed]


def length_distribution(rows: list[dict[str, Any]]) -> list[tuple[int, int, float, float]]:
    counts = Counter(text_chunk_count(row.get("reference_segment", "")) for row in rows)
    total = sum(counts.values()) or 1
    cumulative = 0
    result: list[tuple[int, int, float, float]] = []
    for length in sorted(counts):
        count = counts[length]
        cumulative += count
        result.append((length, count, 100.0 * count / total, 100.0 * cumulative / total))
    return result


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.2f}"


def fmt_int(value: int) -> str:
    return f"{value:,}"


def render_t3_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| 标号 | 模型 | 指标 | s-SER% | c-SER% | w-SER% | 对齐总和/chunk总数 | 总错误/错误段 | 替换/插入/删除 或 语义/非语义 |",
        "|:---:|---|:---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"| T3-{idx}a | {row['display_name']} | WER | {fmt_pct(row['wer'])} | — | — | "
            f"{fmt_int(row['ref_chars'])} | {fmt_int(row['errors'])} | "
            f"{row['substitutions']}/{row['insertions']}/{row['deletions']} |"
        )
        lines.append(
            f"| T3-{idx}b | {row['display_name']} | SER | {fmt_pct(row['s_ser'])} | "
            f"{fmt_pct(row['c_ser'])} | {fmt_pct(row['w_ser'])} | "
            f"{fmt_int(row['total_target_chunks'])} | {fmt_int(row['error_spans'])} | "
            f"{row['semantic_error_spans']}/{row['non_semantic_error_spans']} |"
        )
    return "\n".join(lines)


def render_t4_table(distribution: list[tuple[int, int, float, float]], total: int) -> str:
    lines = [
        "| 标号 | jieba 段长 | 错误段数 | 占比 | 累计占比 |",
        "|:---:|---:|---:|---:|---:|",
    ]
    for idx, (length, count, pct, cum_pct) in enumerate(distribution, start=1):
        length_cell = f"**{length}**" if length == 37 else str(length)
        lines.append(
            f"| T4-{idx} | {length_cell} | {count} | {pct:.1f}% | {cum_pct:.1f}% |"
        )
    lengths = [item[0] for item in distribution for _ in range(item[1])]
    if lengths:
        median = statistics.median(lengths)
        p75 = statistics.quantiles(lengths, n=4)[2] if len(lengths) >= 4 else max(lengths)
        lines.extend(
            [
                "",
                f"**[T4-RANGE]** 最小 {min(lengths)}，最大 {max(lengths)}，中位数 {median:g}，P75 {p75:g}。",
                f"**[T4-NOTE]** Qwen3-ASR，{total} 个错误段（排除 AISHELL-4，按 source 去重后）。",
            ]
        )
    return "\n".join(lines)


def generate(
    *,
    benchmark_root: Path,
    mapping_root: Path,
    prediction_root: Path,
) -> tuple[str, dict[str, Any]]:
    composite_source_map = load_composite_source_map(benchmark_root)
    allowed_composites = build_non_aishell_deduped_composites(composite_source_map)

    t3_rows: list[dict[str, Any]] = []
    for model_key, display_name in MODELS:
        wer = compute_model_wer(
            model_key,
            benchmark_root=benchmark_root,
            prediction_root=prediction_root,
            composite_source_map=composite_source_map,
        )
        ser = compute_model_ser(
            mapping_root / model_key,
            allowed_composites=allowed_composites,
            composite_source_map=composite_source_map,
        )
        t3_rows.append(
            {
                "model_key": model_key,
                "display_name": display_name,
                **wer,
                **ser,
            }
        )

    qwen_dir = mapping_root / "qwen3-asr"
    qwen_rows = filter_rows_by_composites(
        read_jsonl(qwen_dir / "results" / "classification_results.jsonl"),
        allowed_composites,
    )
    input_keys = load_input_keys(qwen_dir)
    if input_keys:
        qwen_rows = [row for row in qwen_rows if classification_key(row) in input_keys]
    qwen_rows = dedupe_classification_rows(qwen_rows)
    qwen_rows, removed_dupes = dedupe_rows_by_source_error_key(qwen_rows, composite_source_map)
    t4_distribution = length_distribution(qwen_rows)

    length37_rows = [
        row for row in qwen_rows if text_chunk_count(row.get("reference_segment", "")) == 37
    ]
    e6_spec = ExampleSpec("E6", "X0Y2Z0:0000008", 19, "段长 37 长删除（source 去重后唯一）")
    e6_example = build_example(e6_spec, model_dir=qwen_dir, composite_source_map=composite_source_map)

    meta = {
        "non_aishell_deduped_utterances": len(allowed_composites),
        "qwen3_error_spans": len(qwen_rows),
        "duplicate_source_error_rows_removed": removed_dupes,
        "length37_count": len(length37_rows),
    }

    non_aishell_before_dedupe = sum(
        1
        for composite_id, (source_dataset, _) in composite_source_map.items()
        if source_dataset != AISHELL_DATASET
    )
    markdown_parts = [
        "## [T3] 排除 AISHELL-4 — WER / SER 汇总（source 去重）",
        "",
        f"**[T3-NOTE]** 排除 AISHELL-4 后 benchmark 内 {non_aishell_before_dedupe} 条 composite utterance，"
        f"按 `(source_dataset, source_segment_id)` 去重后 **{len(allowed_composites)}** 条 unique utterance。"
        "结构与 [T1] 相同：WER 行最后一列为 替换/插入/删除；SER 行为 s-SER / c-SER / w-SER 及 语义/非语义。",
        "",
        render_t3_table(t3_rows),
        "",
        "## [T4] 排除 AISHELL-4 — 错误段 jieba 长度分布（source 去重）",
        "",
        render_t4_table(t4_distribution, len(qwen_rows)),
        "",
        "## [T3-T4-L37] 段长 = 37 的样例（E6）",
        "",
        render_example(e6_example).rstrip(),
        "",
        "**[E6-NOTE]** source 去重后，段长 37 在全 benchmark（排除 AISHELL）中仅剩 **1 条** unique 错误段；"
        "原先 E7–E10 为同一 `canvas-speech:0000962` / `focus_id=19` 在多个 XYZ bucket 的重复计数。",
        "",
    ]
    if not length37_rows:
        markdown_parts.append("未找到 jieba 段长 = 37 的错误段。")

    return "\n".join(markdown_parts) + "\n", meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--mapping-root", type=Path, default=DEFAULT_MAPPING_ROOT)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    markdown, meta = generate(
        benchmark_root=args.benchmark_root,
        mapping_root=args.mapping_root,
        prediction_root=args.prediction_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "markdown": markdown}
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.write_text(markdown, encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
