#!/usr/bin/env python3
"""Generate benchmark table T5: chunk (jieba) -> word count distribution."""

from __future__ import annotations

import _path_setup  # noqa: F401

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from _text_units import iter_jieba_chunks, word_count

from asr_testing.config import ASR_FULL_METRICS, TESTING_ROOT
from asr_testing.eval_subset import (
    dedupe_pred_rows_by_source,
    load_composite_source_map,
)

ROOT = TESTING_ROOT.parents[1]
DEFAULT_BENCHMARK_ROOT = ROOT / "benchmark" / "benchmark"
DEFAULT_MAPPING_ROOT = ROOT / "segments-mapping" / "benchmark"
DEFAULT_OUTPUT = ASR_FULL_METRICS / "tables" / "t5" / "metrics_table_t5.md"
DEFAULT_MODEL = "qwen3-asr"
DEFAULT_MODEL_DISPLAY = "Qwen3-ASR"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def chunk_word_length_distribution(
    utterances: list[tuple[str, str]],
) -> tuple[list[tuple[int, int, float, float]], int, dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    """
    For each jieba chunk in all targets, count words (CJK chars + English words).

    Returns (distribution rows, total_chunks, first example per bucket, all examples per bucket).
    """
    counts: Counter[int] = Counter()
    examples: dict[int, dict[str, Any]] = {}
    examples_all: dict[int, list[dict[str, Any]]] = {}
    for utterance_id, target in utterances:
        for chunk in iter_jieba_chunks(target):
            length = word_count(chunk)
            counts[length] += 1
            item = {
                "utterance_id": utterance_id,
                "chunk": chunk,
                "word_count": length,
            }
            if length not in examples:
                examples[length] = item
            examples_all.setdefault(length, []).append(item)

    total = sum(counts.values()) or 1
    cumulative = 0
    distribution: list[tuple[int, int, float, float]] = []
    for length in sorted(counts):
        count = counts[length]
        cumulative += count
        distribution.append((length, count, 100.0 * count / total, 100.0 * cumulative / total))
    return distribution, total, examples, examples_all


def distribution_until_cumulative(
    distribution: list[tuple[int, int, float, float]],
    *,
    max_cumulative_pct: float,
) -> list[tuple[int, int, float, float]]:
    """Keep rows strictly before cumulative share reaches max_cumulative_pct."""
    selected: list[tuple[int, int, float, float]] = []
    for row in distribution:
        if row[3] >= max_cumulative_pct:
            break
        selected.append(row)
    return selected


def tail_rows_for_examples(
    rows: list[tuple[int, int, float, float]],
    *,
    tail_cumulative_pct: float = 10.0,
) -> list[tuple[int, int, float, float]]:
    """Rows whose cumulative mass falls in the last tail_cumulative_pct percent."""
    threshold = 100.0 - tail_cumulative_pct
    return [row for row in rows if row[3] > threshold]


def build_full_deduped_composites(
    composite_source_map: dict[str, tuple[str, str]],
) -> set[str]:
    allowed: set[str] = set()
    seen_sources: set[tuple[str, str]] = set()
    for composite_id in sorted(composite_source_map):
        source_key = composite_source_map[composite_id]
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        allowed.add(composite_id)
    return allowed


def load_deduped_utterances(
    model_dir: Path,
    *,
    allowed_composites: set[str],
    composite_source_map: dict[str, tuple[str, str]],
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    pred_rows = [row for row in read_jsonl(model_dir / "pred.raw.jsonl") if row.get("id") in allowed_composites]
    pred_rows, duplicate_utterances_removed = dedupe_pred_rows_by_source(
        pred_rows,
        composite_source_map,
    )
    utterances = [
        (str(row.get("id") or ""), str(row.get("target") or ""))
        for row in pred_rows
        if row.get("target")
    ]
    meta = {
        "utterances": len(pred_rows),
        "targets_with_text": len(utterances),
        "duplicate_source_utterances_removed": duplicate_utterances_removed,
    }
    return utterances, meta


def _format_example_cell(example: dict[str, Any] | None) -> str:
    if not example:
        return "—"
    chunk = str(example.get("chunk") or "").replace("|", "\\|")
    utterance_id = example.get("utterance_id") or ""
    return f"`{chunk}`（{utterance_id}）"


def _format_examples_detail(items: list[dict[str, Any]], *, max_inline: int = 3) -> str:
    if not items:
        return "—"
    inline = [_format_example_cell(item) for item in items[:max_inline]]
    if len(items) <= max_inline:
        return "；".join(inline)
    return "；".join(inline) + f"；… 共 {len(items)} 条"


def render_t5_high_word_section(
    rows: list[tuple[int, int, float, float]],
    *,
    examples_all: dict[int, list[dict[str, Any]]],
    start_index: int,
    min_words: int = 3,
    list_all_examples_threshold: int = 10,
) -> list[str]:
    """Render full detail for word-count buckets >= min_words."""
    high_rows = [row for row in rows if row[0] >= min_words]
    if not high_rows:
        return []

    lines = [
        "",
        f"## [T5-{min_words}+] chunk含word数 ≥ {min_words} 明细",
        "",
        f"**[T5-{min_words}+-NOTE]** 累计 ≥99% 区间的 {len(high_rows)} 档全部展开；"
        "汇总表每档给出前 3 条示例，下方分档列出该档全部 chunk。",
        "",
        "| 标号 | chunk含word数 | chunk 数 | 占比 | 累计占比 | 示例 |",
        "|:---:|---:|---:|---:|---:|---|",
    ]
    for offset, (length, count, pct, cum_pct) in enumerate(high_rows):
        idx = start_index + offset
        bucket_examples = examples_all.get(length, [])
        if count <= list_all_examples_threshold:
            example_cell = _format_examples_detail(bucket_examples, max_inline=count)
        else:
            example_cell = _format_examples_detail(bucket_examples, max_inline=3)
        lines.append(
            f"| T5-{idx} | {length} | {count:,} | {pct:.2f}% | {cum_pct:.2f}% | {example_cell} |"
        )

    for length, count, _, _ in high_rows:
        bucket_examples = examples_all.get(length, [])
        lines.extend(["", f"### chunk含word数 = {length}（共 {count} 条）", ""])
        for item in bucket_examples:
            lines.append(f"- {_format_example_cell(item)}")
    return lines


def render_t5_table(
    distribution: list[tuple[int, int, float, float]],
    *,
    full_distribution: list[tuple[int, int, float, float]],
    examples: dict[int, dict[str, Any]],
    examples_all: dict[int, list[dict[str, Any]]],
    total_chunks: int,
    model_display: str,
    utterances: int,
    max_cumulative_pct: float = 99.0,
    tail_cumulative_pct: float = 10.0,
    high_word_min: int = 3,
) -> str:
    tail_lengths = {row[0] for row in tail_rows_for_examples(distribution, tail_cumulative_pct=tail_cumulative_pct)}
    has_tail_examples = bool(tail_lengths)

    header = "| 标号 | chunk含word数 | chunk 数 | 占比 | 累计占比 |"
    if has_tail_examples:
        header += " 示例 |"
    separator = "|:---:|---:|---:|---:|---:|"
    if has_tail_examples:
        separator += "---|"

    lines = [
        "## [T5] 全量 benchmark — chunk含word数分布",
        "",
        f"**[T5-NOTE]** 模型 scope：`{model_display}` 全量 benchmark（{utterances} 条 utterance，"
        "按 `(source_dataset, source_segment_id)` source 去重）。"
        "**chunk** = jieba 有效词段（排除纯标点/符号）；"
        "**word** = 最小分割单元（中文 1 汉字 = 1 word，英文 1 单词 = 1 word）。"
        f" 下表列出累计占比 **<{max_cumulative_pct:g}%** 的全部行；"
        f"累计占比 **后 {tail_cumulative_pct:g}%** 区间（>{100 - tail_cumulative_pct:g}%）各行附具体 chunk 示例；"
        f"**≥{high_word_min} word** 档位见下方明细表。",
        "",
        header,
        separator,
    ]
    for idx, (length, count, pct, cum_pct) in enumerate(distribution, start=1):
        row = f"| T5-{idx} | {length} | {count:,} | {pct:.1f}% | {cum_pct:.1f}% |"
        if has_tail_examples:
            example_cell = _format_example_cell(examples.get(length)) if length in tail_lengths else "—"
            row += f" {example_cell} |"
        lines.append(row)

    high_rows = [row for row in full_distribution if row[0] >= high_word_min]
    if high_rows:
        lines.extend(
            render_t5_high_word_section(
                full_distribution,
                examples_all=examples_all,
                start_index=len(distribution) + 1,
                min_words=high_word_min,
            )
        )

    all_lengths = [item[0] for item in full_distribution for _ in range(item[1])]
    if all_lengths:
        median = statistics.median(all_lengths)
        p75 = statistics.quantiles(all_lengths, n=4)[2] if len(all_lengths) >= 4 else max(all_lengths)
        lines.extend(
            [
                "",
                f"**[T5-RANGE]** 共 {total_chunks:,} 个 chunk；"
                f"word 数最小 {min(all_lengths)}，最大 {max(all_lengths)}，中位数 {median:g}，P75 {p75:g}。",
            ]
        )
    return "\n".join(lines) + "\n"


def generate(
    *,
    benchmark_root: Path,
    mapping_root: Path,
    model_key: str = DEFAULT_MODEL,
    model_display: str = DEFAULT_MODEL_DISPLAY,
) -> tuple[str, dict[str, Any]]:
    composite_source_map = load_composite_source_map(benchmark_root)
    allowed_composites = build_full_deduped_composites(composite_source_map)
    model_dir = mapping_root / model_key

    utterances, load_meta = load_deduped_utterances(
        model_dir,
        allowed_composites=allowed_composites,
        composite_source_map=composite_source_map,
    )
    full_distribution, total_chunks, examples, examples_all = chunk_word_length_distribution(utterances)
    displayed_distribution = distribution_until_cumulative(full_distribution, max_cumulative_pct=99.0)
    tail_rows = tail_rows_for_examples(displayed_distribution, tail_cumulative_pct=10.0)
    high_word_rows = [row for row in full_distribution if row[0] >= 3]

    meta = {
        "model_key": model_key,
        "model_display": model_display,
        "deduped_utterances": load_meta["utterances"],
        "total_chunks": total_chunks,
        "composite_before_dedupe": len(composite_source_map),
        **load_meta,
        "word_definition": "1 CJK character or 1 English/Latin word token per minimum unit",
        "displayed_until_cumulative_pct": 99.0,
        "tail_example_cumulative_pct": 10.0,
        "high_word_min": 3,
        "distribution_full": [
            {"words_per_chunk": length, "chunk_count": count, "pct": pct, "cum_pct": cum_pct}
            for length, count, pct, cum_pct in full_distribution
        ],
        "distribution_displayed": [
            {"words_per_chunk": length, "chunk_count": count, "pct": pct, "cum_pct": cum_pct}
            for length, count, pct, cum_pct in displayed_distribution
        ],
        "distribution_high_word": [
            {
                "words_per_chunk": row[0],
                "chunk_count": row[1],
                "pct": row[2],
                "cum_pct": row[3],
                "examples": examples_all.get(row[0], []),
            }
            for row in high_word_rows
        ],
        "tail_examples": [
            {
                "words_per_chunk": row[0],
                **examples[row[0]],
            }
            for row in tail_rows
            if row[0] in examples
        ],
    }
    markdown = render_t5_table(
        displayed_distribution,
        full_distribution=full_distribution,
        examples=examples,
        examples_all=examples_all,
        total_chunks=total_chunks,
        model_display=model_display,
        utterances=load_meta["utterances"],
    )
    return markdown, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--mapping-root", type=Path, default=DEFAULT_MAPPING_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model key under segments-mapping/benchmark/")
    parser.add_argument(
        "--model-display",
        default=None,
        help="Display name in table note (default: derived from --model)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model_display = args.model_display
    if model_display is None:
        model_display = DEFAULT_MODEL_DISPLAY if args.model == DEFAULT_MODEL else args.model
    markdown, meta = generate(
        benchmark_root=args.benchmark_root,
        mapping_root=args.mapping_root,
        model_key=args.model,
        model_display=model_display,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps({"meta": meta, "markdown": markdown}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.write_text(markdown, encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
