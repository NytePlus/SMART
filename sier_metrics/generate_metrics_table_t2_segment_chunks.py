#!/usr/bin/env python3
"""T2-variant: chunks per error segment, compared across low/medium/high WER models."""

from __future__ import annotations

import _path_setup  # noqa: F401

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from asr_testing.config import ASR_FULL_METRICS, TESTING_ROOT

ROOT = TESTING_ROOT.parents[1]
DEFAULT_MAPPING_ROOT = ROOT / "segments-mapping" / "benchmark"
DEFAULT_WER_METRICS = ASR_FULL_METRICS / "wer_metrics.json"
DEFAULT_OUTPUT = ASR_FULL_METRICS / "tables" / "t2_segment_chunks" / "metrics_table_t2_segment_chunks.md"

from _text_units import iter_jieba_chunks  # noqa: E402


def load_segment_mapping(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def error_segments_from_mapping(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        utterance_id = str(record.get("id") or "")
        for segment in record.get("segments") or []:
            op = str(segment.get("op_segment") or "")
            if op and all(ch == "=" for ch in op):
                continue
            rows.append(
                {
                    "utterance_id": utterance_id,
                    "reference_segment": str(segment.get("reference_segment") or ""),
                    "chunk_count": len(iter_jieba_chunks(str(segment.get("reference_segment") or ""))),
                }
            )
    return rows


def analyze_model(model_key: str, mapping_root: Path, wer_metrics: dict[str, Any]) -> dict[str, Any]:
    mapping_path = mapping_root / model_key / "segment_mapping.json"
    records = load_segment_mapping(mapping_path)
    error_rows = error_segments_from_mapping(records)
    chunk_counts = [row["chunk_count"] for row in error_rows if row["chunk_count"] > 0]
    distribution = Counter(chunk_counts)
    total_segments = len(error_rows)
    total_chunks = sum(chunk_counts)
    wer_info = wer_metrics["models"][model_key]

    dist_rows: list[dict[str, Any]] = []
    cumulative = 0
    for chunks_per_segment in sorted(distribution):
        count = distribution[chunks_per_segment]
        cumulative += count
        dist_rows.append(
            {
                "chunks_per_segment": chunks_per_segment,
                "segment_count": count,
                "pct": 100.0 * count / max(total_segments, 1),
                "cum_pct": 100.0 * cumulative / max(total_segments, 1),
            }
        )

    return {
        "model_key": model_key,
        "display_name": wer_info["display_name"],
        "wer": wer_info["wer"],
        "utterances": len(records),
        "error_segments": total_segments,
        "error_segment_chunks": total_chunks,
        "avg_chunks_per_segment": (total_chunks / total_segments) if total_segments else 0.0,
        "median_chunks_per_segment": statistics.median(chunk_counts) if chunk_counts else 0.0,
        "distribution": dist_rows,
    }


def pick_wer_tiers(wer_metrics: dict[str, Any]) -> tuple[str, str, str]:
    ranked = sorted(
        wer_metrics["models"].items(),
        key=lambda item: item[1]["wer"],
    )
    low_key = ranked[0][0]
    high_key = ranked[-1][0]
    med_key = ranked[len(ranked) // 2][0]
    return low_key, med_key, high_key


def render_summary_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "## [T2-SC] 全量 benchmark — 每个 error segment 含 chunk 数（三组 WER 对比）",
        "",
        "**[T2-SC-NOTE]** **segment** = segment mapping 错误段（`op_segment` 含非 `=`）；"
        "**chunk** = 对该 segment 的 `reference_segment` 做 jieba 分词后的有效词段数。"
        "选取 WER 最低 / 中等 / 最高三模型，全量 681 composite utterances。",
        "",
        "| 档位 | 模型 | WER% | 错误段数 | segment 内 chunk 总数 | 平均每 segment 含 chunk 数 | 中位数 |",
        "|:---:|---|---:|---:|---:|---:|---:|",
    ]
    tier_names = ["低 WER", "中 WER", "高 WER"]
    for tier, row in zip(tier_names, results):
        lines.append(
            f"| {tier} | {row['display_name']} | {row['wer']:.2f} | "
            f"{row['error_segments']:,} | {row['error_segment_chunks']:,} | "
            f"{row['avg_chunks_per_segment']:.3f} | {row['median_chunks_per_segment']:.0f} |"
        )
    return "\n".join(lines)


def render_distribution_table(result: dict[str, Any], *, label: str) -> list[str]:
    lines = [
        "",
        f"### [{label}] {result['display_name']} — segment 含 chunk 数分布",
        "",
        "| chunks/segment | 错误段数 | 占比 | 累计占比 |",
        "|---:|---:|---:|---:|",
    ]
    for row in result["distribution"]:
        lines.append(
            f"| {row['chunks_per_segment']} | {row['segment_count']:,} | "
            f"{row['pct']:.1f}% | {row['cum_pct']:.1f}% |"
        )
    return lines


def build_conclusion(low: dict[str, Any], high: dict[str, Any], *, title_suffix: str = "") -> str:
    wer_ratio = high["wer"] / low["wer"] if low["wer"] else 0.0
    seg_ratio = high["error_segments"] / low["error_segments"] if low["error_segments"] else 0.0
    chunk_ratio = high["error_segment_chunks"] / low["error_segment_chunks"] if low["error_segment_chunks"] else 0.0
    avg_ratio = high["avg_chunks_per_segment"] / low["avg_chunks_per_segment"] if low["avg_chunks_per_segment"] else 0.0

    if avg_ratio > seg_ratio * 1.05:
        chunk_mechanism = (
            "**平均每 segment 含 chunk 数明显上升**（增幅高于 segment 数增幅），"
            "说明除 word 错误并入已有 segment 外，还有相当部分错误体现为 segment 内 **chunk 边界增加**（新 word 形成新 chunk）。"
        )
    elif avg_ratio <= 1.05:
        chunk_mechanism = (
            "**平均每 segment 含 chunk 数几乎不变**，"
            "说明新增 word 级错误主要被 **包含在已有 chunk 内部**，而非切分出新的 jieba chunk。"
        )
    else:
        chunk_mechanism = (
            "平均每 segment 含 chunk 数有所上升，但低于 WER 增幅，"
            "两种机制（并入旧 chunk / 产生新 chunk）并存。"
        )

    title = f"## [T2-SC-CONC{title_suffix}] 结论"
    return "\n".join(
        [
            "",
            title,
            "",
            f"对比 **{low['display_name']}**（WER {low['wer']:.2f}%）与 "
            f"**{high['display_name']}**（WER {high['wer']:.2f}%）：",
            "",
            f"- WER 倍数：**{wer_ratio:.2f}×**",
            f"- 错误 segment 数：{low['error_segments']:,} → {high['error_segments']:,}（**{seg_ratio:.2f}×**）",
            f"- segment 内 chunk 总数：{low['error_segment_chunks']:,} → {high['error_segment_chunks']:,}（**{chunk_ratio:.2f}×**）",
            f"- 平均每 segment 含 chunk 数：{low['avg_chunks_per_segment']:.3f} → {high['avg_chunks_per_segment']:.3f}（**{avg_ratio:.2f}×**）",
            "",
            "**解读：** WER 升高时，若 segment 数增幅 **远小于** WER 增幅，则更多 word 级错误被归并进 **已有 segment**。"
            f"本组 segment 数仅 **{seg_ratio:.2f}×**（相对 WER **{wer_ratio:.2f}×**）。"
            f"{chunk_mechanism}",
        ]
    )


def render_pair_table(low: dict[str, Any], high: dict[str, Any], *, title: str) -> str:
    wer_ratio = high["wer"] / low["wer"]
    seg_ratio = high["error_segments"] / low["error_segments"]
    avg_ratio = high["avg_chunks_per_segment"] / low["avg_chunks_per_segment"]
    return "\n".join(
        [
            "",
            title,
            "",
            "| 指标 | 低 WER（T1-1） | 高 WER（T1-2） | 倍数 |",
            "|---|---:|---:|---:|",
            f"| 模型 | {low['display_name']} | {high['display_name']} | — |",
            f"| WER% | {low['wer']:.2f} | {high['wer']:.2f} | {wer_ratio:.2f}× |",
            f"| 错误 segment 数 | {low['error_segments']:,} | {high['error_segments']:,} | {seg_ratio:.2f}× |",
            f"| segment 内 chunk 总数 | {low['error_segment_chunks']:,} | {high['error_segment_chunks']:,} | {high['error_segment_chunks']/low['error_segment_chunks']:.2f}× |",
            f"| 平均每 segment 含 chunk 数 | {low['avg_chunks_per_segment']:.3f} | {high['avg_chunks_per_segment']:.3f} | {avg_ratio:.2f}× |",
        ]
    )


def generate(
    *,
    mapping_root: Path,
    wer_metrics_path: Path,
) -> tuple[str, dict[str, Any]]:
    wer_metrics = json.loads(wer_metrics_path.read_text(encoding="utf-8"))
    low_key, med_key, high_key = pick_wer_tiers(wer_metrics)
    results = [
        analyze_model(low_key, mapping_root, wer_metrics),
        analyze_model(med_key, mapping_root, wer_metrics),
        analyze_model(high_key, mapping_root, wer_metrics),
    ]

    # ~2× WER pair for T1-1 / T1-2 narrative.
    t1_low = analyze_model("qwen3-asr", mapping_root, wer_metrics)
    t1_high = analyze_model("whisper-large-v3", mapping_root, wer_metrics)

    lines = [render_summary_table(results)]
    for label, row in zip(["T2-SC-L", "T2-SC-M", "T2-SC-H"], results):
        lines.extend(render_distribution_table(row, label=label))
    lines.append(
        render_pair_table(
            t1_low,
            t1_high,
            title="## [T2-SC-2X] 约 2× WER 对比（T1-1 Qwen3-ASR vs T1-2 Whisper-large-v3）",
        )
    )
    lines.append(build_conclusion(t1_low, t1_high, title_suffix="-2X"))
    lines.append(build_conclusion(results[0], results[2], title_suffix=""))

    meta = {
        "selected_models": {"low": low_key, "medium": med_key, "high": high_key},
        "t1_pair": {"t1_1": "qwen3-asr", "t1_2": "whisper-large-v3"},
        "results": results,
        "t1_pair_results": {"t1_1": t1_low, "t1_2": t1_high},
    }
    return "\n".join(lines) + "\n", meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-root", type=Path, default=DEFAULT_MAPPING_ROOT)
    parser.add_argument("--wer-metrics", type=Path, default=DEFAULT_WER_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    markdown, meta = generate(mapping_root=args.mapping_root, wer_metrics_path=args.wer_metrics)
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
