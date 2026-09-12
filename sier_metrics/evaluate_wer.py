#!/usr/bin/env python3
"""Compute benchmark WER from ASR predictions and emit a LaTeX table."""

from __future__ import annotations

import _path_setup  # noqa: F401

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from asr_testing.config import ASR_FULL_METRICS, ASR_FULL_PREDICTIONS, TESTING_ROOT
from asr_testing.eval_subset import dedupe_benchmark_text_map_by_source
from asr_testing.wer import (
    benchmark_ground_truth_size,
    compute_wer,
    load_benchmark_ground_truth,
    load_model_predictions,
)

DEFAULT_PREDICTION_ROOT = ASR_FULL_PREDICTIONS
DEFAULT_BENCHMARK_ROOT = TESTING_ROOT.parent / "benchmark"
DEFAULT_OUTPUT_TEX = ASR_FULL_METRICS / "latex" / "large_scale_evaluation_wer.tex"
DEFAULT_OUTPUT_JSON = ASR_FULL_METRICS / "wer_metrics.json"

LATEX_TEMPLATE = """\\begin{{table}}[h]
\\caption{{{caption}}}
\\label{{{label}}}
\\centering
\\begin{{tabular}}{{lrrrrr}}
\\hline
ASR System & WER $\\downarrow$ & SER $\\downarrow$ & $\\bar{{s}}^{{a}}$ & $\\bar{{s}}^{{l}}$ & $\\bar{{s}}^{{c}}$\\\\
\\hline
\\multicolumn{{6}}{{l}}{{\\textbf{{Open-Source Models}}}} \\\\
{open_source_rows}\\hline
\\multicolumn{{6}}{{l}}{{\\textbf{{Closed-Source Models}}}} \\\\
{closed_source_rows}\\hline
\\end{{tabular}}
\\end{{table}}
"""


@dataclass(frozen=True)
class TableRow:
    section: str
    display_name: str
    model_key: str | None


TABLE_ROWS: tuple[TableRow, ...] = (
    TableRow("open_source", "Qwen3-ASR", "qwen3-asr"),
    TableRow("open_source", "whisper-large-v3", "whisper-large-v3"),
    TableRow("open_source", "whisper-large-v3-turbo", "whisper-large-v3-turbo"),
    TableRow("open_source", "sensevoice", "sensevoice"),
    TableRow("open_source", "paraformer", "paraformer"),
    TableRow("closed_source", "Qwen3-ASR-Flash-Filetrans", "qwen-asr-flash-filetrans"),
    TableRow("closed_source", "Doubao Filetrans", "doubao"),
)


def fmt_wer(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.2f}"


def render_table_row(display_name: str, wer: float | None) -> str:
    return f"{display_name} & {fmt_wer(wer)} & -- & -- & -- & -- \\\\"


def build_latex(rows: list[tuple[str, float | None]]) -> str:
    open_source = [render_table_row(name, wer) for section, name, wer in rows if section == "open_source"]
    closed_source = [render_table_row(name, wer) for section, name, wer in rows if section == "closed_source"]
    return LATEX_TEMPLATE.format(
        caption="Large-scale evaluation on the constructed benchmark.",
        label="tab:large-scale-evaluation",
        open_source_rows="\n".join(open_source) + "\n",
        closed_source_rows="\n".join(closed_source) + "\n",
    )


def evaluate_models(
    prediction_root: Path,
    benchmark_root: Path,
    *,
    min_coverage: float = 1.0,
) -> tuple[list[tuple[str, str, float | None]], dict]:
    ground_truth = load_benchmark_ground_truth(benchmark_root)
    ground_truth, source_dedupe_meta = dedupe_benchmark_text_map_by_source(
        ground_truth,
        benchmark_root,
    )
    benchmark_size = benchmark_ground_truth_size(ground_truth)

    table_values: list[tuple[str, str, float | None]] = []
    summary: dict = {
        "benchmark_root": str(benchmark_root),
        "prediction_root": str(prediction_root),
        "benchmark_size": benchmark_size,
        "benchmark_size_before_source_dedupe": source_dedupe_meta["input_utterances"],
        "source_dedupe": source_dedupe_meta,
        "min_coverage_for_tex": min_coverage,
        "models": {},
    }

    for row in TABLE_ROWS:
        if row.model_key is None:
            table_values.append((row.section, row.display_name, None))
            continue

        predictions = load_model_predictions(
            prediction_root,
            row.model_key,
            benchmark_root=benchmark_root,
        )
        predictions, _ = dedupe_benchmark_text_map_by_source(predictions, benchmark_root)
        if not predictions:
            table_values.append((row.section, row.display_name, None))
            summary["models"][row.model_key] = {
                "display_name": row.display_name,
                "evaluated": 0,
                "coverage": 0.0,
                "wer": None,
            }
            continue

        stats = compute_wer(predictions, ground_truth)
        coverage = stats.evaluated / benchmark_size if benchmark_size else 0.0
        summary["models"][row.model_key] = {
            "display_name": row.display_name,
            "evaluated": stats.evaluated,
            "coverage": coverage,
            "missing_gt": stats.missing_gt,
            "substitutions": stats.substitutions,
            "insertions": stats.insertions,
            "deletions": stats.deletions,
            "ref_chars": stats.ref_chars,
            "errors": stats.errors,
            "wer": stats.wer,
            "reported_in_tex": stats.wer if coverage >= min_coverage else None,
        }
        table_values.append(
            (row.section, row.display_name, stats.wer if coverage >= min_coverage else None)
        )

    return table_values, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=DEFAULT_PREDICTION_ROOT,
        help="Directory containing per-model prediction JSONL files",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help="Benchmark root containing <dataset>/processed/gt.jsonl",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=DEFAULT_OUTPUT_TEX,
        help="Output LaTeX table path",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON metrics path",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.98,
        help="Only fill WER in LaTeX when evaluated coverage reaches this ratio (default: 0.98)",
    )
    args = parser.parse_args()

    table_values, summary = evaluate_models(
        args.prediction_root,
        args.benchmark_root,
        min_coverage=args.min_coverage,
    )
    latex = build_latex(table_values)

    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(latex, encoding="utf-8")
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote LaTeX table to {args.output_tex}")
    print(f"Wrote metrics JSON to {args.output_json}")
    for section, display_name, wer in table_values:
        print(f"{display_name}: {fmt_wer(wer)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
