#!/usr/bin/env python3
"""Analyze cases where WER and s-SER rank models in opposite directions."""

from __future__ import annotations

import _path_setup  # noqa: F401

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from asr_testing.config import ASR_FULL_METRICS, ASR_FULL_PREDICTIONS, TESTING_ROOT
from asr_testing.eval_subset import (
    dedupe_benchmark_text_map_by_source,
    load_composite_source_map,
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
DEFAULT_T1_JSON = ASR_FULL_METRICS / "tables" / "t1" / "metrics_table_t1.json"
DEFAULT_OUTPUT = ASR_FULL_METRICS / "analysis" / "wer_sser_divergence.md"

MODELS: tuple[str, ...] = (
    "qwen3-asr",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "sensevoice",
    "paraformer",
    "qwen-asr-flash-filetrans",
    "doubao",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def classification_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("id") or ""), str(row.get("focus_id") or ""))


def dedupe_classification_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = {classification_key(row): row for row in rows}
    return list(deduped.values())


def load_input_keys(model_dir: Path) -> set[tuple[str, str]]:
    return {
        classification_key(row)
        for row in read_jsonl(model_dir / "llm_classification_input.jsonl")
    }


def is_semantic_true(row: dict[str, Any]) -> bool:
    classification = row.get("classification")
    return bool(classification.get("is_semantic_error")) if isinstance(classification, dict) else False


def composite_id(key: tuple[str, str]) -> str:
    return f"{key[0]}:{key[1]}"


CLASSIFICATION_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "target",
    "pred",
    "reference_segment",
    "hypothesis_segment",
    "op_segment",
)


def load_model_classifications(
    model_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], set[str], dict[str, Any]]:
    input_rows = dedupe_classification_rows(
        read_jsonl(model_dir / "llm_classification_input.jsonl")
    )
    result_rows = dedupe_classification_rows(
        read_jsonl(model_dir / "results" / "classification_results.jsonl")
    )
    inputs = {classification_key(row): row for row in input_rows}
    results = {classification_key(row): row for row in result_rows}

    valid_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    missing_keys: list[tuple[str, str]] = []
    expected_by_id: dict[str, int] = defaultdict(int)
    valid_by_id: dict[str, int] = defaultdict(int)
    for key, input_row in inputs.items():
        cid = key[0]
        expected_by_id[cid] += 1
        result_row = results.get(key)
        if result_row is None:
            missing_keys.append(key)
            continue
        differing_fields = [
            field
            for field in CLASSIFICATION_FINGERPRINT_FIELDS
            if str(input_row.get(field) or "") != str(result_row.get(field) or "")
        ]
        if differing_fields:
            mismatches.append(
                {
                    "id": key[0],
                    "focus_id": key[1],
                    "differing_fields": differing_fields,
                    "input_reference_segment": input_row.get("reference_segment"),
                    "result_reference_segment": result_row.get("reference_segment"),
                }
            )
            continue
        valid_rows.append(result_row)
        valid_by_id[cid] += 1

    complete_ids = {
        cid for cid, count in expected_by_id.items() if valid_by_id.get(cid, 0) == count
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        grouped[str(row.get("id") or "")].append(row)
    validation = {
        "input_rows": len(inputs),
        "result_rows": len(results),
        "valid_rows": len(valid_rows),
        "mismatched_rows": len(mismatches),
        "missing_rows": len(missing_keys),
        "unexpected_result_rows": len(set(results) - set(inputs)),
        "complete_utterances": len(complete_ids),
        "input_utterances": len(expected_by_id),
        "mismatch_examples": mismatches[:5],
    }
    return grouped, complete_ids, validation


def classification_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    semantic = [row for row in rows if is_semantic_true(row)]
    non_semantic = [row for row in rows if not is_semantic_true(row)]
    return {
        "error_spans": len(rows),
        "semantic_spans": len(semantic),
        "non_semantic_spans": len(non_semantic),
        "semantic_reasons": [
            str((row.get("classification") or {}).get("reason") or "") for row in semantic
        ],
        "non_semantic_reasons": [
            str((row.get("classification") or {}).get("reason") or "")
            for row in non_semantic
        ],
        "non_semantic_segments": [
            {
                "reference": str(row.get("reference_segment") or ""),
                "hypothesis": str(row.get("hypothesis_segment") or ""),
                "reason": str((row.get("classification") or {}).get("reason") or ""),
            }
            for row in non_semantic
        ],
    }


def build_utterance_metrics(
    *,
    benchmark_root: Path,
    mapping_root: Path,
    prediction_root: Path,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    ground_truth = load_benchmark_ground_truth(benchmark_root)
    deduped_ground_truth, dedupe_meta = dedupe_benchmark_text_map_by_source(
        ground_truth,
        benchmark_root,
    )
    source_map = load_composite_source_map(benchmark_root)

    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    validation_by_model: dict[str, Any] = {}
    for model_key in MODELS:
        predictions = load_model_predictions(
            prediction_root,
            model_key,
            benchmark_root=benchmark_root,
        )
        classifications, complete_ids, validation = load_model_classifications(
            mapping_root / model_key
        )
        validation_by_model[model_key] = validation
        model_rows: dict[str, dict[str, Any]] = {}
        for key, reference in deduped_ground_truth.items():
            hypothesis = predictions.get(key)
            if hypothesis is None:
                continue
            stats = compute_wer({key: hypothesis}, {key: reference})
            cid = composite_id(key)
            class_summary = classification_summary(classifications.get(cid, []))
            model_rows[cid] = {
                "id": cid,
                "source": ":".join(source_map.get(cid, ("", ""))),
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": stats.wer,
                "wer_errors": stats.errors,
                "ref_tokens": stats.ref_chars,
                "substitutions": stats.substitutions,
                "insertions": stats.insertions,
                "deletions": stats.deletions,
                "classification_complete": cid in complete_ids,
                **class_summary,
            }
        by_model[model_key] = model_rows
    return by_model, dedupe_meta, validation_by_model


def find_inversion_pairs(models: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for left_key, right_key in itertools.combinations(MODELS, 2):
        left = models[left_key]
        right = models[right_key]
        if left["wer"] == right["wer"] or left["s_ser"] == right["s_ser"]:
            continue
        if left["wer"] > right["wer"] and left["s_ser"] < right["s_ser"]:
            high_key, low_key = left_key, right_key
        elif right["wer"] > left["wer"] and right["s_ser"] < left["s_ser"]:
            high_key, low_key = right_key, left_key
        else:
            continue
        high = models[high_key]
        low = models[low_key]
        pairs.append(
            {
                "high_wer_low_sser_model": high_key,
                "comparison_model": low_key,
                "high_wer": high["wer"],
                "low_wer": low["wer"],
                "high_s_ser": high["s_ser"],
                "low_s_ser": low["s_ser"],
                "wer_gap": high["wer"] - low["wer"],
                "s_ser_gap": high["s_ser"] - low["s_ser"],
            }
        )
    return sorted(pairs, key=lambda row: (-row["wer_gap"], row["s_ser_gap"]))


def find_direct_contributors(
    pair: dict[str, Any],
    utterance_metrics: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    high_key = pair["high_wer_low_sser_model"]
    low_key = pair["comparison_model"]
    high_rows = utterance_metrics[high_key]
    low_rows = utterance_metrics[low_key]
    candidates: list[dict[str, Any]] = []
    for cid in sorted(high_rows.keys() & low_rows.keys()):
        high = high_rows[cid]
        low = low_rows[cid]
        if not high["classification_complete"] or not low["classification_complete"]:
            continue
        error_delta = high["wer_errors"] - low["wer_errors"]
        semantic_delta = high["semantic_spans"] - low["semantic_spans"]
        if error_delta <= 0 or semantic_delta >= 0:
            continue
        candidates.append(
            {
                "id": cid,
                "source": high["source"],
                "reference": high["reference"],
                "high_model": high_key,
                "high_hypothesis": high["hypothesis"],
                "high_wer": high["wer"],
                "high_wer_errors": high["wer_errors"],
                "high_semantic_spans": high["semantic_spans"],
                "high_non_semantic_spans": high["non_semantic_spans"],
                "high_non_semantic_segments": high["non_semantic_segments"],
                "comparison_model": low_key,
                "comparison_hypothesis": low["hypothesis"],
                "comparison_wer": low["wer"],
                "comparison_wer_errors": low["wer_errors"],
                "comparison_semantic_spans": low["semantic_spans"],
                "wer_error_delta": error_delta,
                "semantic_span_delta": semantic_delta,
            }
        )
    return sorted(
        candidates,
        key=lambda row: (
            row["semantic_span_delta"],
            -row["wer_error_delta"],
            -(row["high_wer"] or 0.0),
        ),
    )


def find_high_wer_zero_semantic_cases(
    utterance_metrics: dict[str, dict[str, dict[str, Any]]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for model_key, rows in utterance_metrics.items():
        for row in rows.values():
            if (
                row["wer"] is not None
                and row["wer"] >= threshold
                and row["classification_complete"]
                and row["error_spans"] > 0
                and row["semantic_spans"] == 0
            ):
                cases.append({"model": model_key, **row})
    return sorted(cases, key=lambda row: (-row["wer"], -row["wer_errors"], row["id"]))


def md_text(value: Any, *, limit: int = 180) -> str:
    text = " ".join(str(value).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "..."


def render_report(payload: dict[str, Any], *, top_per_pair: int, top_zero: int) -> str:
    models = payload["models"]
    display = {key: value["display_name"] for key, value in models.items()}
    lines = [
        "# WER / s-SER 反趋势分析",
        "",
        "## 口径",
        "",
        f"- 模型级反转基于 T1 全量 {payload['scope']['composite_utterances']} 条 composite utterance。",
        f"- case 分析按 source 去重，共 {payload['scope']['unique_source_utterances']} 条，避免同一原始音频跨 XYZ bucket 重复贡献。",
        "- 直接贡献 case 定义：同一句上，模型 A 的 WER error 数更多，同时语义错误 segment 数更少。",
        f"- 强反例候选定义：单句 WER >= {payload['scope']['high_wer_threshold']:.0f}%，有错误 segment，但语义错误 segment 为 0。",
        "- 分类结果必须与当前 LLM 输入的 target/pred/segment/op 指纹完全一致，否则该句不进入 case 结论。",
        "",
        "## 分类结果一致性",
        "",
        "| 模型 | 当前输入行 | 指纹一致 | 内容错配 | 缺失 | 完整utterance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in MODELS:
        row = payload["classification_validation"][key]
        lines.append(
            f"| {models[key]['display_name']} | {row['input_rows']} | {row['valid_rows']} | "
            f"{row['mismatched_rows']} | {row['missing_rows']} | "
            f"{row['complete_utterances']}/{row['input_utterances']} |"
        )
    total_inputs = sum(
        row["input_rows"] for row in payload["classification_validation"].values()
    )
    total_mismatches = sum(
        row["mismatched_rows"] for row in payload["classification_validation"].values()
    )
    if total_mismatches:
        validation_conclusion = (
            f"**结论：** 当前 {total_inputs} 条输入中有 **{total_mismatches} 条内容错配**。"
            "下方 T1 模型级 S-SER 及排序反转为已有产物的报告值，"
            "必须补跑错配分类并重算后才能作为最终结论。"
        )
    else:
        validation_conclusion = (
            f"**结论：** 当前 {total_inputs} 条输入全部通过内容指纹校验，"
            "下方 T1 模型级 S-SER 及排序反转基于修复后重算结果。"
        )
    lines.extend(
        [
        "",
        validation_conclusion,
        "",
        "## 全模型汇总",
        "",
        "| 模型 | WER% | WER排名 | s-SER% | s-SER排名 | c-SER% | w-SER% | 语义段/全部错误段 | 非语义段占比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    wer_rank = {
        key: idx
        for idx, key in enumerate(sorted(models, key=lambda item: models[item]["wer"]), start=1)
    }
    sser_rank = {
        key: idx
        for idx, key in enumerate(sorted(models, key=lambda item: models[item]["s_ser"]), start=1)
    }
    for key in sorted(models, key=lambda item: models[item]["wer"]):
        row = models[key]
        non_semantic_share = 100.0 * row["non_semantic_error_spans"] / row["error_spans"]
        lines.append(
            f"| {row['display_name']} | {row['wer']:.2f} | {wer_rank[key]} | "
            f"{row['s_ser']:.2f} | {sser_rank[key]} | {row['c_ser']:.2f} | "
            f"{row['w_ser']:.2f} | {row['semantic_error_spans']}/{row['error_spans']} | "
            f"{non_semantic_share:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## 模型级排序反转",
            "",
            "| WER更高但s-SER更低 | 对照模型 | WER（前/后） | s-SER（前/后） | 直接贡献case数 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for pair in payload["inversion_pairs"]:
        lines.append(
            f"| {display[pair['high_wer_low_sser_model']]} | {display[pair['comparison_model']]} | "
            f"{pair['high_wer']:.2f}/{pair['low_wer']:.2f} | "
            f"{pair['high_s_ser']:.2f}/{pair['low_s_ser']:.2f} | "
            f"{pair['direct_contributor_count']} |"
        )

    for pair in payload["inversion_pairs"]:
        high_name = display[pair["high_wer_low_sser_model"]]
        low_name = display[pair["comparison_model"]]
        lines.extend(
            [
                "",
                f"### {high_name} vs {low_name}",
                "",
                "| ID/source | ref | 高WER模型hyp | 对照hyp | WER%（高/对照） | 语义段（高/对照） | 高WER模型非语义判定 |",
                "|---|---|---|---|---:|---:|---|",
            ]
        )
        for case in pair["direct_contributors"][:top_per_pair]:
            reasons = "; ".join(
                f"{item['reference']}->{item['hypothesis']} ({item['reason']})"
                for item in case["high_non_semantic_segments"]
            )
            lines.append(
                f"| {case['id']}<br>{case['source']} | {md_text(case['reference'])} | "
                f"{md_text(case['high_hypothesis'])} | {md_text(case['comparison_hypothesis'])} | "
                f"{case['high_wer']:.1f}/{case['comparison_wer']:.1f} | "
                f"{case['high_semantic_spans']}/{case['comparison_semantic_spans']} | "
                f"{md_text(reasons)} |"
            )

    lines.extend(
        [
            "",
            "## 高 WER 且零语义错误候选",
            "",
            f"共 {len(payload['high_wer_zero_semantic_cases'])} 个 model-case；其中参考长度至少 8 token 的有 "
            f"{sum(1 for row in payload['high_wer_zero_semantic_cases'] if row['ref_tokens'] >= 8)} 个。"
            f"下表按 WER 展示长样本中的前 {top_zero} 个，减少一字短句造成的极端百分比干扰。",
            "",
            "| 模型 | ID/source | WER% | ref | hyp | 非语义判定 |",
            "|---|---|---:|---|---|---|",
        ]
    )
    substantive_cases = [
        row for row in payload["high_wer_zero_semantic_cases"] if row["ref_tokens"] >= 8
    ]
    for case in substantive_cases[:top_zero]:
        reasons = "; ".join(
            f"{item['reference']}->{item['hypothesis']} ({item['reason']})"
            for item in case["non_semantic_segments"]
        )
        lines.append(
            f"| {display[case['model']]} | {case['id']}<br>{case['source']} | {case['wer']:.1f} | "
            f"{md_text(case['reference'])} | {md_text(case['hypothesis'])} | {md_text(reasons)} |"
        )
    lines.extend(
        [
            "",
            "## 解读边界",
            "",
            "这里的反转不自动等价于 S-SER 有效。它可能来自三类机制：真正不影响语义的字面差异；多个 WER token 错误聚合为少量 segment；或语义分类/分段漏判。最终结论需对上表代表 case 做人工复核。",
            "",
        ]
    )
    return "\n".join(lines)


def generate(
    *,
    benchmark_root: Path,
    mapping_root: Path,
    prediction_root: Path,
    t1_json: Path,
    high_wer_threshold: float,
    top_per_pair: int,
    top_zero: int,
) -> tuple[str, dict[str, Any]]:
    t1 = json.loads(t1_json.read_text(encoding="utf-8"))
    models = t1["meta"]["models"]
    missing_models = set(MODELS) - set(models)
    if missing_models:
        raise ValueError(f"T1 JSON missing models: {sorted(missing_models)}")

    utterance_metrics, dedupe_meta, classification_validation = build_utterance_metrics(
        benchmark_root=benchmark_root,
        mapping_root=mapping_root,
        prediction_root=prediction_root,
    )
    pairs = find_inversion_pairs(models)
    for pair in pairs:
        contributors = find_direct_contributors(pair, utterance_metrics)
        pair["direct_contributor_count"] = len(contributors)
        pair["direct_contributor_wer_error_delta"] = sum(
            row["wer_error_delta"] for row in contributors
        )
        pair["direct_contributor_semantic_span_delta"] = sum(
            row["semantic_span_delta"] for row in contributors
        )
        pair["direct_contributors"] = contributors

    payload = {
        "scope": {
            "composite_utterances": t1["meta"]["composite_before_dedupe"],
            "unique_source_utterances": dedupe_meta["unique_source_utterances"],
            "duplicate_source_utterances_removed": dedupe_meta[
                "duplicate_source_utterances_removed"
            ],
            "high_wer_threshold": high_wer_threshold,
        },
        "definitions": t1["meta"]["ser_definitions"],
        "classification_validation": classification_validation,
        "models": {key: models[key] for key in MODELS},
        "inversion_pairs": pairs,
        "high_wer_zero_semantic_cases": find_high_wer_zero_semantic_cases(
            utterance_metrics,
            threshold=high_wer_threshold,
        ),
    }
    return render_report(payload, top_per_pair=top_per_pair, top_zero=top_zero), payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--mapping-root", type=Path, default=DEFAULT_MAPPING_ROOT)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--t1-json", type=Path, default=DEFAULT_T1_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--high-wer-threshold", type=float, default=25.0)
    parser.add_argument("--top-per-pair", type=int, default=8)
    parser.add_argument("--top-zero", type=int, default=20)
    args = parser.parse_args()

    markdown, payload = generate(
        benchmark_root=args.benchmark_root,
        mapping_root=args.mapping_root,
        prediction_root=args.prediction_root,
        t1_json=args.t1_json,
        high_wer_threshold=args.high_wer_threshold,
        top_per_pair=args.top_per_pair,
        top_zero=args.top_zero,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown + "\n", encoding="utf-8")
    json_output = args.output.with_suffix(".json")
    json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(f"wrote {json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
