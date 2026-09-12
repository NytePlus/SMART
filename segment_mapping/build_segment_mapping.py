#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 model-test 下的预测 JSONL（id, target, pred），为每个 id 生成 segment mapping。

用法示例：
    python build_segment_mapping.py \
        --input /hpc_stor03/sjtu_home/xiaoyu.gu/workspace/data-prepare/model-test/canvas-speech/qwen/pred.norm.jsonl \
        --dataset canvas-speech

输出格式（每个 id 一条记录）：
{
  "id": "0000001",
  "target": "...",
  "pred": "...",
  "mapping": {
    "mapped_reference": [...],
    "mapped_hypothesis": [...],
    "mapped_ops": [...],
    "mer": 0.837209,
    "aligned": "..."
  },
  "segments": [
    {
      "index": 0,
      "reference_segment": "...",
      "hypothesis_segment": "...",
      "op_segment": "...",
      "modified_target": "..."
    }
  ]
}
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segment_mapping_core import compute_mapping
from text_normalization import TextNormalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SemaScore-style segment mapping for model predictions."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL path with id, target, pred fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output segment_mapping.json path. If not given, inferred from --input.",
    )
    parser.add_argument(
        "--dataset",
        default="canvas-speech",
        help="Dataset name used when inferring output path (default: canvas-speech).",
    )
    parser.add_argument(
        "--normalize-for-mapping",
        action="store_true",
        help="Normalize target/pred before computing segment mapping.",
    )
    parser.add_argument(
        "--normalizer-engine",
        default="auto",
        choices=["auto", "wetext", "nemo", "rule", "none"],
        help="Normalizer engine used with --normalize-for-mapping (default: auto).",
    )
    parser.add_argument(
        "--normalizer-language",
        default="auto",
        choices=["auto", "zh", "en"],
        help="Language hint for normalization (default: auto).",
    )
    parser.add_argument(
        "--normalizer-fallback",
        default="none",
        choices=["rule", "none"],
        help="Fallback when mature normalizer is unavailable (default: none).",
    )
    parser.add_argument(
        "--normalizer-spacing",
        default="smart",
        choices=["smart", "single-space", "remove-all", "none"],
        help="Spacing policy after normalization (default: smart).",
    )
    parser.add_argument(
        "--normalizer-report",
        action="store_true",
        help="Print normalization changed/fallback counts.",
    )
    return parser.parse_args()


def infer_output_path(input_path: Path, dataset: str) -> Path:
    """根据输入路径自动推导输出路径到 segments-mapping/{dataset}/{model}/segment_mapping.json。"""
    base_dir = Path("/hpc_stor03/sjtu_home/xiaoyu.gu/workspace/data-prepare/segments-mapping")
    model_name = input_path.parent.name  # e.g. qwen / doubao
    return base_dir / dataset / model_name / "segment_mapping.json"


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {line_number} in {path}: {exc}") from exc
    return records


def build_segments(ref_segs, hyp_segs, op_segs, target=""):
    """只为错误 segment 生成 modified_target 等信息。"""
    segments = []
    spans = []
    cursor = 0
    span_ok = bool(target)
    for ref_seg in ref_segs:
        start = target.find(ref_seg, cursor) if span_ok and ref_seg else cursor
        if start < 0:
            span_ok = False
            break
        end = start + len(ref_seg)
        spans.append((start, end))
        cursor = end

    separator = " " if " " in target else ""
    for i, (ref_seg, hyp_seg, op_seg) in enumerate(zip(ref_segs, hyp_segs, op_segs)):
        if all(c == "=" for c in op_seg):
            continue
        if span_ok and i < len(spans):
            start, end = spans[i]
            modified_target = target[:start] + hyp_seg + target[end:]
        else:
            modified_target = separator.join(ref_segs[:i] + [hyp_seg] + ref_segs[i + 1 :])
        segments.append(
            {
                "index": i,
                "reference_segment": ref_seg,
                "hypothesis_segment": hyp_seg,
                "op_segment": op_seg,
                "modified_target": modified_target,
            }
        )
    return segments


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = args.output if args.output else infer_output_path(input_path, args.dataset)

    print(f"[1/3] 读取 {input_path}")
    records = read_jsonl(input_path)
    print(f"       共 {len(records)} 条记录")

    print("[2/3] 计算 segment mapping")
    normalizer = None
    norm_stats = {
        "target_changed": 0,
        "pred_changed": 0,
        "engines": {},
        "warnings": 0,
    }
    if args.normalize_for_mapping:
        normalizer = TextNormalizer(
            engine=args.normalizer_engine,
            language=args.normalizer_language,
            fallback=args.normalizer_fallback,
            spacing=args.normalizer_spacing,
        )
        print(
            "       启用 normalization: "
            f"engine={args.normalizer_engine}, "
            f"language={args.normalizer_language}, "
            f"fallback={args.normalizer_fallback}, "
            f"spacing={args.normalizer_spacing}"
        )

    output = []
    for idx, rec in enumerate(records, 1):
        rec_id = rec.get("id", "")
        raw_target = rec.get("target", "")
        raw_pred = rec.get("pred", "")

        if not raw_target or not raw_pred:
            print(f"       跳过 {rec_id}：target 或 pred 为空")
            continue

        target = raw_target
        pred = raw_pred
        normalization_meta = None
        if normalizer is not None:
            norm_target = normalizer.normalize(raw_target)
            norm_pred = normalizer.normalize(raw_pred)
            target = norm_target.text
            pred = norm_pred.text
            norm_stats["target_changed"] += int(norm_target.changed)
            norm_stats["pred_changed"] += int(norm_pred.changed)
            norm_stats["warnings"] += len(norm_target.warnings) + len(norm_pred.warnings)
            for engine in (norm_target.engine, norm_pred.engine):
                norm_stats["engines"][engine] = norm_stats["engines"].get(engine, 0) + 1
            normalization_meta = {
                "target_engine": norm_target.engine,
                "pred_engine": norm_pred.engine,
                "target_changed": norm_target.changed,
                "pred_changed": norm_pred.changed,
                "warnings": norm_target.warnings + norm_pred.warnings,
            }

        if not target or not pred:
            print(f"       跳过 {rec_id}：normalization 后 target 或 pred 为空")
            continue

        mapping = compute_mapping(target, pred)
        ref_segs = mapping["mapped_reference"]
        hyp_segs = mapping["mapped_hypothesis"]
        op_segs = mapping["mapped_ops"]

        item = {
            "id": rec_id,
            "target": raw_target,
            "pred": raw_pred,
            "mapping": {
                "mapped_reference": ref_segs,
                "mapped_hypothesis": hyp_segs,
                "mapped_ops": op_segs,
                "mer": mapping.get("mer"),
                "aligned": mapping.get("aligned"),
            },
            "segments": build_segments(ref_segs, hyp_segs, op_segs, target),
        }
        if normalization_meta is not None:
            item["mapping_target"] = target
            item["mapping_pred"] = pred
            item["normalization"] = normalization_meta
        output.append(item)

        if idx % 100 == 0:
            print(f"       已处理 {idx}/{len(records)}")

    if args.normalizer_report and normalizer is not None:
        print("       normalization report:")
        print(f"         target_changed: {norm_stats['target_changed']}")
        print(f"         pred_changed: {norm_stats['pred_changed']}")
        print(f"         engines: {norm_stats['engines']}")
        print(f"         warning_count: {norm_stats['warnings']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[3/3] 写入 {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"✅ 完成，共输出 {len(output)} 条记录")


if __name__ == "__main__":
    main()
