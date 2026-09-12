#!/usr/bin/env python3
"""校验 rubric 人工标注，并比较人工整体标签与确定性聚合结果。"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rubric_eval.aggregation import aggregate_rubric_result
from rubric_eval.config import load_rubric
from rubric_eval.io_utils import read_jsonl
from rubric_eval.schema import validate_rubric_judgment


OVERALL_LABEL_MAP = {
    "有实质语义后果": "CONSEQUENTIAL",
    "无实质语义后果": "INCONSEQUENTIAL",
    "不可评分": "UNSCORABLE",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--rubric",
        type=Path,
        default=ROOT / "config" / "rubric_v1.json",
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    config = load_rubric(args.rubric)
    records = read_jsonl(args.input)
    issues: list[dict] = []
    completed = 0
    valid_completed = 0
    agreement = Counter()

    for record in records:
        sample_id = str(record.get("sample_id", ""))
        annotation = record.get("human_rubric_annotation")
        if annotation is None:
            annotation = record.get("human_annotation_minimal_v1")
        if not isinstance(annotation, dict):
            issues.append({"sample_id": sample_id, "errors": ["缺少人工 rubric 标注字段"]})
            continue
        if annotation.get("status") != "已完成":
            if args.require_complete:
                issues.append({"sample_id": sample_id, "errors": ["status 不是 已完成"]})
            continue
        completed += 1
        errors: list[str] = []
        if not str(annotation.get("annotator_id", "")).strip():
            errors.append("annotator_id 不能为空")
        overall = annotation.get("preliminary_overall_label")
        if overall not in OVERALL_LABEL_MAP:
            errors.append("preliminary_overall_label 不是允许值")
        if annotation.get("preliminary_overall_confidence") not in {"高", "中", "低"}:
            errors.append("preliminary_overall_confidence 必须是 高/中/低")
        judgment = annotation.get("rubric_judgment")
        if not isinstance(judgment, dict):
            errors.append("rubric_judgment 必须是对象")
        else:
            errors.extend(validate_rubric_judgment(judgment, record, config))

        if errors:
            issues.append({"sample_id": sample_id, "errors": errors})
            continue
        aggregated = aggregate_rubric_result(judgment, config)["final_label"]
        preliminary = OVERALL_LABEL_MAP[overall]
        agreement[(preliminary, aggregated)] += 1
        valid_completed += 1

    print(f"总记录数: {len(records)}")
    print(f"已完成: {completed}")
    print(f"有效完成: {valid_completed}")
    print(f"整体标签 vs rubric聚合: {dict(agreement)}")
    if issues:
        for issue in issues[:20]:
            print(issue)
        raise SystemExit(f"发现 {len(issues)} 条标注问题")
    print("人工标注校验通过")


if __name__ == "__main__":
    main()
