#!/usr/bin/env python3
"""生成不包含旧标签的 rubric 人工标注 JSONL 模板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rubric_eval.config import load_rubric
from rubric_eval.io_utils import read_jsonl, write_jsonl


def _blank_item() -> dict:
    return {
        "answer": None,
        "reason_code": None,
        "confidence": None,
        "evidence": [],
    }


def _annotation_field(config: dict) -> str:
    if config["rubric_version"].startswith("minimal-"):
        return "human_annotation_minimal_v1"
    return "human_rubric_annotation"


def build_annotation_record(record: dict, config: dict) -> dict:
    visible_input = {
        key: value
        for key, value in record.items()
        if not key.startswith("human_annotation") and key != "human_rubric_annotation"
    }
    visible_input[_annotation_field(config)] = {
        "annotator_id": "",
        "preliminary_overall_label": None,
        "preliminary_overall_confidence": None,
        "rubric_judgment": {
            "rubric_version": config["rubric_version"],
            "items": {
                question["id"]: _blank_item()
                for question in config["questions"]
            },
            "change_types": [],
        },
        "final_comment": "",
        "status": "待标注"
    }
    return visible_input


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--rubric",
        type=Path,
        default=ROOT / "config" / "rubric_v1.json",
    )
    args = parser.parse_args()

    config = load_rubric(args.rubric)
    records = [build_annotation_record(record, config) for record in read_jsonl(args.input)]
    count = write_jsonl(args.output, records)
    print(f"已生成 {count} 条盲标模板：{args.output}")


if __name__ == "__main__":
    main()
