#!/usr/bin/env python3
"""Generate proof examples E1–E6 for benchmark metrics documentation."""

from __future__ import annotations

import _path_setup  # noqa: F401

import argparse
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jieba

from asr_testing.config import ASR_FULL_METRICS, TESTING_ROOT
from asr_testing.eval_subset import load_composite_source_map
from asr_testing.wer import edit_distance, normalize_asr_text

ROOT = TESTING_ROOT.parents[1]
DEFAULT_BENCHMARK_ROOT = ROOT / "benchmark" / "benchmark"
DEFAULT_MAPPING_ROOT = ROOT / "segments-mapping" / "benchmark"
DEFAULT_MODEL = "qwen3-asr"
DEFAULT_OUTPUT = ASR_FULL_METRICS / "examples" / "metrics_examples.md"


@dataclass(frozen=True)
class ExampleSpec:
    label: str
    utterance_id: str
    focus_id: int | None
    description: str


EXAMPLE_SPECS: tuple[ExampleSpec, ...] = (
    ExampleSpec("E1", "X0Y0Z2:0000002", 2, "多错误段、语义/非语义混合"),
    ExampleSpec("E2", "X0Y2Z2:0000001", None, "长句、ASR 续写"),
    ExampleSpec("E3", "X2Y2Z2:0000011", None, "短句、单语义错误"),
    ExampleSpec("E4", "X2Y2Z2:0000001", 0, "低 WER、仅 1 个非语义错误段"),
    ExampleSpec("E5", "X0Y0Z0:0000001", 0, "严重漏识别"),
    ExampleSpec("E6", "X0Y2Z0:0000008", 19, "段长 37 长删除（source 去重后唯一）"),
)


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


def _is_punctuation_or_symbol(token: str) -> bool:
    return all(unicodedata.category(ch).startswith(("P", "S")) for ch in token)


def jieba_segment_count(text: str) -> int:
    tokens = []
    for token in jieba.lcut(str(text or "")):
        stripped = token.strip()
        if not stripped or _is_punctuation_or_symbol(stripped):
            continue
        tokens.append(stripped)
    return len(tokens)


def is_semantic_true(row: dict[str, Any]) -> bool:
    classification = row.get("classification")
    if isinstance(classification, dict):
        return bool(classification.get("is_semantic_error"))
    return False


def classification_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("id") or ""), str(row.get("focus_id") or ""))


def dedupe_classification_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        deduped[classification_key(row)] = row
    return list(deduped.values())


def load_segment_mapping_index(model_dir: Path) -> dict[str, dict[str, Any]]:
    path = model_dir / "segment_mapping.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {str(item.get("id") or ""): item for item in payload}
    return {}


def pick_focus_row(
    rows: list[dict[str, Any]],
    *,
    focus_id: int | None,
) -> dict[str, Any]:
    if focus_id is not None:
        for row in rows:
            if int(row.get("focus_id") or -1) == focus_id:
                return row
    for row in rows:
        if is_semantic_true(row):
            return row
    return rows[0] if rows else {}


def build_example(
    spec: ExampleSpec,
    *,
    model_dir: Path,
    composite_source_map: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    utterance_id = spec.utterance_id
    pred_row = next(
        (row for row in read_jsonl(model_dir / "pred.raw.jsonl") if row.get("id") == utterance_id),
        {},
    )
    utterance_rows = dedupe_classification_rows(
        [
            row
            for row in read_jsonl(model_dir / "results" / "classification_results.jsonl")
            if row.get("id") == utterance_id
        ]
    )
    focus_row = pick_focus_row(utterance_rows, focus_id=spec.focus_id)

    source_dataset, source_segment_id = composite_source_map.get(utterance_id, ("", ""))
    mapping = load_segment_mapping_index(model_dir).get(utterance_id, {})
    mer = None
    if isinstance(mapping.get("mapping"), dict):
        mer = mapping["mapping"].get("mer")

    full_ref = str(pred_row.get("target") or focus_row.get("target") or "")
    full_hyp = str(pred_row.get("pred") or focus_row.get("pred") or "")
    ref_norm = normalize_asr_text(full_ref)
    hyp_norm = normalize_asr_text(full_hyp)
    subs, ins, dels, ref_len = edit_distance(ref_norm, hyp_norm)
    utterance_wer = 100.0 * (subs + ins + dels) / ref_len if ref_len else None

    total_segments = jieba_segment_count(full_ref)
    error_spans = len(utterance_rows)
    semantic = sum(1 for row in utterance_rows if is_semantic_true(row))
    classification = (
        focus_row.get("classification") if isinstance(focus_row.get("classification"), dict) else {}
    )

    return {
        "label": spec.label,
        "description": spec.description,
        "utterance_id": utterance_id,
        "source_dataset": source_dataset or "--",
        "source_segment_id": source_segment_id or "--",
        "full_ref": full_ref,
        "full_hyp": full_hyp,
        "focus_row": focus_row,
        "jieba_len": jieba_segment_count(focus_row.get("reference_segment", "")),
        "is_semantic": is_semantic_true(focus_row),
        "semantic_reason": classification.get("reason") or focus_row.get("llm_response") or "",
        "mer": mer,
        "utterance_wer": utterance_wer,
        "total_segments": total_segments,
        "error_spans": error_spans,
        "semantic": semantic,
        "non_semantic": error_spans - semantic,
    }


def render_example(example: dict[str, Any]) -> str:
    row = example["focus_row"]
    mer_value = example["mer"]
    mer_text = f"{mer_value:.3f}" if isinstance(mer_value, (int, float)) else "--"
    wer_text = f"{example['utterance_wer']:.2f}%" if example["utterance_wer"] is not None else "--"
    jieba_note = (
        f"（jieba 段长={example['jieba_len']}）" if example["jieba_len"] else ""
    )
    lines = [
        f"### [{example['label']}] {example['utterance_id']} — {example['description']}",
        "",
        f"**source**: {example['source_dataset']}:{example['source_segment_id']}",
        "",
        "| 项 | 值 |",
        "|---|---|",
        "| 模型 | Qwen3-ASR |",
        f"| **总 ref** | {example['full_ref']} |",
        f"| **总 hyp** | {example['full_hyp']} |",
        f"| focus_id | {row.get('focus_id', '--')} |",
        f"| **错误段 ref{jieba_note}** | {row.get('reference_segment', '')} |",
        f"| **错误段 hyp** | {row.get('hypothesis_segment', '')} |",
        f"| op | `{row.get('op_segment', '')}` |",
        f"| 语义? | **{'是' if example['is_semantic'] else '否'}** — {example['semantic_reason']} |",
        f"| 本句 MER | {mer_text} |",
        f"| 本句 WER | {wer_text} |",
        f"| **本句段级** | 段总和={example['total_segments']}，错误段={example['error_spans']}，"
        f"语义={example['semantic']}，非语义={example['non_semantic']} |",
        "",
    ]
    return "\n".join(lines)


def generate_examples(
    *,
    model_dir: Path,
    benchmark_root: Path,
    specs: tuple[ExampleSpec, ...] = EXAMPLE_SPECS,
) -> str:
    composite_source_map = load_composite_source_map(benchmark_root)
    parts = [
        "## 正确性证明样例（Qwen3-ASR）",
        "",
        "**[EX-NOTE]** 每条样例包含 utterance 级 **总 ref / 总 hyp**，以及其中一个代表性 **错误段 ref / 错误段 hyp**。",
        "",
    ]
    for spec in specs:
        example = build_example(spec, model_dir=model_dir, composite_source_map=composite_source_map)
        parts.append(render_example(example))
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--mapping-root", type=Path, default=DEFAULT_MAPPING_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    markdown = generate_examples(
        model_dir=args.mapping_root / args.model,
        benchmark_root=args.benchmark_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
