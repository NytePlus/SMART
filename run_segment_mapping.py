#!/usr/bin/env python3
"""Build canonical segment mappings from reference/hypothesis JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "segment_mapping"))

from segment_mapping_core import compute_mapping  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            for field in ("id", "reference", "hypothesis"):
                if field not in row:
                    raise ValueError(f"missing {field!r} at {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def error_segments(reference: str, ref: list[str], hyp: list[str], ops: list[str]) -> list[dict]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    span_ok = True
    for text in ref:
        start = reference.find(text, cursor) if text else cursor
        if start < 0:
            span_ok = False
            break
        end = start + len(text)
        spans.append((start, end))
        cursor = end
    separator = " " if " " in reference else ""
    result = []
    for index, (ref_text, hyp_text, op_text) in enumerate(zip(ref, hyp, ops)):
        if op_text and all(char == "=" for char in op_text):
            continue
        if span_ok and index < len(spans):
            start, end = spans[index]
            modified = reference[:start] + hyp_text + reference[end:]
        else:
            modified = separator.join(ref[:index] + [hyp_text] + ref[index + 1 :])
        result.append({
            "segment_index": index,
            "reference_segment": ref_text,
            "hypothesis_segment": hyp_text,
            "op_segment": op_text,
            "modified_reference": modified,
        })
    return result


def map_record(row: dict) -> dict:
    reference = str(row["reference"])
    hypothesis = str(row["hypothesis"])
    mapping = compute_mapping(reference, hypothesis)
    ref = mapping["mapped_reference"]
    hyp = mapping["mapped_hypothesis"]
    ops = mapping["mapped_ops"]
    output = {key: value for key, value in row.items() if key not in {"reference", "hypothesis"}}
    output.update({
        "id": str(row["id"]),
        "reference": reference,
        "hypothesis": hypothesis,
        "mapping": {
            "mapped_reference": ref,
            "mapped_hypothesis": hyp,
            "mapped_ops": ops,
            "mer": mapping.get("mer"),
            "aligned": mapping.get("aligned"),
        },
        "error_segments": error_segments(reference, ref, hyp, ops),
    })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("input ids must be unique")
    write_jsonl(args.output, [map_record(row) for row in rows])
    print(f"mapped {len(rows)} utterances -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
