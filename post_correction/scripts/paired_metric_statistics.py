#!/usr/bin/env python3
"""Paired utterance-level uncertainty for official WER and c-SEER2."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import re
import sys
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [percentile(ordered, 0.025), percentile(ordered, 0.975)]


def paired_statistics(
    old_num: list[int],
    new_num: list[int],
    repaired_num: list[int],
    introduced_num: list[int],
    denominator: list[int],
    rng: random.Random,
    samples: int,
) -> dict[str, Any]:
    if not (
        len(old_num)
        == len(new_num)
        == len(repaired_num)
        == len(introduced_num)
        == len(denominator)
    ):
        raise ValueError("paired metric inputs have different lengths")
    if any(
        after != old - repaired + introduced
        for old, after, repaired, introduced in zip(
            old_num, new_num, repaired_num, introduced_num
        )
    ):
        raise ValueError("repair accounting must satisfy after = old - repaired + new")
    n = len(denominator)
    old_samples: list[float] = []
    new_samples: list[float] = []
    repaired_samples: list[float] = []
    introduced_samples: list[float] = []
    differences: list[float] = []
    reductions: list[float] = []
    for _ in range(samples):
        sampled = [rng.randrange(n) for _ in range(n)]
        sampled_den = sum(denominator[index] for index in sampled)
        old_rate_sample = 100.0 * sum(old_num[index] for index in sampled) / sampled_den
        new_rate_sample = 100.0 * sum(new_num[index] for index in sampled) / sampled_den
        repaired_rate_sample = (
            100.0 * sum(repaired_num[index] for index in sampled) / sampled_den
        )
        introduced_rate_sample = (
            100.0 * sum(introduced_num[index] for index in sampled) / sampled_den
        )
        difference = new_rate_sample - old_rate_sample
        old_samples.append(old_rate_sample)
        new_samples.append(new_rate_sample)
        repaired_samples.append(repaired_rate_sample)
        introduced_samples.append(introduced_rate_sample)
        differences.append(difference)
        if old_rate_sample:
            reductions.append(-100.0 * difference / old_rate_sample)

    paired_delta = [new - old for old, new in zip(old_num, new_num)]
    observed = abs(sum(paired_delta))
    extreme = 0
    for _ in range(samples):
        permuted = abs(sum(
            delta if rng.getrandbits(1) else -delta for delta in paired_delta
        ))
        extreme += permuted >= observed
    p_value = (1 + extreme) / (samples + 1)

    old_rate = 100.0 * sum(old_num) / sum(denominator)
    new_rate = 100.0 * sum(new_num) / sum(denominator)
    repaired_rate = 100.0 * sum(repaired_num) / sum(denominator)
    introduced_rate = 100.0 * sum(introduced_num) / sum(denominator)
    return {
        "utterances": n,
        "old": old_rate,
        "old_95ci": interval(old_samples),
        "after": new_rate,
        "after_95ci": interval(new_samples),
        "repaired": repaired_rate,
        "repaired_95ci": interval(repaired_samples),
        "new": introduced_rate,
        "new_95ci": interval(introduced_samples),
        "old_error_units": sum(old_num),
        "after_error_units": sum(new_num),
        "repaired_error_units": sum(repaired_num),
        "new_error_units": sum(introduced_num),
        "paired_change_points": new_rate - old_rate,
        "paired_change_95ci": interval(differences),
        "relative_reduction_pct": 100.0 * (old_rate - new_rate) / old_rate,
        "relative_reduction_95ci": interval(reductions),
        "paired_randomization_p_two_sided": p_value,
        "bootstrap_samples": samples,
    }


def _alignment_error_events(ref: list[str], hyp: list[str]) -> Counter[tuple[str, int]]:
    """Return WER errors keyed to stable reference positions and boundaries."""
    rows = len(ref) + 1
    cols = len(hyp) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0] = i
    for j in range(1, cols):
        dp[0][j] = j
    for i, ref_token in enumerate(ref, start=1):
        for j, hyp_token in enumerate(hyp, start=1):
            cost = 0 if ref_token == hyp_token else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    events: Counter[tuple[str, int]] = Counter()
    i, j = len(ref), len(hyp)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
            0 if ref[i - 1] == hyp[j - 1] else 1
        ):
            if ref[i - 1] != hyp[j - 1]:
                events[("ref", i - 1)] += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            events[("ref", i - 1)] += 1
            i -= 1
        else:
            events[("boundary", i)] += 1
            j -= 1
    return events


def event_turnover(
    old_events: Counter[Any], new_events: Counter[Any]
) -> tuple[int, int]:
    persistent = old_events & new_events
    return sum((old_events - persistent).values()), sum((new_events - persistent).values())


_FOCUS_RE = re.compile(r'<focus\s+id="[^"]+">')


def _semantic_event_spans(row: dict[str, Any], count: int) -> list[tuple[int, int]]:
    structured = str(row.get("structured_prediction") or "")
    match = _FOCUS_RE.search(structured)
    if match:
        start = match.start()
        end = start + len(str(row.get("reference_segment") or ""))
    else:
        start = int(row.get("segment_index") or 0)
        end = start + 1
    return [(start, end)] * count


def _spans_compatible(left: tuple[int, int], right: tuple[int, int]) -> bool:
    left_start, left_end = left
    right_start, right_end = right
    if left_start == left_end or right_start == right_end:
        return left_start == right_start and left_end == right_end
    return max(left_start, right_start) < min(left_end, right_end)


def semantic_turnover(
    old_spans: list[tuple[int, int]], new_spans: list[tuple[int, int]]
) -> tuple[int, int]:
    """Match semantic c-SEER2 units by overlapping reference-local spans."""
    owner_by_new: dict[int, int] = {}

    def match(old_index: int, seen: set[int]) -> bool:
        for new_index, new_span in enumerate(new_spans):
            if new_index in seen or not _spans_compatible(old_spans[old_index], new_span):
                continue
            seen.add(new_index)
            owner = owner_by_new.get(new_index)
            if owner is None or match(owner, seen):
                owner_by_new[new_index] = old_index
                return True
        return False

    matched = sum(match(index, set()) for index in range(len(old_spans)))
    return len(old_spans) - matched, len(new_spans) - matched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-prepare-root", type=Path, required=True)
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--after-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()

    testing_scripts = args.data_prepare_root / "benchmark" / "testing" / "scripts"
    metric_scripts = args.data_prepare_root / "scripts" / "26.8.2"
    sys.path[:0] = [str(testing_scripts), str(metric_scripts)]
    from asr_testing import wer
    from _ser_metrics import (
        classification_key,
        dedupe_classification_rows,
        is_semantic_true,
        validate_classification_rows,
    )
    from _text_units import error_chunk_count_in_hypothesis, text_chunk_count

    old_pred = {str(row["id"]): row for row in read_jsonl(args.old_dir / "pred.raw.jsonl")}
    new_pred = {str(row["id"]): row for row in read_jsonl(args.after_dir / "pred.raw.jsonl")}
    if set(old_pred) != set(new_pred):
        raise ValueError("old and after prediction IDs differ")

    def semantic_events(model_dir: Path) -> dict[str, list[tuple[int, int]]]:
        inputs = read_jsonl(model_dir / "llm_classification_input.jsonl")
        results = read_jsonl(model_dir / "results" / "classification_results.jsonl")
        validate_classification_rows(inputs, results)
        input_keys = {classification_key(row) for row in inputs}
        rows = [
            row for row in dedupe_classification_rows(results)
            if classification_key(row) in input_keys and is_semantic_true(row)
        ]
        totals: dict[str, list[tuple[int, int]]] = {row_id: [] for row_id in old_pred}
        for row in rows:
            count = error_chunk_count_in_hypothesis(
                str(row.get("hypothesis_segment") or ""),
                str(row.get("op_segment") or ""),
            )
            totals[str(row["id"])].extend(_semantic_event_spans(row, count))
        return totals

    old_semantic = semantic_events(args.old_dir)
    new_semantic = semantic_events(args.after_dir)

    benchmark_root = args.data_prepare_root / "benchmark" / "benchmark"
    sys.path.insert(0, str(metric_scripts))
    import generate_metrics_table_t1 as t1
    source_map = t1.load_composite_source_map(benchmark_root)

    groups: dict[str, list[str]] = {"Canvas-Speech": [], "TEDLIUM-3": [], "Average": []}
    for row_id in sorted(old_pred):
        source_dataset = source_map[row_id][0]
        corpus = "TEDLIUM-3" if source_dataset.startswith("TEDLIUM_release-3") else "Canvas-Speech"
        groups[corpus].append(row_id)
        groups["Average"].append(row_id)

    output: dict[str, Any] = {
        "seed": args.seed,
        "samples": args.samples,
        "unit": "paired utterance",
        "significance_test": "two-sided paired randomization",
        "repair_accounting": {
            "identity": "after = old - repaired + new",
            "wer": "mixed-token errors matched by reference position or insertion boundary",
            "c_ser2": "semantic hypothesis-error chunks matched by overlapping reference-local span",
            "rates": "old, after, repaired, and new use the metric's official denominator",
        },
        "groups": {},
    }
    rng = random.Random(args.seed)
    for group, ids in groups.items():
        references = [str(old_pred[row_id]["target"]) for row_id in ids]
        ref_tokens = [len(wer.tokenize_asr_text(text)) for text in references]

        def wer_events(rows: dict[str, dict[str, Any]]) -> list[Counter[tuple[str, int]]]:
            values: list[Counter[tuple[str, int]]] = []
            for row_id, reference in zip(ids, references):
                ref = wer.tokenize_asr_text(reference)
                hyp = wer.tokenize_asr_text(str(rows[row_id]["pred"]))
                values.append(_alignment_error_events(ref, hyp))
            return values

        chunks = [text_chunk_count(text) for text in references]
        old_wer_events = wer_events(old_pred)
        new_wer_events = wer_events(new_pred)
        wer_turnovers = [
            event_turnover(old, new)
            for old, new in zip(old_wer_events, new_wer_events)
        ]
        semantic_turnovers = [
            semantic_turnover(old_semantic[row_id], new_semantic[row_id])
            for row_id in ids
        ]
        output["groups"][group] = {
            "wer": paired_statistics(
                [sum(events.values()) for events in old_wer_events],
                [sum(events.values()) for events in new_wer_events],
                [turnover[0] for turnover in wer_turnovers],
                [turnover[1] for turnover in wer_turnovers],
                ref_tokens,
                rng,
                args.samples,
            ),
            "c_ser2": paired_statistics(
                [len(old_semantic[row_id]) for row_id in ids],
                [len(new_semantic[row_id]) for row_id in ids],
                [turnover[0] for turnover in semantic_turnovers],
                [turnover[1] for turnover in semantic_turnovers],
                chunks,
                rng,
                args.samples,
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
