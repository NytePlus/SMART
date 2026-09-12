"""Compute s-SER / c-SER / w-SER with shared numerator/denominator conventions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _text_units import (
    error_chunk_count_in_hypothesis,
    segment_error_unit_counts,
    text_chunk_count,
    text_word_count,
)

from asr_testing.eval_subset import (
    dedupe_pred_rows_by_source,
    dedupe_rows_by_source_error_key,
    filter_rows_by_utterance,
)
from asr_testing.wer import edit_distance, tokenize_asr_text


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


def load_input_keys(model_dir: Path) -> set[tuple[str, str]]:
    return {
        classification_key(row)
        for row in read_jsonl(model_dir / "llm_classification_input.jsonl")
    }


CLASSIFICATION_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "target",
    "pred",
    "reference_segment",
    "hypothesis_segment",
    "op_segment",
)


def validate_classification_rows(
    input_rows: list[dict[str, Any]],
    classification_rows: list[dict[str, Any]],
) -> None:
    """Reject stale results that reuse an (id, focus_id) for different content."""
    inputs = {classification_key(row): row for row in dedupe_classification_rows(input_rows)}
    results = {
        classification_key(row): row for row in dedupe_classification_rows(classification_rows)
    }
    missing = sorted(set(inputs) - set(results))
    mismatches: list[tuple[tuple[str, str], list[str]]] = []
    for key in sorted(set(inputs) & set(results)):
        differing_fields = [
            field
            for field in CLASSIFICATION_FINGERPRINT_FIELDS
            if str(inputs[key].get(field) or "") != str(results[key].get(field) or "")
        ]
        if differing_fields:
            mismatches.append((key, differing_fields))

    if not missing and not mismatches:
        return
    examples = ", ".join(
        f"{key[0]}/{key[1]}:{'/'.join(fields)}" for key, fields in mismatches[:3]
    )
    raise ValueError(
        "classification results do not match current LLM input: "
        f"missing={len(missing)}, content_mismatches={len(mismatches)}"
        + (f"; examples={examples}" if examples else "")
    )


def filter_rows_by_composites(
    rows: list[dict[str, Any]],
    allowed_composites: set[str],
) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("id") or "") in allowed_composites]


def _pct(numerator: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return 100.0 * numerator / denominator


def _sum_segment_units(rows: list[dict[str, Any]], *, semantic_only: bool = False) -> dict[str, int]:
    totals = {
        "error_chunks": 0,
        "non_error_chunks": 0,
        "error_words": 0,
        "non_error_words": 0,
    }
    for row in rows:
        if semantic_only and not is_semantic_true(row):
            continue
        units = segment_error_unit_counts(
            str(row.get("reference_segment") or ""),
            str(row.get("op_segment") or ""),
        )
        for key in totals:
            totals[key] += units[key]
    return totals


def _sum_segment_edit_operations(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Sum WER-compatible S/I/D operations over the supplied error segments."""
    totals = {
        "substitutions": 0,
        "insertions": 0,
        "deletions": 0,
        "errors": 0,
    }
    for row in rows:
        reference = tokenize_asr_text(str(row.get("reference_segment") or ""))
        hypothesis = tokenize_asr_text(str(row.get("hypothesis_segment") or ""))
        substitutions, insertions, deletions, _ = edit_distance(reference, hypothesis)
        totals["substitutions"] += substitutions
        totals["insertions"] += insertions
        totals["deletions"] += deletions
        totals["errors"] += substitutions + insertions + deletions
    return totals


def _sum_hypothesis_error_chunks(rows: list[dict[str, Any]]) -> int:
    """Sum hyp-side chunks containing errors or anchoring ref-side deletions."""
    return sum(
        error_chunk_count_in_hypothesis(
            str(row.get("hypothesis_segment") or ""),
            str(row.get("op_segment") or ""),
        )
        for row in rows
    )


def compute_ser_variants(
    model_dir: Path,
    *,
    eval_utterance_ids: set[str] | None = None,
    allowed_composites: set[str] | None = None,
    composite_source_map: dict[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """
    s-SER = semantic error segments / total chunks (target text)
    c-SER = ref-side semantic error chunks (with error words only) / total chunks
    c-SER2 = hyp-side semantic error chunks (including deletion anchors) / total chunks
    w-SER = semantic error words / total words (target text)
    w-MER2 = WER-compatible edit errors in semantic error segments / total words
    """
    pred_rows = read_jsonl(model_dir / "pred.raw.jsonl")
    classification_rows = read_jsonl(model_dir / "results" / "classification_results.jsonl")

    input_rows = read_jsonl(model_dir / "llm_classification_input.jsonl")
    if input_rows:
        validate_classification_rows(input_rows, classification_rows)

    if allowed_composites is not None:
        pred_rows = filter_rows_by_composites(pred_rows, allowed_composites)
        classification_rows = filter_rows_by_composites(classification_rows, allowed_composites)
    if eval_utterance_ids is not None:
        pred_rows = filter_rows_by_utterance(pred_rows, eval_utterance_ids)
        classification_rows = filter_rows_by_utterance(classification_rows, eval_utterance_ids)

    input_keys = {classification_key(row) for row in input_rows}
    if input_keys:
        classification_rows = [
            row for row in classification_rows if classification_key(row) in input_keys
        ]

    unique_classification_rows = dedupe_classification_rows(classification_rows)
    duplicate_benchmark_error_rows = len(unique_classification_rows)
    duplicate_source_error_rows = 0
    duplicate_source_utterances = 0

    if composite_source_map:
        unique_classification_rows, duplicate_source_error_rows = dedupe_rows_by_source_error_key(
            unique_classification_rows,
            composite_source_map,
        )
        pred_rows, duplicate_source_utterances = dedupe_pred_rows_by_source(
            pred_rows,
            composite_source_map,
        )

    semantic_rows = [row for row in unique_classification_rows if is_semantic_true(row)]
    semantic_error_spans = len(semantic_rows)
    error_spans = len(unique_classification_rows)

    total_target_chunks = sum(text_chunk_count(row.get("target", "")) for row in pred_rows)
    total_target_words = sum(text_word_count(row.get("target", "")) for row in pred_rows)

    all_error_units = _sum_segment_units(unique_classification_rows)
    semantic_error_units = _sum_segment_units(unique_classification_rows, semantic_only=True)
    non_semantic_rows = [row for row in unique_classification_rows if not is_semantic_true(row)]
    non_semantic_error_units = _sum_segment_units(non_semantic_rows)
    semantic_edit_units = _sum_segment_edit_operations(semantic_rows)
    non_semantic_edit_units = _sum_segment_edit_operations(non_semantic_rows)
    semantic_hyp_error_chunks = _sum_hypothesis_error_chunks(semantic_rows)
    non_semantic_hyp_error_chunks = _sum_hypothesis_error_chunks(non_semantic_rows)
    error_segment_hyp_error_chunks = semantic_hyp_error_chunks + non_semantic_hyp_error_chunks

    semantic_error_chunks = semantic_error_units["error_chunks"]
    semantic_error_words = semantic_error_units["error_words"]
    non_semantic_error_chunks = non_semantic_error_units["error_chunks"]
    non_semantic_error_words = non_semantic_error_units["error_words"]
    error_segment_error_chunks = all_error_units["error_chunks"]
    error_segment_error_words = all_error_units["error_words"]

    s_ser = _pct(semantic_error_spans, total_target_chunks)
    c_ser = _pct(semantic_error_chunks, total_target_chunks)
    c_ser2 = _pct(semantic_hyp_error_chunks, total_target_chunks)
    w_ser = _pct(semantic_error_words, total_target_words)
    w_mer2 = _pct(semantic_edit_units["errors"], total_target_words)

    return {
        "utterances": len(pred_rows),
        "total_target_chunks": total_target_chunks,
        "total_target_words": total_target_words,
        "classification_rows": len(classification_rows),
        "duplicate_classification_rows": len(classification_rows) - duplicate_benchmark_error_rows,
        "duplicate_source_error_rows": duplicate_source_error_rows,
        "duplicate_source_utterances": duplicate_source_utterances,
        "error_spans": error_spans,
        "semantic_error_spans": semantic_error_spans,
        "semantic_error_chunks": semantic_error_chunks,
        "semantic_hyp_error_chunks": semantic_hyp_error_chunks,
        "semantic_error_words": semantic_error_words,
        "non_semantic_error_chunks": non_semantic_error_chunks,
        "non_semantic_hyp_error_chunks": non_semantic_hyp_error_chunks,
        "non_semantic_error_words": non_semantic_error_words,
        "error_segment_error_chunks": error_segment_error_chunks,
        "error_segment_hyp_error_chunks": error_segment_hyp_error_chunks,
        "error_segment_error_words": error_segment_error_words,
        "non_semantic_error_spans": error_spans - semantic_error_spans,
        "s_ser": s_ser,
        "c_ser": c_ser,
        "c_ser2": c_ser2,
        "w_ser": w_ser,
        "w_mer2": w_mer2,
        "semantic_edit_errors": semantic_edit_units["errors"],
        "semantic_edit_substitutions": semantic_edit_units["substitutions"],
        "semantic_edit_insertions": semantic_edit_units["insertions"],
        "semantic_edit_deletions": semantic_edit_units["deletions"],
        "non_semantic_edit_errors": non_semantic_edit_units["errors"],
        "non_semantic_edit_substitutions": non_semantic_edit_units["substitutions"],
        "non_semantic_edit_insertions": non_semantic_edit_units["insertions"],
        "non_semantic_edit_deletions": non_semantic_edit_units["deletions"],
        # Backward-compatible aliases used by older scripts/reports.
        "ser": s_ser,
        "target_segments": total_target_chunks,
        "semantic_error_target_segments": semantic_error_chunks,
        "better_ser": c_ser,
    }
