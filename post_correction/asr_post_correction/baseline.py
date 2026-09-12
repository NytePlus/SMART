from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .io import prediction_text, read_jsonl


NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*|[零〇一二两三四五六七八九十百千万亿]+")
VALID_CATEGORIES = {"entity", "homophone", "term", "disfluency", "typo"}
RUBRIC_DIMENSIONS = ("S1", "S2", "S3", "S4", "C1", "C2")


@dataclass(frozen=True)
class BaselineEdit:
    segment_ref: str
    original_fragment: str
    replacement: str
    category: str
    reason: str
    start: int
    end: int


def render_baseline_messages(prompt: dict[str, Any], history: str, hypothesis: str) -> list[dict[str, str]]:
    payload = {
        "context_ground_truth_history": history,
        "segment": {"segment_ref": "s1", "text": hypothesis},
    }
    return [
        {"role": "system", "content": str(prompt["system"])},
        {
            "role": "user",
            "content": str(prompt["user"]).replace(
                "{{PAYLOAD}}", json.dumps(payload, ensure_ascii=False)
            ),
        },
    ]


def render_split_baseline_messages(
    prompt: dict[str, Any],
    stage: str,
    history: str,
    hypothesis: str,
    candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Render one stage without exposing the current reference transcript."""
    if stage not in {"detector", "corrector"}:
        raise ValueError(f"unknown split baseline stage: {stage}")
    payload: dict[str, Any] = {
        "context_ground_truth_history": history,
        "segment": {"segment_ref": "s1", "text": hypothesis},
    }
    if stage == "corrector":
        candidate_key = str(prompt.get("candidate_payload_key", "rubric_candidates"))
        payload[candidate_key] = candidates or []
    return [
        {"role": "system", "content": str(prompt[f"{stage}_system"])},
        {
            "role": "user",
            "content": str(prompt[f"{stage}_user"]).replace(
                "{{PAYLOAD}}", json.dumps(payload, ensure_ascii=False)
            ),
        },
    ]


def validate_rubric_candidates(
    hypothesis: str,
    raw_candidates: Any,
    *,
    max_candidates: int = 400,
    require_rubric: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep only extractive, uniquely located detector candidates.

    The default preserves the v1.1 six-dimension rubric contract. Ablation
    prompts can set ``require_rubric=False`` to test span detection without
    asking the detector to produce semantic rubric judgments.
    """
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    items = raw_candidates if isinstance(raw_candidates, list) else []
    seen: set[tuple[str, int, int]] = set()
    for index, item in enumerate(items[:max_candidates]):
        reason = ""
        if not isinstance(item, dict):
            rejected.append({"index": index, "reason": "not_an_object", "candidate": item})
            continue
        fragment = str(item.get("original_fragment", ""))
        rubric = item.get("rubric")
        try:
            start = int(item.get("start", -1))
            end = int(item.get("end", -1))
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            start, end, confidence = -1, -1, 0.0
            reason = "invalid_position_or_confidence"
        if not reason and str(item.get("segment_ref", "")) != "s1":
            reason = "wrong_segment_ref"
        elif not reason and (not fragment or hypothesis.count(fragment) != 1):
            reason = "original_not_unique"
        elif not reason and (start < 0 or end <= start or hypothesis[start:end] != fragment):
            reason = "stale_offset"
        elif not reason and require_rubric and (
            not isinstance(rubric, dict) or set(rubric) != set(RUBRIC_DIMENSIONS)
        ):
            reason = "invalid_rubric"
        elif not reason and not 0.0 <= confidence <= 1.0:
            reason = "invalid_confidence"
        key = (fragment, start, end)
        if not reason and key in seen:
            reason = "duplicate_candidate"
        if reason:
            rejected.append({"index": index, "reason": reason, "candidate": item})
            continue
        seen.add(key)
        accepted_item = {
            "segment_ref": "s1",
            "original_fragment": fragment,
            "start": start,
            "end": end,
            "confidence": confidence,
            "suspected_error_type": str(item.get("suspected_error_type", "")),
            "reason": str(item.get("reason", "")),
        }
        if require_rubric:
            accepted_item["rubric"] = rubric
        accepted.append(accepted_item)
    return accepted, rejected


def restrict_edits_to_candidates(
    raw_edits: Any, candidates: list[dict[str, Any]]
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Reject corrector edits that were not authorized by the detector."""
    items = raw_edits if isinstance(raw_edits, list) else []
    allowed = {
        (str(item["segment_ref"]), str(item["original_fragment"]))
        for item in candidates
    }
    accepted: list[Any] = []
    rejected: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        key = (
            str(item.get("segment_ref", "")),
            str(item.get("original_fragment", "")),
        ) if isinstance(item, dict) else ("", "")
        if key not in allowed:
            rejected.append({
                "index": index,
                "reason": "not_detector_authorized",
                "edit": item,
            })
            continue
        accepted.append(item)
    return accepted, rejected


def apply_baseline_edits(
    hypothesis: str, raw_edits: Any, *, max_edits: int = 400
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[BaselineEdit] = []
    rejected: list[dict[str, Any]] = []
    items = raw_edits if isinstance(raw_edits, list) else []
    for index, item in enumerate(items[:max_edits]):
        reject_reason = ""
        if not isinstance(item, dict):
            rejected.append({"index": index, "reason": "not_an_object", "edit": item})
            continue
        segment_ref = str(item.get("segment_ref", ""))
        original = str(item.get("original_fragment", ""))
        replacement = str(item.get("replacement", ""))
        category = str(item.get("category", ""))
        reason = str(item.get("reason", ""))
        occurrences = hypothesis.count(original) if original else 0
        if segment_ref != "s1":
            reject_reason = "wrong_segment_ref"
        elif not original:
            reject_reason = "empty_original"
        elif occurrences != 1:
            reject_reason = "original_not_unique"
        elif original == replacement:
            reject_reason = "no_change"
        elif category not in VALID_CATEGORIES:
            reject_reason = "invalid_category"
        elif len(reason) > 12:
            reject_reason = "reason_too_long"
        elif replacement == "" and original == hypothesis:
            reject_reason = "whole_segment_deletion"
        elif category == "disfluency" and not _strict_duplicate_deletion(original, replacement):
            reject_reason = "non_duplicate_disfluency"
        elif NUMBER_RE.findall(original) != NUMBER_RE.findall(replacement):
            reject_reason = "number_change"
        if reject_reason:
            rejected.append({"index": index, "reason": reject_reason, "edit": item})
            continue
        start = hypothesis.index(original)
        end = start + len(original)
        if any(max(start, edit.start) < min(end, edit.end) for edit in accepted):
            rejected.append({"index": index, "reason": "overlapping_edit", "edit": item})
            continue
        accepted.append(
            BaselineEdit(segment_ref, original, replacement, category, reason, start, end)
        )

    corrected = hypothesis
    for edit in sorted(accepted, key=lambda value: value.start, reverse=True):
        corrected = corrected[: edit.start] + edit.replacement + corrected[edit.end :]
    return corrected, [asdict(edit) for edit in accepted], rejected


def _strict_duplicate_deletion(original: str, replacement: str) -> bool:
    if not replacement or len(original) < 2:
        return False
    if len(original) % 2 == 0:
        half = len(original) // 2
        return original[:half] == original[half:] and replacement == original[:half]
    parts = original.split()
    return len(parts) == 2 and parts[0] == parts[1] and replacement == parts[0]


def load_headline_rows(data_prepare_root: str | Path, model_key: str) -> list[dict[str, Any]]:
    root = Path(data_prepare_root).resolve()
    benchmark_root = root / "benchmark" / "benchmark"
    prediction_root = root / "benchmark" / "testing" / "reports" / "asr_full" / "predictions"
    semantic_by_composite: dict[str, dict[str, Any]] = {}
    target_by_composite: dict[str, str] = {}
    for path in sorted(benchmark_root.glob("*/processed/semantic_xyz.jsonl")):
        bucket = path.parents[1].name
        gt_path = path.with_name("gt.jsonl")
        for row in read_jsonl(gt_path):
            target_by_composite[f"{bucket}:{row['id']}"] = str(
                row.get("ground_truth", row.get("target", ""))
            )
        for row in read_jsonl(path):
            composite_id = f"{bucket}:{row['id']}"
            if str(row.get("source_dataset")) == "AISHELL-4":
                continue
            semantic_by_composite[composite_id] = row

    testing_scripts = root / "benchmark" / "testing" / "scripts"
    testing_scripts_text = str(testing_scripts)
    if testing_scripts_text not in sys.path:
        sys.path.insert(0, testing_scripts_text)
    from asr_testing.wer import load_model_predictions

    aligned_predictions = load_model_predictions(
        prediction_root, model_key, benchmark_root=benchmark_root
    )
    rows = []
    for composite_id in sorted(semantic_by_composite):
        source = semantic_by_composite[composite_id]
        rows.append(
            {
                "id": composite_id,
                "source_dataset": str(source.get("source_dataset", "")),
                "source_segment_id": str(source.get("source_segment_id", "")),
                "history": str(source.get("history", "")),
                "suffix": str(source.get("suffix", "")),
                "target": target_by_composite[composite_id],
                "pred": aligned_predictions[tuple(composite_id.split(":", 1))],
            }
        )
    if len(rows) != 467:
        raise ValueError(f"headline scope drifted: expected 467 rows, got {len(rows)}")
    return rows


def validate_headline_alignment(
    data_prepare_root: str | Path, model_key: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    root = Path(data_prepare_root).resolve()
    testing_scripts = root / "benchmark" / "testing" / "scripts"
    testing_scripts_text = str(testing_scripts)
    if testing_scripts_text not in sys.path:
        sys.path.insert(0, testing_scripts_text)
    from asr_testing.wer import compute_wer

    ground_truth = {
        tuple(row["id"].split(":", 1)): str(row["target"]) for row in rows
    }
    predictions = {
        tuple(row["id"].split(":", 1)): str(row["pred"]) for row in rows
    }
    stats = compute_wer(predictions, ground_truth)
    if stats.evaluated != len(rows) or stats.missing_gt:
        raise ValueError(
            f"headline alignment failed: evaluated={stats.evaluated}, missing={stats.missing_gt}"
        )
    if stats.wer is None or stats.wer > 50:
        raise ValueError(f"headline WER preflight failed: implausible WER={stats.wer}")
    return {
        "wer": stats.wer,
        "errors": stats.errors,
        "reference_tokens": stats.ref_chars,
        "evaluated": stats.evaluated,
    }
