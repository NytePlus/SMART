from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .audio_samples import DEFAULT_BENCHMARK_ROOT, read_jsonl

_SENSEVOICE_TAG_RE = re.compile(r"<\|[^|]*\|>")
_UNK_RE = re.compile(r"(?i)<\s*unk\s*>")
_PUNCT_RE = re.compile(r"[，。！？、；：,.!?;:\"'“”‘’()（）\[\]{}<>《》…—\-·]")
_ZERO_WIDTH_RE = re.compile(r"[​‌‍﻿]")


def normalize_asr_text(text: str) -> str:
    """Normalize reference/hypothesis text before WER tokenization."""
    text = unicodedata.normalize("NFKC", str(text).strip())
    text = _SENSEVOICE_TAG_RE.sub("", text)
    text = _UNK_RE.sub("", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _PUNCT_RE.sub("", text)
    return text.casefold()


def tokenize_asr_text(text: str) -> list[str]:
    """Tokenize normalized ASR text for mixed Chinese/English WER.

    CJK characters are evaluated one by one. Consecutive Latin letters/digits are
    evaluated as one word, so English is word-level instead of letter-level.
    Other non-space characters keep the previous character-level behavior.
    """
    normalized = normalize_asr_text(text)
    tokens: list[str] = []
    latin_buffer: list[str] = []

    def flush_latin_buffer() -> None:
        if latin_buffer:
            tokens.append("".join(latin_buffer))
            latin_buffer.clear()

    for ch in normalized:
        if ch.isspace():
            flush_latin_buffer()
            continue
        if ch.isascii() and ch.isalnum():
            latin_buffer.append(ch)
            continue
        flush_latin_buffer()
        tokens.append(ch)
    flush_latin_buffer()
    return tokens


def edit_distance(ref: list[str] | tuple[str, ...] | str, hyp: list[str] | tuple[str, ...] | str) -> tuple[int, int, int, int]:
    """Return (substitutions, insertions, deletions, ref_length)."""
    if not ref:
        return 0, len(hyp), 0, 0
    if not hyp:
        return 0, 0, len(ref), len(ref)

    rows = len(ref) + 1
    cols = len(hyp) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0] = i
    for j in range(1, cols):
        dp[0][j] = j

    for i, ref_ch in enumerate(ref, start=1):
        for j, hyp_ch in enumerate(hyp, start=1):
            cost = 0 if ref_ch == hyp_ch else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    i, j = len(ref), len(hyp)
    substitutions = insertions = deletions = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return substitutions, insertions, deletions, len(ref)


@dataclass(frozen=True)
class WerStats:
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    ref_chars: int = 0  # Backward-compatible name; now stores mixed WER token count.
    evaluated: int = 0
    missing_gt: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def wer(self) -> float | None:
        if self.ref_chars == 0:
            return None
        return 100.0 * self.errors / self.ref_chars

    def merge(self, other: WerStats) -> WerStats:
        return WerStats(
            substitutions=self.substitutions + other.substitutions,
            insertions=self.insertions + other.insertions,
            deletions=self.deletions + other.deletions,
            ref_chars=self.ref_chars + other.ref_chars,
            evaluated=self.evaluated + other.evaluated,
            missing_gt=self.missing_gt + other.missing_gt,
        )


def load_ground_truth_index(benchmark_root: Path) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    for dataset_dir in sorted(benchmark_root.iterdir()):
        gt_path = dataset_dir / "processed" / "gt.jsonl"
        if not gt_path.exists():
            continue
        dataset = dataset_dir.name
        for row in read_jsonl(gt_path):
            sample_id = str(row["id"])
            index[(dataset, sample_id)] = str(row.get("ground_truth") or "")
    return index


def load_benchmark_utterance_source_map(
    benchmark_root: Path,
) -> dict[tuple[str, str], tuple[str, str]]:
    """Map (benchmark_bucket, utterance_id) -> (source_dataset, source_segment_id)."""
    mapping: dict[tuple[str, str], tuple[str, str]] = {}
    for dataset_dir in sorted(benchmark_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        semantic_path = dataset_dir / "processed" / "semantic_xyz.jsonl"
        if not semantic_path.exists():
            continue
        bucket = dataset_dir.name
        per_id: dict[str, tuple[str, str]] = {}
        for row in read_jsonl(semantic_path):
            utterance_id = str(row["id"])
            if utterance_id in per_id:
                continue
            per_id[utterance_id] = (
                str(row.get("source_dataset") or ""),
                str(row.get("source_segment_id") or ""),
            )
        for utterance_id, source_key in per_id.items():
            mapping[(bucket, utterance_id)] = source_key
    return mapping


def _prediction_source_key(row: dict) -> tuple[str, str, str] | None:
    bucket = str(row.get("dataset") or "")
    extra = row.get("extra") or {}
    source_dataset = extra.get("source_dataset")
    source_segment_id = extra.get("source_segment_id")
    if source_dataset and source_segment_id:
        return bucket, str(source_dataset), str(source_segment_id)

    audio = str(row.get("audio") or "")
    marker = "/benchmark/benchmark/"
    if marker in audio and audio.endswith(".wav"):
        rel = audio.split(marker, 1)[1]
        parts = rel.split("/")
        if len(parts) >= 4 and parts[1] == "processed" and parts[2] == "audios":
            return parts[0], "__benchmark__", parts[3][:-4]
    return None


def load_model_predictions(
    prediction_root: Path,
    model_key: str,
    *,
    benchmark_root: Path | None = None,
) -> dict[tuple[str, str], str]:
    model_dir = prediction_root / model_key
    if not model_dir.is_dir():
        return {}

    by_source: dict[tuple[str, str, str], str] = {}
    by_gt: dict[tuple[str, str], str] = {}
    by_id: dict[tuple[str, str], str] = {}
    for path in sorted(model_dir.glob("*.jsonl")):
        bucket = path.stem
        for row in read_jsonl(path):
            if row.get("status") != "PASS":
                continue
            prediction = str(row.get("prediction") or "")
            source_key = _prediction_source_key(row)
            if source_key is not None:
                by_source[source_key] = prediction
            gt_key = normalize_asr_text(str(row.get("ground_truth") or ""))
            if gt_key:
                by_gt[(bucket, gt_key)] = prediction
            by_id[(bucket, str(row["id"]))] = prediction

    if benchmark_root is None:
        return by_id

    ground_truth = load_ground_truth_index(benchmark_root)
    utterance_sources = load_benchmark_utterance_source_map(benchmark_root)
    aligned: dict[tuple[str, str], str] = {}
    for key, ref_raw in ground_truth.items():
        bucket, utterance_id = key
        prediction = None
        source = utterance_sources.get(key)
        if source is not None:
            source_dataset, source_segment_id = source
            prediction = by_source.get((bucket, source_dataset, source_segment_id))
        if prediction is None:
            prediction = by_gt.get((bucket, normalize_asr_text(ref_raw)))
        if prediction is None:
            prediction = by_source.get((bucket, "__benchmark__", utterance_id))
        if prediction is None:
            prediction = by_id.get(key)
        if prediction is not None:
            aligned[key] = prediction
    return aligned


def compute_wer(
    predictions: dict[tuple[str, str], str],
    ground_truth: dict[tuple[str, str], str],
) -> WerStats:
    stats = WerStats()
    for key, ref_raw in ground_truth.items():
        hyp_raw = predictions.get(key)
        if hyp_raw is None:
            stats = WerStats(
                substitutions=stats.substitutions,
                insertions=stats.insertions,
                deletions=stats.deletions,
                ref_chars=stats.ref_chars,
                evaluated=stats.evaluated,
                missing_gt=stats.missing_gt + 1,
            )
            continue

        ref = tokenize_asr_text(ref_raw)
        hyp = tokenize_asr_text(hyp_raw)
        subs, ins, dels, ref_len = edit_distance(ref, hyp)
        stats = WerStats(
            substitutions=stats.substitutions + subs,
            insertions=stats.insertions + ins,
            deletions=stats.deletions + dels,
            ref_chars=stats.ref_chars + ref_len,
            evaluated=stats.evaluated + 1,
            missing_gt=stats.missing_gt,
        )
    return stats


def benchmark_ground_truth_size(ground_truth: dict[tuple[str, str], str]) -> int:
    return len(ground_truth)


def load_benchmark_ground_truth(benchmark_root: Path | None = None) -> dict[tuple[str, str], str]:
    root = benchmark_root or DEFAULT_BENCHMARK_ROOT
    return load_ground_truth_index(root)
