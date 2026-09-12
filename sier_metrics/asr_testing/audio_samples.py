from __future__ import annotations

import json
import random
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import TESTING_ROOT


DEFAULT_BENCHMARK_ROOT = TESTING_ROOT.parent / "benchmark"


@dataclass(frozen=True)
class AudioSample:
    dataset: str
    sample_id: str
    audio_path: Path
    ground_truth: str
    duration_sec: float | None = None


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
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


def wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except Exception:
        return None


def available_datasets(benchmark_root: Path = DEFAULT_BENCHMARK_ROOT) -> list[str]:
    if not benchmark_root.exists():
        return []
    return sorted(
        p.name
        for p in benchmark_root.iterdir()
        if p.is_dir() and (p / "processed" / "gt.jsonl").exists()
    )


def samples_for_dataset(
    dataset: str,
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT,
    max_duration_sec: float | None = None,
) -> list[AudioSample]:
    processed = benchmark_root / dataset / "processed"
    gt_path = processed / "gt.jsonl"
    audio_dir = processed / "audios"
    if not gt_path.exists() or not audio_dir.exists():
        return []

    samples: list[AudioSample] = []
    for rec in read_jsonl(gt_path):
        sample_id = str(rec.get("id", ""))
        if not sample_id:
            continue
        audio_path = audio_dir / f"{sample_id}.wav"
        if not audio_path.exists():
            continue
        duration = wav_duration(audio_path)
        if max_duration_sec is not None and duration is not None and duration > max_duration_sec:
            continue
        samples.append(
            AudioSample(
                dataset=dataset,
                sample_id=sample_id,
                audio_path=audio_path,
                ground_truth=str(rec.get("ground_truth", "")),
                duration_sec=duration,
            )
        )
    return samples


def select_samples(
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT,
    datasets: Iterable[str] | None = None,
    limit_per_dataset: int = 1,
    sample_policy: str = "first",
    seed: int = 42,
    max_duration_sec: float | None = None,
    max_total: int | None = None,
) -> list[AudioSample]:
    dataset_names = list(datasets) if datasets else available_datasets(benchmark_root)
    rng = random.Random(seed)
    selected: list[AudioSample] = []
    for dataset in dataset_names:
        samples = samples_for_dataset(dataset, benchmark_root, max_duration_sec=max_duration_sec)
        if sample_policy == "random":
            rng.shuffle(samples)
        elif sample_policy != "first":
            raise ValueError(f"Unsupported sample policy: {sample_policy}")
        selected.extend(samples[:limit_per_dataset])
        if max_total is not None and len(selected) >= max_total:
            return selected[:max_total]
    return selected
