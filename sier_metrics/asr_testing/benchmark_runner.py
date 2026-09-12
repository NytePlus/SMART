from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .adapters.base import ASRResult
from .audio_samples import DEFAULT_BENCHMARK_ROOT, AudioSample, available_datasets, samples_for_dataset
from .config import DEFAULT_MODELS_CONFIG, TESTING_ROOT, ModelSpec, load_model_specs, select_model_specs, write_json
from .registry import create_adapter

DEFAULT_DATASETS = ["X0Y0Z0", "X0Y0Z2", "X0Y2Z0", "X0Y2Z2", "X2Y0Z0", "X2Y0Z2", "X2Y2Z0", "X2Y2Z2"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment(device: str) -> dict[str, Any]:
    env: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(), "device": device}
    try:
        import torch

        env.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
        )
        if torch.cuda.is_available():
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        env.update({"torch": None, "cuda_available": False, "torch_error": str(exc)})
    return env


def jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {line_number} in {path}: {exc}") from exc
            if isinstance(value, dict):
                yield value


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        json.dump(row, f, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def sample_to_row(sample: AudioSample) -> dict[str, Any]:
    return {
        "dataset": sample.dataset,
        "id": sample.sample_id,
        "audio": str(sample.audio_path),
        "duration_sec": sample.duration_sec,
        "ground_truth": sample.ground_truth,
    }


def existing_pass_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for row in jsonl_records(path):
        if row.get("status") == "PASS" and row.get("id") is not None:
            ids.add(str(row["id"]))
    return ids


def result_row(spec: ModelSpec, sample: AudioSample, result: ASRResult) -> dict[str, Any]:
    return {
        "created_at": utc_now(),
        "model": spec.key,
        "display_name": spec.display_name,
        **sample_to_row(sample),
        "prediction": result.text,
        "language": result.language,
        "infer_time_sec": result.infer_time_sec,
        "status": "PASS" if result.success else "FAIL",
        "error_type": result.error_type,
        "error_message": result.error_message,
        "extra": result.extra,
    }


def failure_row(
    spec: ModelSpec,
    dataset: str | None,
    sample: AudioSample | None,
    status: str,
    error_type: str,
    error_message: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "created_at": utc_now(),
        "model": spec.key,
        "display_name": spec.display_name,
        "dataset": dataset,
        "status": status,
        "error_type": error_type,
        "error_message": error_message,
    }
    if sample is not None:
        row.update(sample_to_row(sample))
    if extra:
        row["extra"] = extra
    return row


def select_datasets(benchmark_root: Path, datasets: list[str] | None) -> list[str]:
    if datasets:
        return datasets
    available = available_datasets(benchmark_root)
    ordered = [dataset for dataset in DEFAULT_DATASETS if dataset in available]
    extras = [dataset for dataset in available if dataset not in ordered]
    return ordered + extras


def select_samples_for_run(
    benchmark_root: Path,
    datasets: list[str],
    limit_per_dataset: int | None,
    max_duration_sec: float | None,
) -> dict[str, list[AudioSample]]:
    selected: dict[str, list[AudioSample]] = {}
    for dataset in datasets:
        samples = samples_for_dataset(dataset, benchmark_root=benchmark_root, max_duration_sec=max_duration_sec)
        if limit_per_dataset is not None:
            samples = samples[:limit_per_dataset]
        selected[dataset] = samples
    return selected


def validate_model(spec: ModelSpec) -> tuple[bool, list[str], list[str]]:
    if not spec.enabled:
        return False, ["model disabled in models.yaml"], []
    if not spec.local_dir.exists() or not any(spec.local_dir.iterdir()):
        return False, [f"local model directory missing or empty: {spec.local_dir}"], []
    adapter = create_adapter(spec.adapter)
    issues = adapter.validate_model_dir(spec.local_dir)
    errors = [issue.message for issue in issues if issue.level == "error"]
    warnings = [issue.message for issue in issues if issue.level != "error"]
    return not errors, errors, warnings


def run_one_model(
    spec: ModelSpec,
    samples_by_dataset: dict[str, list[AudioSample]],
    output_dir: Path,
    device: str,
    dtype: str | None,
    resume: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model": spec.key,
        "display_name": spec.display_name,
        "status": "UNKNOWN",
        "model_dir": str(spec.local_dir),
        "datasets": {},
        "load_time_sec": None,
        "total_samples": 0,
        "written": 0,
        "skipped_existing": 0,
        "failures": 0,
        "warnings": [],
        "errors": [],
    }
    failures_path = output_dir / "runs" / "failures.jsonl"
    if sum(len(samples) for samples in samples_by_dataset.values()) == 0:
        summary.update(status="SKIP_NO_SAMPLES")
        return summary

    if not spec.enabled:
        message = "model disabled in models.yaml"
        summary.update(status="SKIP_DISABLED")
        summary["errors"].append(message)
        append_jsonl(failures_path, failure_row(spec, None, None, "SKIP_DISABLED", "DisabledModel", message))
        return summary

    if not spec.local_dir.exists() or not any(spec.local_dir.iterdir()):
        message = f"local model directory missing or empty: {spec.local_dir}"
        summary.update(status="FAIL_MODEL_FILES")
        summary["errors"].append(message)
        append_jsonl(failures_path, failure_row(spec, None, None, "FAIL_MODEL_FILES", "ModelFilesMissing", message))
        return summary

    try:
        adapter = create_adapter(spec.adapter)
    except Exception as exc:
        summary.update(status="FAIL_ADAPTER")
        summary["errors"].append(str(exc))
        append_jsonl(failures_path, failure_row(spec, None, None, "FAIL_ADAPTER", exc.__class__.__name__, str(exc)))
        return summary

    issues = adapter.validate_model_dir(spec.local_dir)
    errors = [issue.message for issue in issues if issue.level == "error"]
    warnings = [issue.message for issue in issues if issue.level != "error"]
    summary["warnings"].extend(warnings)
    if errors:
        summary.update(status="FAIL_MODEL_FILES")
        summary["errors"].extend(errors)
        for message in errors:
            append_jsonl(failures_path, failure_row(spec, None, None, "FAIL_MODEL_FILES", "ModelDirValidationError", message))
        return summary

    load_kwargs = dict(spec.load)
    configured_dtype = load_kwargs.pop("dtype", None)
    load_dtype = dtype or configured_dtype
    started = time.perf_counter()
    try:
        adapter.load(spec.local_dir, device=device, dtype=load_dtype, **load_kwargs)
        summary["load_time_sec"] = time.perf_counter() - started
    except ImportError as exc:
        summary.update(status="FAIL_DEPENDENCY", load_time_sec=time.perf_counter() - started)
        summary["errors"].append(str(exc))
        append_jsonl(failures_path, failure_row(spec, None, None, "FAIL_DEPENDENCY", exc.__class__.__name__, str(exc)))
        return summary
    except Exception as exc:
        summary.update(status="FAIL_LOAD", load_time_sec=time.perf_counter() - started)
        summary["errors"].append(str(exc))
        append_jsonl(failures_path, failure_row(spec, None, None, "FAIL_LOAD", exc.__class__.__name__, str(exc)))
        return summary

    try:
        for dataset, samples in samples_by_dataset.items():
            predictions_path = output_dir / "predictions" / spec.key / f"{dataset}.jsonl"
            skipped_ids = existing_pass_ids(predictions_path) if resume else set()
            dataset_summary = {
                "samples": len(samples),
                "written": 0,
                "skipped_existing": 0,
                "failures": 0,
                "output": str(predictions_path),
            }
            summary["total_samples"] += len(samples)
            print(f"[{spec.key}] dataset={dataset} samples={len(samples)} output={predictions_path}", flush=True)

            for index, sample in enumerate(samples, start=1):
                if sample.sample_id in skipped_ids:
                    dataset_summary["skipped_existing"] += 1
                    summary["skipped_existing"] += 1
                    continue
                print(f"[{spec.key}] {dataset} {index}/{len(samples)} id={sample.sample_id}", flush=True)
                try:
                    result = adapter.transcribe(sample.audio_path, **spec.load)
                except Exception as exc:
                    result = ASRResult(success=False, error_type=exc.__class__.__name__, error_message=str(exc))
                row = result_row(spec, sample, result)
                append_jsonl(predictions_path, row)
                dataset_summary["written"] += 1
                summary["written"] += 1
                if not result.success:
                    dataset_summary["failures"] += 1
                    summary["failures"] += 1
                    append_jsonl(failures_path, row)
            summary["datasets"][dataset] = dataset_summary
    finally:
        try:
            adapter.unload()
        except Exception:
            pass

    summary["status"] = "PASS" if summary["failures"] == 0 else "WARN"
    return summary


def run_asr_benchmark(
    models_config: Path,
    benchmark_root: Path,
    output_dir: Path,
    models: list[str] | None,
    all_models: bool,
    datasets: list[str] | None,
    device: str,
    dtype: str | None,
    resume: bool,
    limit_per_dataset: int | None,
    max_duration_sec: float | None,
) -> dict[str, Any]:
    benchmark_root = benchmark_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if models and all_models:
        raise ValueError("Use either --model/--models or --all, not both")

    specs = load_model_specs(models_config)
    if models:
        selected_specs = []
        for model in models:
            selected_specs.extend(select_model_specs(specs, model=model, all_models=False))
    else:
        selected_specs = select_model_specs(specs, model=None, all_models=all_models)

    dataset_names = select_datasets(benchmark_root, datasets)
    samples_by_dataset = select_samples_for_run(
        benchmark_root=benchmark_root,
        datasets=dataset_names,
        limit_per_dataset=limit_per_dataset,
        max_duration_sec=max_duration_sec,
    )

    run_config = {
        "created_at": utc_now(),
        "testing_root": str(TESTING_ROOT),
        "models_config": str(models_config),
        "benchmark_root": str(benchmark_root),
        "output_dir": str(output_dir),
        "models": [spec.key for spec in selected_specs],
        "datasets": dataset_names,
        "device": device,
        "dtype": dtype,
        "resume": resume,
        "limit_per_dataset": limit_per_dataset,
        "max_duration_sec": max_duration_sec,
        "environment": environment(device),
        "sample_counts": {dataset: len(samples) for dataset, samples in samples_by_dataset.items()},
    }
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    write_json(runs_dir / "run_config.json", run_config)

    summaries = []
    for spec in selected_specs:
        print(f"=== Running model {spec.key} ({spec.display_name}) ===", flush=True)
        summaries.append(
            run_one_model(
                spec=spec,
                samples_by_dataset=samples_by_dataset,
                output_dir=output_dir,
                device=device,
                dtype=dtype,
                resume=resume,
            )
        )

    report = {"created_at": utc_now(), "run_config": run_config, "models": summaries}
    write_json(runs_dir / "run_summary.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local ASR models on benchmark datasets and write prediction JSONL files.")
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=TESTING_ROOT / "reports" / "asr_full")
    parser.add_argument("--model", action="append", default=None, help="Model key to run. Can be passed multiple times.")
    parser.add_argument("--models", nargs="*", default=None, help="Model keys to run.")
    parser.add_argument("--all", action="store_true", help="Run all configured models.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument("--max-duration-sec", type=float, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    requested_models = []
    if args.model:
        requested_models.extend(args.model)
    if args.models:
        requested_models.extend(args.models)

    report = run_asr_benchmark(
        models_config=args.models_config,
        benchmark_root=args.benchmark_root,
        output_dir=args.output_dir,
        models=requested_models or None,
        all_models=args.all,
        datasets=args.datasets,
        device=args.device,
        dtype=args.dtype,
        resume=args.resume,
        limit_per_dataset=args.limit_per_dataset,
        max_duration_sec=args.max_duration_sec,
    )
    print(f"Run summary: {Path(args.output_dir).resolve() / 'runs' / 'run_summary.json'}")
    failed = [row for row in report["models"] if row.get("status") not in {"PASS", "WARN", "SKIP_DISABLED", "SKIP_NO_SAMPLES"}]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
