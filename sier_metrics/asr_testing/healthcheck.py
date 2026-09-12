from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio_samples import DEFAULT_BENCHMARK_ROOT, AudioSample, select_samples
from .config import DEFAULT_HEALTHCHECK_CONFIG, DEFAULT_MODELS_CONFIG, TESTING_ROOT, load_healthcheck_config, load_model_specs, select_model_specs
from .registry import create_adapter
from .reporting import print_health_table, to_plain, write_json_report, write_markdown_health_report


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment(device: str) -> dict[str, Any]:
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "device": device}
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


def sample_to_row(sample: AudioSample) -> dict[str, Any]:
    return {
        "dataset": sample.dataset,
        "id": sample.sample_id,
        "audio": str(sample.audio_path),
        "ground_truth": sample.ground_truth,
        "duration_sec": sample.duration_sec,
    }


def check_one_model(spec, samples: list[AudioSample], device: str, dtype: str | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": spec.key,
        "display_name": spec.display_name,
        "status": "UNKNOWN",
        "model_dir": str(spec.local_dir),
        "sample_count": len(samples),
        "sample_successes": 0,
        "samples": [],
        "errors": [],
        "load_time_sec": None,
        "avg_infer_time_sec": None,
    }
    if not spec.enabled:
        row.update(status="SKIP_DISABLED", note="disabled in models.yaml")
        return row
    if not spec.local_dir.exists() or not any(spec.local_dir.iterdir()):
        if not spec.repo_id:
            row.update(status="SKIP_CONFIG", note="repo_id missing and local model directory absent")
        else:
            row.update(status="FAIL_MODEL_FILES", note="local model directory missing or empty")
        return row

    try:
        adapter = create_adapter(spec.adapter)
    except Exception as exc:
        row.update(status="FAIL_DEPENDENCY", error=f"cannot create adapter {spec.adapter}: {exc}")
        return row

    issues = adapter.validate_model_dir(spec.local_dir)
    errors = [issue.message for issue in issues if issue.level == "error"]
    warnings = [issue.message for issue in issues if issue.level != "error"]
    row["errors"].extend(errors)
    if warnings:
        row["warnings"] = warnings
    if errors:
        row.update(status="FAIL_MODEL_FILES", note="; ".join(errors[:2]))
        return row

    load_kwargs = dict(spec.load)
    configured_dtype = load_kwargs.pop("dtype", None)
    load_dtype = dtype or configured_dtype
    started = time.perf_counter()
    try:
        adapter.load(spec.local_dir, device=device, dtype=load_dtype, **load_kwargs)
        row["load_time_sec"] = time.perf_counter() - started
    except ImportError as exc:
        row.update(status="FAIL_DEPENDENCY", error=str(exc), load_time_sec=time.perf_counter() - started)
        return row
    except Exception as exc:
        row.update(status="FAIL_LOAD", error=str(exc), error_type=exc.__class__.__name__, load_time_sec=time.perf_counter() - started)
        return row

    infer_times: list[float] = []
    try:
        for sample in samples:
            result = adapter.transcribe(sample.audio_path, **spec.load)
            if result.infer_time_sec is not None:
                infer_times.append(result.infer_time_sec)
            if result.success:
                row["sample_successes"] += 1
            row["samples"].append(
                {
                    **sample_to_row(sample),
                    "status": "PASS" if result.success else "FAIL",
                    "text": result.text,
                    "language": result.language,
                    "infer_time_sec": result.infer_time_sec,
                    "error_type": result.error_type,
                    "error_message": result.error_message,
                }
            )
    finally:
        try:
            adapter.unload()
        except Exception:
            pass

    if infer_times:
        row["avg_infer_time_sec"] = sum(infer_times) / len(infer_times)
    if row["sample_successes"] == len(samples) and samples:
        row.update(status="PASS", note=f"{row['sample_successes']}/{len(samples)} samples OK")
    elif row["sample_successes"] > 0:
        row.update(status="WARN", note=f"{row['sample_successes']}/{len(samples)} samples OK")
    else:
        row.update(status="WARN", note="model loaded but no sample produced non-empty text")
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test local ASR models on benchmark audio.")
    parser.add_argument("--models-config", default=str(DEFAULT_MODELS_CONFIG))
    parser.add_argument("--healthcheck-config", default=str(DEFAULT_HEALTHCHECK_CONFIG))
    parser.add_argument("--model", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list-samples", action="store_true")
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument("--sample-policy", choices=["first", "random"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-duration-sec", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    hc = load_healthcheck_config(Path(args.healthcheck_config))
    defaults = hc.get("defaults", {})
    benchmark_root = args.benchmark_root or (TESTING_ROOT / hc.get("benchmark_root", "../benchmark")).resolve()
    if not benchmark_root.exists():
        benchmark_root = DEFAULT_BENCHMARK_ROOT
    datasets = args.datasets if args.datasets else defaults.get("datasets")
    limit = args.limit_per_dataset or int(defaults.get("limit_per_dataset", 1))
    policy = args.sample_policy or str(defaults.get("sample_policy", "first"))
    seed = args.seed if args.seed is not None else int(defaults.get("seed", 42))
    max_total = args.max_samples
    if args.smoke or defaults.get("smoke", True):
        max_total = max_total or int(defaults.get("max_smoke_samples", 8))

    samples = select_samples(
        benchmark_root=benchmark_root,
        datasets=datasets,
        limit_per_dataset=limit,
        sample_policy=policy,
        seed=seed,
        max_duration_sec=args.max_duration_sec,
        max_total=max_total,
    )
    if args.list_samples:
        for sample in samples:
            duration = "-" if sample.duration_sec is None else f"{sample.duration_sec:.2f}s"
            print(f"{sample.dataset}\t{sample.sample_id}\t{duration}\t{sample.audio_path}")
        return 0
    if not samples:
        print(f"No benchmark audio samples found under {benchmark_root}")
        return 1

    specs = load_model_specs(Path(args.models_config))
    selected = select_model_specs(specs, model=args.model, all_models=args.all)
    device = args.device or str(defaults.get("device", "cuda:0"))
    dtype = args.dtype or defaults.get("dtype")

    rows = [check_one_model(spec, samples=samples, device=device, dtype=dtype) for spec in selected]
    report = {
        "created_at": utc_now(),
        "root": str(TESTING_ROOT),
        "benchmark_root": str(benchmark_root),
        "environment": environment(device),
        "samples": [sample_to_row(sample) for sample in samples],
        "models": to_plain(rows),
    }
    reports = hc.get("reports", {})
    json_path = TESTING_ROOT / reports.get("json", "reports/healthcheck_latest.json")
    md_path = TESTING_ROOT / reports.get("markdown", "reports/healthcheck_latest.md")
    write_json_report(json_path, report)
    write_markdown_health_report(md_path, report)
    print_health_table(rows)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    return 0 if all(row["status"] in {"PASS", "WARN", "SKIP_CONFIG", "SKIP_DISABLED"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
