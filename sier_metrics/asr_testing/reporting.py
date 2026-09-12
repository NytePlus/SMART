from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .config import write_json

console = Console()


def to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return to_plain(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    return value


def print_download_table(rows: list[dict[str, Any]]) -> None:
    table = Table(title="ASR ModelScope download status")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Repo ID")
    table.add_column("Local path")
    table.add_column("Note")
    for row in rows:
        table.add_row(
            str(row.get("model", "")),
            str(row.get("status", "")),
            str(row.get("repo_id") or "-"),
            str(row.get("local_dir", "")),
            str(row.get("note") or row.get("error") or ""),
        )
    console.print(table)


def print_health_table(rows: list[dict[str, Any]]) -> None:
    table = Table(title="ASR model health check")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Load")
    table.add_column("Samples")
    table.add_column("Avg infer")
    table.add_column("Note")
    for row in rows:
        load = row.get("load_time_sec")
        avg = row.get("avg_infer_time_sec")
        table.add_row(
            str(row.get("model", "")),
            str(row.get("status", "")),
            "-" if load is None else f"{load:.2f}s",
            f"{row.get('sample_successes', 0)}/{row.get('sample_count', 0)}",
            "-" if avg is None else f"{avg:.2f}s",
            str(row.get("note") or row.get("error") or ""),
        )
    console.print(table)


def write_markdown_health_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# ASR Model Health Check", ""]
    env = report.get("environment", {})
    lines.extend(
        [
            "## Environment",
            "",
            f"- Python: {env.get('python', '-')}",
            f"- Torch: {env.get('torch', '-')}",
            f"- CUDA available: {env.get('cuda_available', '-')}",
            f"- Device: {env.get('device', '-')}",
            "",
            "## Summary",
            "",
            "| Model | Status | Load | Samples | Avg infer | Note |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for model in report.get("models", []):
        load = model.get("load_time_sec")
        avg = model.get("avg_infer_time_sec")
        lines.append(
            "| {model} | {status} | {load} | {samples} | {avg} | {note} |".format(
                model=model.get("model", ""),
                status=model.get("status", ""),
                load="-" if load is None else f"{load:.2f}s",
                samples=f"{model.get('sample_successes', 0)}/{model.get('sample_count', 0)}",
                avg="-" if avg is None else f"{avg:.2f}s",
                note=str(model.get("note") or model.get("error") or "").replace("|", "\\|"),
            )
        )
    lines.append("")
    for model in report.get("models", []):
        lines.extend([f"## {model.get('model', '')}", ""])
        lines.append(f"Status: **{model.get('status', '')}**")
        lines.append("")
        for sample in model.get("samples", []):
            lines.append(f"- `{sample.get('dataset')}/{sample.get('id')}`: {sample.get('status')} — {sample.get('text', '')[:200]}")
        if model.get("errors"):
            lines.append("")
            lines.append("Errors:")
            for error in model.get("errors", []):
                lines.append(f"- {error}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json_report(path: Path, report: dict[str, Any]) -> None:
    write_json(path, to_plain(report))
