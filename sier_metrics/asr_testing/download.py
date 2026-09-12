from __future__ import annotations

import argparse
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DEFAULT_MODELS_CONFIG, TESTING_ROOT, ModelSpec, load_model_specs, rel_to_root, select_model_specs, write_json
from .reporting import print_download_table


OK_STATUSES = {"OK_DOWNLOADED", "OK_EXISTS"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dir_non_empty(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def dir_size(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    count = 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return count, total


def classify_exception(exc: Exception) -> str:
    message = str(exc).lower()
    if "permission" in message or "401" in message or "403" in message or "unauthorized" in message:
        return "FAILED_AUTH"
    if isinstance(exc, (socket.timeout, TimeoutError, ConnectionError)):
        return "FAILED_NETWORK"
    if "not found" in message or "404" in message or "does not exist" in message:
        return "FAILED_MODELSCOPE_UNAVAILABLE"
    if "network" in message or "connection" in message or "timed out" in message:
        return "FAILED_NETWORK"
    return "FAILED_UNKNOWN"


def status_for_spec(spec: ModelSpec) -> dict[str, Any]:
    files, size = dir_size(spec.local_dir)
    if not spec.enabled:
        status = "SKIPPED_DISABLED"
        note = "disabled in models.yaml"
    elif dir_non_empty(spec.local_dir):
        status = "OK_EXISTS"
        note = f"{files} files, {size / (1024 ** 3):.2f} GiB"
    elif not spec.repo_id:
        status = "SKIPPED_CONFIG_REQUIRED"
        note = "source.repo_id is empty and local model directory is absent"
    else:
        status = "MISSING"
        note = "local model directory is absent or empty"
    return {
        "model": spec.key,
        "display_name": spec.display_name,
        "status": status,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "local_dir": rel_to_root(spec.local_dir),
        "files": files,
        "size_bytes": size,
        "note": note,
    }


def download_spec(spec: ModelSpec, force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    row = status_for_spec(spec)
    if row["status"] == "SKIPPED_DISABLED":
        return row
    if row["status"] == "OK_EXISTS" and not force:
        return row
    if not spec.repo_id:
        return row
    if dry_run:
        row.update(status="DRY_RUN", note="would download from ModelScope")
        return row

    if force and spec.local_dir.exists():
        shutil.rmtree(spec.local_dir)
    spec.local_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        row.update(
            status="FAILED_DEPENDENCY",
            error="Cannot import modelscope. Install dependencies from benchmark/testing/requirements.txt.",
            error_type=exc.__class__.__name__,
        )
        return row

    try:
        kwargs: dict[str, Any] = {
            "model_id": spec.repo_id,
            "local_dir": str(spec.local_dir),
        }
        if spec.revision:
            kwargs["revision"] = spec.revision
        if spec.source.get("ignore_patterns"):
            kwargs["ignore_patterns"] = spec.source["ignore_patterns"]
        if spec.source.get("allow_patterns"):
            kwargs["allow_patterns"] = spec.source["allow_patterns"]
        if spec.source.get("max_workers"):
            kwargs["max_workers"] = int(spec.source["max_workers"])
        path = snapshot_download(**kwargs)
        files, size = dir_size(spec.local_dir)
        row.update(
            status="OK_DOWNLOADED",
            local_dir=rel_to_root(spec.local_dir),
            snapshot_path=str(path),
            files=files,
            size_bytes=size,
            note=f"downloaded {files} files, {size / (1024 ** 3):.2f} GiB",
            downloaded_at=utc_now(),
        )
    except TypeError:
        try:
            kwargs = {"model_id": spec.repo_id, "cache_dir": str(spec.local_dir.parent)}
            if spec.revision:
                kwargs["revision"] = spec.revision
            if spec.source.get("ignore_patterns"):
                kwargs["ignore_patterns"] = spec.source["ignore_patterns"]
            if spec.source.get("allow_patterns"):
                kwargs["allow_patterns"] = spec.source["allow_patterns"]
            if spec.source.get("max_workers"):
                kwargs["max_workers"] = int(spec.source["max_workers"])
            path = snapshot_download(**kwargs)
            snapshot_path = Path(path)
            if snapshot_path.resolve() != spec.local_dir.resolve():
                if spec.local_dir.exists():
                    shutil.rmtree(spec.local_dir)
                shutil.copytree(snapshot_path, spec.local_dir)
            files, size = dir_size(spec.local_dir)
            row.update(
                status="OK_DOWNLOADED",
                local_dir=rel_to_root(spec.local_dir),
                snapshot_path=str(snapshot_path),
                files=files,
                size_bytes=size,
                note=f"downloaded {files} files, {size / (1024 ** 3):.2f} GiB",
                downloaded_at=utc_now(),
            )
        except Exception as exc:
            row.update(status=classify_exception(exc), error=str(exc), error_type=exc.__class__.__name__)
    except Exception as exc:
        row.update(status=classify_exception(exc), error=str(exc), error_type=exc.__class__.__name__)
    return row


def write_download_report(rows: list[dict[str, Any]]) -> None:
    report = {
        "created_at": utc_now(),
        "root": str(TESTING_ROOT),
        "models": rows,
    }
    write_json(TESTING_ROOT / "reports" / "download_status.json", report)


def run_list(args: argparse.Namespace) -> int:
    specs = load_model_specs(Path(args.config))
    rows = [status_for_spec(spec) for spec in specs.values()]
    print_download_table(rows)
    return 0


def run_status(args: argparse.Namespace) -> int:
    specs = load_model_specs(Path(args.config))
    selected = select_model_specs(specs, model=args.model, all_models=args.all)
    rows = [status_for_spec(spec) for spec in selected]
    print_download_table(rows)
    write_download_report(rows)
    return 0 if all(row["status"] in OK_STATUSES or row["status"].startswith("SKIPPED") for row in rows) else 1


def run_download(args: argparse.Namespace) -> int:
    specs = load_model_specs(Path(args.config))
    selected = select_model_specs(specs, model=args.model, all_models=args.all)
    rows = [download_spec(spec, force=args.force, dry_run=args.dry_run) for spec in selected]
    print_download_table(rows)
    write_download_report(rows)
    failed = [row for row in rows if row["status"].startswith("FAILED")]
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and inspect ASR models from ModelScope.")
    parser.add_argument("--config", default=str(DEFAULT_MODELS_CONFIG), help="Path to models.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List configured models and current local status")

    status = subparsers.add_parser("status", help="Write and print local model status")
    status.add_argument("--model", default=None)
    status.add_argument("--all", action="store_true")

    download = subparsers.add_parser("download", help="Download models from ModelScope")
    download.add_argument("--model", default=None)
    download.add_argument("--all", action="store_true")
    download.add_argument("--force", action="store_true", help="Remove existing model dir before downloading")
    download.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        return run_list(args)
    if args.command == "status":
        return run_status(args)
    if args.command == "download":
        if not args.model and not args.all:
            parser.error("download requires --model NAME or --all")
        return run_download(args)
    parser.error(f"unknown command: {args.command}")
    return 2
