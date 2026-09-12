from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class Issue:
    level: str
    message: str


@dataclass
class ASRResult:
    success: bool
    text: str = ""
    language: str | None = None
    infer_time_sec: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseASRAdapter(Protocol):
    name: str

    def validate_model_dir(self, model_dir: Path) -> list[Issue]:
        ...

    def load(self, model_dir: Path, device: str, dtype: str | None = None, **kwargs: Any) -> None:
        ...

    def transcribe(self, audio_path: Path, **kwargs: Any) -> ASRResult:
        ...

    def unload(self) -> None:
        ...


def exception_result(exc: Exception, infer_time_sec: float | None = None) -> ASRResult:
    return ASRResult(
        success=False,
        infer_time_sec=infer_time_sec,
        error_type=exc.__class__.__name__,
        error_message=str(exc),
        extra={"traceback": traceback.format_exc(limit=5)},
    )


def generic_model_dir_issues(model_dir: Path) -> list[Issue]:
    if not model_dir.exists():
        return [Issue("error", f"model directory not found: {model_dir}")]
    if not model_dir.is_dir():
        return [Issue("error", f"model path is not a directory: {model_dir}")]
    try:
        next(model_dir.iterdir())
    except StopIteration:
        return [Issue("error", f"model directory is empty: {model_dir}")]
    return []
