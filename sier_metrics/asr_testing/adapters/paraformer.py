from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .base import ASRResult, Issue, exception_result, generic_model_dir_issues


def funasr_text(result: Any) -> str:
    if isinstance(result, list) and result:
        return funasr_text(result[0])
    if isinstance(result, dict):
        return str(result.get("text") or result.get("preds") or result.get("value") or "")
    return "" if result is None else str(result)


class ParaformerAdapter:
    name = "paraformer"

    def __init__(self) -> None:
        self.model = None

    def validate_model_dir(self, model_dir: Path) -> list[Issue]:
        issues = generic_model_dir_issues(model_dir)
        if issues:
            return issues
        if not (model_dir / "configuration.json").exists() and not (model_dir / "config.yaml").exists():
            issues.append(Issue("warning", "configuration.json/config.yaml not found; FunASR may still load if model files are complete"))
        return issues

    def load(self, model_dir: Path, device: str, dtype: str | None = None, **kwargs: Any) -> None:
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise ImportError("Cannot import funasr. Install funasr for Paraformer.") from exc
        self.model = AutoModel(model=str(model_dir), device=device, disable_update=True)

    def transcribe(self, audio_path: Path, **kwargs: Any) -> ASRResult:
        if self.model is None:
            return ASRResult(success=False, error_type="RuntimeError", error_message="model is not loaded")
        started = time.perf_counter()
        try:
            result = self.model.generate(input=str(audio_path))
            text = funasr_text(result)
            return ASRResult(success=bool(text.strip()), text=text, infer_time_sec=time.perf_counter() - started, extra={"raw": result})
        except Exception as exc:
            return exception_result(exc, infer_time_sec=time.perf_counter() - started)

    def unload(self) -> None:
        self.model = None
