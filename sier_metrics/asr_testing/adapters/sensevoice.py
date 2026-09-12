from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .base import ASRResult, Issue, exception_result, generic_model_dir_issues
from .paraformer import funasr_text


class SenseVoiceAdapter:
    name = "sensevoice"

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
            raise ImportError("Cannot import funasr. Install funasr for SenseVoice.") from exc
        self.model = AutoModel(
            model=str(model_dir),
            device=device,
            trust_remote_code=bool(kwargs.get("trust_remote_code", True)),
            disable_update=True,
        )

    def transcribe(self, audio_path: Path, **kwargs: Any) -> ASRResult:
        if self.model is None:
            return ASRResult(success=False, error_type="RuntimeError", error_message="model is not loaded")
        started = time.perf_counter()
        try:
            language = kwargs.get("language", "auto")
            use_itn = bool(kwargs.get("use_itn", True))
            result = self.model.generate(input=str(audio_path), language=language, use_itn=use_itn)
            text = funasr_text(result)
            return ASRResult(success=bool(text.strip()), text=text, infer_time_sec=time.perf_counter() - started, extra={"raw": result})
        except TypeError:
            try:
                result = self.model.generate(input=str(audio_path))
                text = funasr_text(result)
                return ASRResult(success=bool(text.strip()), text=text, infer_time_sec=time.perf_counter() - started, extra={"raw": result})
            except Exception as exc:
                return exception_result(exc, infer_time_sec=time.perf_counter() - started)
        except Exception as exc:
            return exception_result(exc, infer_time_sec=time.perf_counter() - started)

    def unload(self) -> None:
        self.model = None
