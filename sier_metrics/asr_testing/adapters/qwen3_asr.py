from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .base import ASRResult, Issue, exception_result, generic_model_dir_issues


class Qwen3ASRAdapter:
    name = "qwen3_asr"

    def __init__(self) -> None:
        self.model = None

    def validate_model_dir(self, model_dir: Path) -> list[Issue]:
        issues = generic_model_dir_issues(model_dir)
        if issues:
            return issues
        if not (model_dir / "config.json").exists():
            issues.append(Issue("error", f"missing config.json in {model_dir}"))
        if not any(model_dir.glob("*.safetensors")):
            issues.append(Issue("error", f"no .safetensors files found in {model_dir}"))
        return issues

    def load(self, model_dir: Path, device: str, dtype: str | None = None, **kwargs: Any) -> None:
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise ImportError("Cannot import qwen_asr/torch. Install qwen-asr and torch.") from exc

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            None: torch.bfloat16,
        }
        self.model = Qwen3ASRModel.from_pretrained(
            str(model_dir),
            dtype=dtype_map.get(dtype, torch.bfloat16),
            device_map=device,
            max_inference_batch_size=int(kwargs.get("batch_size", 1)),
            max_new_tokens=int(kwargs.get("max_new_tokens", 256)),
        )

    def transcribe(self, audio_path: Path, **kwargs: Any) -> ASRResult:
        if self.model is None:
            return ASRResult(success=False, error_type="RuntimeError", error_message="model is not loaded")
        started = time.perf_counter()
        try:
            language = kwargs.get("language", "Chinese")
            language_arg = None if str(language).lower() == "auto" else [language]
            results = self.model.transcribe(audio=[str(audio_path)], language=language_arg)
            first = results[0] if results else None
            text = ""
            out_language = None
            if isinstance(first, dict):
                text = str(first.get("text") or "")
                out_language = first.get("language")
            elif first is not None:
                text = str(getattr(first, "text", "") or "")
                out_language = getattr(first, "language", None)
            return ASRResult(success=bool(text.strip()), text=text, language=out_language, infer_time_sec=time.perf_counter() - started)
        except Exception as exc:
            return exception_result(exc, infer_time_sec=time.perf_counter() - started)

    def unload(self) -> None:
        self.model = None
