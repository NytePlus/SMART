from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .base import ASRResult, Issue, exception_result, generic_model_dir_issues


class WhisperAdapter:
    name = "whisper"

    def __init__(self) -> None:
        self.pipeline = None

    def validate_model_dir(self, model_dir: Path) -> list[Issue]:
        issues = generic_model_dir_issues(model_dir)
        if issues:
            return issues
        if not (model_dir / "config.json").exists():
            issues.append(Issue("error", f"missing config.json in {model_dir}"))
        if not any(model_dir.glob("*.safetensors")) and not any(model_dir.glob("*.bin")):
            issues.append(Issue("error", f"no model weight file (*.safetensors or *.bin) found in {model_dir}"))
        return issues

    def load(self, model_dir: Path, device: str, dtype: str | None = None, **kwargs: Any) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError as exc:
            raise ImportError("Cannot import transformers/torch. Install whisper dependencies.") from exc

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            None: torch.float16 if device.startswith("cuda") else torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.float16 if device.startswith("cuda") else torch.float32)
        processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(model_dir),
            torch_dtype=torch_dtype,
            local_files_only=True,
            trust_remote_code=bool(kwargs.get("trust_remote_code", False)),
        )
        if device != "cpu":
            model.to(device)
        device_arg = -1 if device == "cpu" else int(device.split(":", 1)[1]) if ":" in device else 0
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device_arg,
        )

    def transcribe(self, audio_path: Path, **kwargs: Any) -> ASRResult:
        if self.pipeline is None:
            return ASRResult(success=False, error_type="RuntimeError", error_message="model is not loaded")
        started = time.perf_counter()
        try:
            generate_kwargs: dict[str, Any] = {}
            language = kwargs.get("language")
            task = kwargs.get("task", "transcribe")
            if language and str(language).lower() != "auto":
                generate_kwargs["language"] = language
            if task:
                generate_kwargs["task"] = task
            result = self.pipeline(str(audio_path), generate_kwargs=generate_kwargs or None)
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            return ASRResult(success=bool(text.strip()), text=str(text), infer_time_sec=time.perf_counter() - started)
        except Exception as exc:
            return exception_result(exc, infer_time_sec=time.perf_counter() - started)

    def unload(self) -> None:
        self.pipeline = None
