from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .io import source_id


def resolve_audio_path(
    row: dict[str, Any],
    message_by_id: dict[str, dict[str, Any]],
    audio_root: str | Path,
) -> Path:
    local_id = source_id(row)
    message = message_by_id.get(local_id, {})
    relative = str(message.get("audio") or message.get("path") or f"audios/{local_id}.wav")
    path = Path(relative)
    return path if path.is_absolute() else Path(audio_root) / path


class LocalWhisperTranscriber:
    """Offline Whisper wrapper used only to create non-GT ASR evidence."""

    def __init__(self, model_dir: str | Path, *, device: str = "cpu", dtype: str | None = None):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        model_dtype = dtype_map.get(dtype or ("float32" if device == "cpu" else "float16"))
        model_dir = str(Path(model_dir).resolve())
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_dir, dtype=model_dtype, local_files_only=True
        )
        if device != "cpu":
            model.to(device)
        device_arg = -1 if device == "cpu" else int(device.split(":", 1)[1]) if ":" in device else 0
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=model_dtype,
            device=device_arg,
        )

    def transcribe(self, audio_path: str | Path, *, language: str | None = None) -> tuple[str, float]:
        generate_kwargs: dict[str, Any] = {"task": "transcribe"}
        if language and language.lower() != "auto":
            generate_kwargs["language"] = language
        started = time.perf_counter()
        result = self.pipeline(str(audio_path), generate_kwargs=generate_kwargs)
        elapsed = time.perf_counter() - started
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return str(text).strip(), elapsed


class LocalQwen2AudioTranscriber:
    """Offline Qwen2-Audio transcription for acoustic correction evidence."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
    ):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        model_dir = str(Path(model_dir).resolve())
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_dir, dtype=dtype_map[dtype], local_files_only=True
        )
        self.model.eval()
        self.torch = torch
        self.max_new_tokens = max_new_tokens

    def transcribe(self, audio_path: str | Path, *, language: str = "Chinese") -> tuple[str, float]:
        audio, sampling_rate = self._read_audio(audio_path)
        prompt = self._prompt(language)
        inputs = self.processor(
            text=prompt,
            audio=audio,
            audio_kwargs={"sampling_rate": sampling_rate},
            return_tensors="pt",
        )
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        generated = generated[:, inputs.input_ids.size(1):]
        text = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return str(text).strip(), time.perf_counter() - started

    def _read_audio(self, audio_path: str | Path):
        import soundfile as sf

        audio, sampling_rate = sf.read(str(audio_path), dtype="float32")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        expected_rate = int(self.processor.feature_extractor.sampling_rate)
        if sampling_rate != expected_rate:
            from scipy.signal import resample_poly
            import math

            divisor = math.gcd(sampling_rate, expected_rate)
            audio = resample_poly(audio, expected_rate // divisor, sampling_rate // divisor)
            sampling_rate = expected_rate
        return audio, sampling_rate

    @staticmethod
    def _prompt(language: str) -> str:
        language_name = "Chinese" if language.lower() in {"zh", "chinese"} else "English" if language.lower() in {"en", "english"} else language
        return (
            "<|audio_bos|><|AUDIO|><|audio_eos|>"
            f"Transcribe the {language_name} speech exactly:"
        )

    def score_transcripts(
        self,
        audio_path: str | Path,
        transcripts: list[str],
        *,
        language: str = "Chinese",
    ) -> tuple[list[dict[str, float | int]], float]:
        """Return length-normalized conditional log likelihood for each transcript."""
        audio, sampling_rate = self._read_audio(audio_path)
        prompt = self._prompt(language)
        texts = [prompt + transcript for transcript in transcripts]
        inputs = self.processor(
            text=texts,
            audio=[audio] * len(texts),
            audio_kwargs={"sampling_rate": sampling_rate},
            padding=True,
            return_tensors="pt",
        )
        prefix_inputs = self.processor(
            text=prompt,
            audio=audio,
            audio_kwargs={"sampling_rate": sampling_rate},
            return_tensors="pt",
        )
        prefix_length = int(prefix_inputs.input_ids.shape[1])
        started = time.perf_counter()
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits.float()
        scores: list[dict[str, float | int]] = []
        for index in range(len(transcripts)):
            length = int(inputs.attention_mask[index].sum())
            token_ids = inputs.input_ids[index, prefix_length:length]
            token_logits = logits[index, prefix_length - 1:length - 1]
            log_probs = self.torch.log_softmax(token_logits, dim=-1)
            values = log_probs.gather(1, token_ids[:, None]).squeeze(1)
            token_count = int(token_ids.numel())
            total = float(values.sum())
            scores.append({
                "mean_log_probability": total / token_count if token_count else float("-inf"),
                "total_log_probability": total,
                "token_count": token_count,
            })
        return scores, time.perf_counter() - started
