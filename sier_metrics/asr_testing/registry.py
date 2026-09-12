from __future__ import annotations

from .adapters.base import BaseASRAdapter


ADAPTER_IMPORTS = {
    "qwen3_asr": ("asr_testing.adapters.qwen3_asr", "Qwen3ASRAdapter"),
    "whisper": ("asr_testing.adapters.whisper", "WhisperAdapter"),
    "paraformer": ("asr_testing.adapters.paraformer", "ParaformerAdapter"),
    "sensevoice": ("asr_testing.adapters.sensevoice", "SenseVoiceAdapter"),
}


def create_adapter(adapter_name: str) -> BaseASRAdapter:
    if adapter_name not in ADAPTER_IMPORTS:
        available = ", ".join(sorted(ADAPTER_IMPORTS))
        raise KeyError(f"Unknown adapter '{adapter_name}'. Available adapters: {available}")
    module_name, class_name = ADAPTER_IMPORTS[adapter_name]
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls()
