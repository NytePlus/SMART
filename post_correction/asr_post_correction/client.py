from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from openai import OpenAI


class CachedJsonClient:
    def __init__(self, *, model: str, base_url: str, cache_dir: str | Path, max_attempts: int = 3):
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("set LLM_API_KEY or OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max_attempts
        self._lock = threading.Lock()

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re_sub_fence(stripped)
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            start, end = stripped.find("{"), stripped.rfind("}")
            if start < 0 or end <= start:
                raise
            obj = json.loads(stripped[start : end + 1])
        if not isinstance(obj, dict):
            raise ValueError("LLM response must be a JSON object")
        return obj

    def complete_json(self, messages: list[dict[str, str]], *, purpose: str) -> tuple[dict[str, Any], dict[str, Any]]:
        key_payload = {"model": self.model, "messages": messages, "purpose": purpose}
        key = hashlib.sha256(json.dumps(key_payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return cached["parsed"], {**cached.get("meta", {}), "cached": True}
        errors: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            repair = [] if not errors else [{"role": "user", "content": "上次输出不是有效的目标 JSON。请只返回一个符合结构的 JSON 对象。"}]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages + repair,
                temperature=0,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content or ""
            try:
                parsed = self._extract_json(text)
            except Exception as exc:
                errors.append(f"attempt {attempt}: {exc}")
                continue
            usage = response.usage
            meta = {
                "cached": False,
                "attempt": attempt,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            with self._lock:
                cache_path.write_text(json.dumps({"parsed": parsed, "meta": meta}, ensure_ascii=False), encoding="utf-8")
            return parsed, meta
        raise RuntimeError("invalid JSON after retries: " + "; ".join(errors))


def re_sub_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
