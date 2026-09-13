"""Resumable OpenAI-compatible inference for rubric judgments."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from .aggregation import aggregate_rubric_result
from .io_utils import append_jsonl, read_jsonl, write_json
from .prompting import SYSTEM_PROMPT, build_messages
from .response_parser import parse_json_response
from .schema import validate_dataset, validate_rubric_judgment


def load_env_file(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.removeprefix("export ").strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def resolve_api_settings(model: str | None, base_url: str | None, api_key: str | None) -> tuple[str, str | None, str]:
    resolved_model = model or os.environ.get("LLM_MODEL")
    resolved_base_url = base_url or os.environ.get("LLM_BASE_URL")
    resolved_api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("LLM_API_KEY_MASKED")
    if not resolved_model:
        raise RuntimeError("model is required through --model or LLM_MODEL")
    if not resolved_api_key:
        raise RuntimeError("API key is required through LLM_API_KEY or LLM_API_KEY_MASKED")
    return resolved_model, resolved_base_url, resolved_api_key


def _usage_dict(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _call_model(client: OpenAI, messages: list[dict], model: str, temperature: float, seed: int | None, max_tokens: int, timeout: int):
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        # APIFusion's DeepSeek gateway recognizes this OpenAI-compatible
        # switch; chat_template_kwargs is ignored by that gateway.
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    if seed is not None:
        kwargs["seed"] = seed
    return client.chat.completions.create(**kwargs)


def _repair_message(validation_errors: list[str], sample: dict, config: dict) -> str:
    reference_span = json.dumps(str(sample.get("reference_segment", "")), ensure_ascii=False)
    hypothesis_span = json.dumps(str(sample.get("hypothesis_segment", "")), ensure_ascii=False)
    if config.get("prompt_language") == "zh":
        return (
            "上一次输出未通过严格校验，错误如下：\n- "
            + "\n- ".join(validation_errors)
            + "\n请只返回修正后的完整 JSON。证据必须逐字复制，包括空格和标点。"
            + "删除无效的非必需证据；必须提供证据时，优先使用以下精确短片段：\n"
            + f"reference_span={reference_span}\n"
            + f"hypothesis_span={hypothesis_span}\n"
            + "不得改动证据中的任何一个字符。"
        )
    return (
        "The response failed strict validation for these reasons:\n- "
        + "\n- ".join(validation_errors)
        + "\nReturn a corrected complete JSON object only. Evidence quotes must be exact substrings, including spaces and punctuation. Delete invalid optional evidence entries. For required evidence, prefer one of these exact short spans when relevant:\n"
        + f"reference_span={reference_span}\n"
        + f"hypothesis_span={hypothesis_span}\n"
        + "Do not alter even one character of a quoted span."
    )


def classify_one(
    sample: dict,
    config: dict,
    client: OpenAI,
    model: str,
    temperature: float,
    seed: int | None,
    max_tokens: int,
    timeout: int,
    max_attempts: int,
) -> dict:
    base_messages = build_messages(sample, config)
    messages = list(base_messages)
    attempts: list[dict] = []
    started = time.monotonic()

    for attempt_number in range(1, max_attempts + 1):
        try:
            attempt_temperature = temperature if attempt_number == 1 else max(temperature, 0.2)
            response = _call_model(client, messages, model, attempt_temperature, seed, max_tokens, timeout)
            raw = response.choices[0].message.content or ""
            usage = _usage_dict(response)
            try:
                judgment = parse_json_response(raw)
                validation_errors = validate_rubric_judgment(judgment, sample, config)
            except Exception as exc:
                judgment = None
                validation_errors = [f"PARSE_ERROR: {exc}"]
            attempts.append({
                "attempt": attempt_number,
                "raw_response": raw,
                "validation_errors": validation_errors,
                "usage": usage,
                "temperature": attempt_temperature,
            })
            if not validation_errors and judgment is not None:
                return {
                    **sample,
                    "inference_status": "SUCCESS",
                    "rubric_version": config["rubric_version"],
                    "prompt_version": config["prompt_version"],
                    "judge_model": model,
                    "rubric_judgment": judgment,
                    "aggregated_result": aggregate_rubric_result(judgment, config),
                    "raw_response": raw,
                    "attempts": attempts,
                    "usage": usage,
                    "latency_seconds": round(time.monotonic() - started, 3),
                }
            messages = [
                *base_messages,
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": _repair_message(validation_errors, sample, config),
                },
            ]
        except Exception as exc:
            attempts.append({
                "attempt": attempt_number,
                "raw_response": None,
                "validation_errors": [f"API_ERROR: {type(exc).__name__}: {exc}"],
                "usage": {},
                "temperature": attempt_temperature,
            })
        if attempt_number < max_attempts:
            time.sleep(min(2 ** (attempt_number - 1), 4))

    return {
        **sample,
        "inference_status": "FAILED",
        "rubric_version": config["rubric_version"],
        "prompt_version": config["prompt_version"],
        "judge_model": model,
        "rubric_judgment": None,
        "aggregated_result": None,
        "raw_response": attempts[-1].get("raw_response") if attempts else None,
        "attempts": attempts,
        "failure_type": "MAX_ATTEMPTS_EXCEEDED",
        "latency_seconds": round(time.monotonic() - started, 3),
    }


def _existing_statuses(output_path: Path) -> dict[str, str]:
    if not output_path.exists():
        return {}
    return {
        str(record.get("sample_id")): str(record.get("inference_status"))
        for record in read_jsonl(output_path)
    }


def _code_version() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _code_hash() -> str:
    experiment_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    included_suffixes = {".py", ".json", ".md"}
    for path in sorted(experiment_root.rglob("*")):
        if not path.is_file() or path.suffix not in included_suffixes:
            continue
        if any(part in {"data", "outputs", "__pycache__"} for part in path.parts):
            continue
        digest.update(str(path.relative_to(experiment_root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _prompt_hash(config: dict) -> str:
    payload = config.get("system_prompt", SYSTEM_PROMPT) + json.dumps(
        config, ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_inference(
    records: list[dict],
    output_path: Path,
    config: dict,
    model: str,
    base_url: str | None,
    api_key: str,
    workers: int = 1,
    temperature: float = 0.0,
    seed: int | None = None,
    max_tokens: int = 3200,
    timeout: int = 90,
    max_attempts: int = 2,
    retry_failures: bool = False,
) -> dict:
    issues = validate_dataset(records)
    if issues:
        raise ValueError(f"input validation failed for {len(issues)} records: {issues[:3]}")

    statuses = _existing_statuses(output_path)
    pending = []
    for record in records:
        status = statuses.get(str(record["sample_id"]))
        if status is None or (status == "FAILED" and retry_failures):
            pending.append(record)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_records": len(records),
        "pending_records": len(pending),
        "existing_records": len(statuses),
        "rubric_version": config["rubric_version"],
        "prompt_version": config["prompt_version"],
        "rubric_config_hash": config["config_hash"],
        "prompt_hash": _prompt_hash(config),
        "code_version": _code_version(),
        "code_hash": _code_hash(),
        "judge_model": model,
        "base_url_configured": bool(base_url),
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "max_attempts": max_attempts,
        "workers": workers,
    }
    write_json(output_path.with_suffix(".manifest.json"), manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=base_url, api_key=api_key)
    successes = failures = 0
    with output_path.open("a", encoding="utf-8") as output_handle:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    classify_one,
                    record,
                    config,
                    client,
                    model,
                    temperature,
                    seed,
                    max_tokens,
                    timeout,
                    max_attempts,
                ): record
                for record in pending
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                append_jsonl(output_handle, result)
                if result["inference_status"] == "SUCCESS":
                    successes += 1
                else:
                    failures += 1
                print(
                    f"Processed {completed}/{len(pending)} "
                    f"(success={successes}, failed={failures})",
                    flush=True,
                )
    manifest.update({"new_successes": successes, "new_failures": failures})
    write_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest
