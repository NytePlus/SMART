#!/usr/bin/env python3
"""Run an isolated two-turn rubric dialogue for selected samples.

Turn one is the unchanged production rubric request. Turn two is a debug-only
follow-up that asks the same model to explain its item-level decisions and
identify plausible alternative judgments. The script never changes production
aggregation or inference outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rubric_eval.aggregation import aggregate_rubric_result
from rubric_eval.config import load_rubric
from rubric_eval.io_utils import read_jsonl, write_json
from rubric_eval.llm_runner import load_env_file, resolve_api_settings
from rubric_eval.prompting import build_messages
from rubric_eval.response_parser import parse_json_response
from rubric_eval.schema import validate_rubric_judgment


DEFAULT_IDS = (
    "canvas:0000841:1",
    "tedlium3:0000103:0",
    "tedlium3:0000098:1",
)


def _call(client: OpenAI, model: str, messages: list[dict], temperature: float, timeout: int, max_tokens: int):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_body={"thinking": {"type": "disabled"}},
    )


def _content(response) -> str:
    return str(getattr(response.choices[0].message, "content", "") or "")


def run_debug_dialogue(
    sample: dict,
    config: dict,
    client: OpenAI,
    model: str,
    temperature: float = 0.0,
    timeout: int = 120,
    max_tokens: int = 3200,
) -> dict:
    """Execute two turns without changing the production inference path."""
    messages = build_messages(sample, config)
    first_raw = ""
    second_raw = ""
    errors: list[str] = []
    judgment = None
    aggregate = None

    try:
        first_raw = _content(_call(client, model, messages, temperature, timeout, max_tokens))
        try:
            judgment = parse_json_response(first_raw)
            validation_errors = validate_rubric_judgment(judgment, sample, config)
            if validation_errors:
                errors.extend(f"turn1: {error}" for error in validation_errors)
            else:
                aggregate = aggregate_rubric_result(judgment, config)
        except Exception as exc:
            errors.append(f"turn1 parse/validation: {type(exc).__name__}: {exc}")

        followup = (
            "This is a debugging follow-up to your immediately preceding rubric JSON. "
            "Do not silently change the first-turn JSON. Explain why you selected each "
            "of S1, S2, S3, S4, C1, and C2, citing the exact input phrase that drove "
            "the decision. Then state whether a reasonable annotator could instead "
            "treat the changed span as a changed entity, technical term, proposition, "
            "or participant relation, and why. Explicitly discuss the distinction "
            "between a recoverable spelling variant and a wrong proper name or key term. "
            "Return JSON only with this shape: {\"decision_reasons\": {\"S1\": \"...\", "
            "\"S2\": \"...\", \"S3\": \"...\", \"S4\": \"...\", \"C1\": \"...\", "
            "\"C2\": \"...\"}, \"alternative_judgment_possible\": true, "
            "\"alternative_judgment_explanation\": \"...\", "
            "\"would_revise_first_turn\": true|false, "
            "\"revision_if_any\": {}}"
        )
        second_messages = [
            *messages,
            {"role": "assistant", "content": first_raw},
            {"role": "user", "content": followup},
        ]
        second_raw = _content(_call(client, model, second_messages, temperature, timeout, max_tokens))
    except Exception as exc:
        errors.append(f"API: {type(exc).__name__}: {exc}")

    return {
        "sample_id": sample.get("sample_id"),
        "dataset": sample.get("dataset"),
        "reference_segment": sample.get("reference_segment"),
        "hypothesis_segment": sample.get("hypothesis_segment"),
        "reference_text": sample.get("target"),
        "hypothesis_text": sample.get("pred"),
        "model": model,
        "temperature": temperature,
        "turn1_raw": first_raw,
        "turn1_judgment": judgment,
        "turn1_validation_and_api_errors": errors,
        "turn1_aggregated_result": aggregate,
        "turn2_raw_explanation": second_raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "llm_classification_input_fixed300_without_aishell4.jsonl")
    parser.add_argument("--rubric", type=Path, default=ROOT / "config" / "rubric_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "debug_three_deepseek_flash" / "dialogues.json")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=3200)
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    args = parser.parse_args()

    load_env_file(args.env_file)
    model, base_url, api_key = resolve_api_settings(args.model, args.base_url, args.api_key)
    config = load_rubric(args.rubric)
    wanted = set(args.sample_ids or DEFAULT_IDS)
    samples = [record for record in read_jsonl(args.input) if str(record.get("sample_id")) in wanted]
    missing = wanted - {str(record.get("sample_id")) for record in samples}
    if missing:
        raise SystemExit(f"sample IDs not found: {sorted(missing)}")

    client = OpenAI(base_url=base_url, api_key=api_key)
    results = [
        run_debug_dialogue(
            sample,
            config,
            client,
            model,
            temperature=args.temperature,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
        )
        for sample in samples
    ]
    write_json(
        args.output,
        {
            "model": model,
            "base_url": base_url,
            "rubric_version": config["rubric_version"],
            "sample_ids": [result["sample_id"] for result in results],
            "results": results,
        },
    )
    print(f"Wrote {len(results)} debug dialogues to {args.output}")
    for result in results:
        status = "ok" if not result["turn1_validation_and_api_errors"] else "error"
        print(f"{result['sample_id']}: {status}")


if __name__ == "__main__":
    main()
