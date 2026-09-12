from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .client import CachedJsonClient
from .audio_evidence import (LocalQwen2AudioTranscriber, LocalWhisperTranscriber,
                             resolve_audio_path)
from .alternate_selector import propose_consensus_edits, propose_local_edits
from .baseline import (apply_baseline_edits, load_headline_rows,
                       render_baseline_messages, render_split_baseline_messages,
                       restrict_edits_to_candidates, validate_headline_alignment,
                       validate_rubric_candidates)
from .core import CorrectionEngine
from .io import (add_auxiliary_context, append_jsonl, build_contexts,
                 load_completed_ids, prediction_text, read_jsonl, source_id,
                 systematic_indices, target_text)
from .evaluation import evaluate_files, evaluate_headline, evaluate_headline_cseer
from .prompts import load_prompt_bundle


def code_sha256() -> str:
    digest = hashlib.sha256()
    package_dir = Path(__file__).resolve().parent
    for path in sorted(package_dir.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def command_run(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    messages = read_jsonl(args.message_file) if args.message_file else []
    context_rows = read_jsonl(args.context_input) if args.context_input else rows
    contexts = build_contexts(context_rows, messages, args.history_size)
    alternate_inputs = args.alternate_input or []
    alternate_rows = []
    for path in alternate_inputs:
        source_label = Path(path).parent.name or Path(path).stem
        for row in read_jsonl(path):
            # Source labels are metadata for evidence presentation only; the
            # prediction payload itself is left untouched.
            enriched = dict(row)
            enriched.setdefault("_evidence_source", source_label)
            alternate_rows.append(enriched)
    ocr_rows = read_jsonl(args.ocr_file) if args.ocr_file else []
    add_auxiliary_context(
        contexts, context_rows, alternate_rows=alternate_rows, ocr_rows=ocr_rows
    )
    prompts = load_prompt_bundle(args.prompt)
    postprocessors = list(prompts.get("deterministic_postprocessors", []))
    if args.llm_only and postprocessors:
        raise ValueError(
            f"--llm-only forbids deterministic_postprocessors: {postprocessors}"
        )
    client = CachedJsonClient(model=args.model, base_url=args.base_url, cache_dir=args.cache_dir, max_attempts=args.max_attempts)
    engine = CorrectionEngine(client, prompts, detector_threshold=args.detector_threshold,
                              corrector_threshold=args.corrector_threshold,
                              verifier_threshold=args.verifier_threshold,
                              max_segment_chars=args.max_segment_chars,
                              forbid_normalization_edits=args.llm_only)
    completed = load_completed_ids(args.output)
    pending = [row for row in rows if str(row.get("id")) not in completed]
    output_lock = threading.Lock()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "input": str(Path(args.input).resolve()),
        "context_input": str(Path(args.context_input).resolve()) if args.context_input else None,
        "alternate_inputs": [str(Path(path).resolve()) for path in alternate_inputs],
        "alternate_evidence_view": "source_labeled_token_disagreements_v1",
        "ocr_file": str(Path(args.ocr_file).resolve()) if args.ocr_file else None,
        "message_file": str(Path(args.message_file).resolve()) if args.message_file else None,
        "mode": args.mode, "max_rounds": args.max_rounds, "history_size": args.history_size,
        "model": args.model, "base_url": args.base_url, "prompt_version": prompts["version"],
        "prompt_sha256": prompts["sha256"], "detector_threshold": args.detector_threshold,
        "detector_passes": ["primary"] + [str(item["name"]) for item in prompts.get("detector_passes", [])],
        "corrector_passes": ["primary"] + [str(item["name"]) for item in prompts.get("corrector_passes", [])],
        "deterministic_postprocessors": postprocessors,
        "llm_only": bool(args.llm_only),
        "normalization_only_edits_forbidden": bool(args.llm_only),
        "code_sha256": code_sha256(),
        "corrector_threshold": args.corrector_threshold, "max_segment_chars": args.max_segment_chars,
        "verifier_threshold": args.verifier_threshold,
        "test_target_visible_to_engine": False,
    }
    manifest_path = Path(args.output).with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def process(row: dict[str, Any]) -> dict[str, Any]:
        row_id = str(row.get("id"))
        hypothesis = prediction_text(row)
        if not hypothesis:
            raise ValueError("empty hypothesis")
        context = contexts.get(row_id, {})
        result = engine.single_round(hypothesis, context) if args.mode == "single" else engine.multi_round(hypothesis, context, args.max_rounds)
        return {"id": row_id, "status": "ok", "target": target_text(row), **result.to_dict()}

    def save(row: dict[str, Any]) -> None:
        with output_lock:
            append_jsonl(args.output, row)

    if args.workers == 1:
        for index, row in enumerate(pending, 1):
            try:
                save(process(row))
            except Exception as exc:
                save({"id": str(row.get("id")), "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(pending)}] {row.get('id')}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, row): row for row in pending}
            for index, future in enumerate(as_completed(futures), 1):
                row = futures[future]
                try:
                    save(future.result())
                except Exception as exc:
                    save({"id": str(row.get("id")), "status": "error", "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{index}/{len(pending)}] {row.get('id')}", flush=True)
    return 0


def command_gt_context_baseline(args: argparse.Namespace) -> int:
    rows = load_headline_rows(args.data_prepare_root, args.model_key)
    if args.sample_count:
        rows = [rows[index] for index in systematic_indices(len(rows), args.sample_count)]
    if args.limit:
        rows = rows[: args.limit]
    parent_results: dict[str, dict[str, Any]] = {}
    if args.parent_corrected:
        parent_results = {
            str(row["id"]): row
            for row in read_jsonl(args.parent_corrected)
            if row.get("status") == "ok"
        }
        missing_parent = [row["id"] for row in rows if row["id"] not in parent_results]
        if missing_parent:
            raise ValueError(
                f"parent corrected output is missing {len(missing_parent)} selected rows"
            )
    input_wer_preflight = validate_headline_alignment(
        args.data_prepare_root, args.model_key, rows
    )
    prompt_path = Path(args.prompt)
    prompt_bytes = prompt_path.read_bytes()
    prompt = load_prompt_bundle(prompt_path)
    split_prompt = all(
        key in prompt
        for key in ("detector_system", "detector_user", "corrector_system", "corrector_user")
    )
    detector_requires_rubric = bool(prompt.get("detector_requires_rubric", True))
    client = CachedJsonClient(
        model=args.model,
        base_url=args.base_url,
        cache_dir=args.cache_dir,
        max_attempts=args.max_attempts,
    )
    completed = load_completed_ids(args.output)
    pending = [row for row in rows if row["id"] not in completed]
    output_lock = threading.Lock()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "benchmark composites excluding AISHELL-4, without source dedupe",
        "data_prepare_root": str(Path(args.data_prepare_root).resolve()),
        "model_key": args.model_key,
        "input_rows": len(rows),
        "sample_count": args.sample_count,
        "sampling_policy": "content_blind_systematic_by_composite_id" if args.sample_count else None,
        "input_wer_preflight": input_wer_preflight,
        "model": args.model,
        "base_url": args.base_url,
        "prompt": str(prompt_path.resolve()),
        "prompt_version": prompt["version"],
        "prompt_sha256": prompt["sha256"],
        "code_sha256": code_sha256(),
        "pipeline": (
            "no_reference_rubric_detection_then_contextual_edit"
            if split_prompt and detector_requires_rubric
            else "no_reference_surface_detection_then_contextual_edit"
            if split_prompt
            else "single_llm_edit_pass"
        ),
        "rubric_enabled": split_prompt and detector_requires_rubric,
        "detector_enabled": split_prompt,
        "verifier_enabled": False,
        "correction_assets_visible": False,
        "hotwords_visible": False,
        "alternate_visible": False,
        "target_sentence_visible_to_engine": False,
        "ground_truth_context_visible": True,
        "ground_truth_context_source": "benchmark semantic_xyz.history",
        "deterministic_postprocessors": [],
        "llm_only": True,
        "normalization_only_edits_forbidden": True,
        "parent_corrected": str(Path(args.parent_corrected).resolve()) if args.parent_corrected else None,
        "parent_corrected_sha256": hashlib.sha256(Path(args.parent_corrected).read_bytes()).hexdigest() if args.parent_corrected else None,
    }
    manifest_path = Path(args.output).with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    def process(row: dict[str, Any]) -> dict[str, Any]:
        parent = parent_results.get(str(row["id"]))
        hypothesis = str(parent["corrected_hypothesis"]) if parent else str(row["pred"])
        history = str(row["history"])
        candidates: list[dict[str, Any]] = []
        rejected_candidates: list[dict[str, Any]] = []
        detector_meta: dict[str, Any] = {"skipped": "single_pass_prompt"}
        raw_candidates: Any = []
        if split_prompt:
            detector_raw, detector_meta = client.complete_json(
                render_split_baseline_messages(prompt, "detector", history, hypothesis),
                purpose=f"gt-context-detector:{prompt['version']}:{args.model_key}",
            )
            if not isinstance(detector_raw, dict):
                detector_raw = {}
            raw_candidates = detector_raw.get("candidates", [])
            candidates, rejected_candidates = validate_rubric_candidates(
                hypothesis,
                raw_candidates,
                require_rubric=bool(prompt.get("detector_requires_rubric", True)),
            )
            if candidates:
                raw, corrector_meta = client.complete_json(
                    render_split_baseline_messages(
                        prompt, "corrector", history, hypothesis, candidates
                    ),
                    purpose=f"gt-context-corrector:{prompt['version']}:{args.model_key}",
                )
            else:
                raw, corrector_meta = {"edits": []}, {"skipped": "no_candidates"}
        else:
            raw, corrector_meta = client.complete_json(
                render_baseline_messages(prompt, history, hypothesis),
                purpose=f"gt-context-baseline:{prompt['version']}:{args.model_key}",
            )
        if not isinstance(raw, dict):
            raw = {}
        raw_edits = raw.get("edits", [])
        unauthorized_edits: list[dict[str, Any]] = []
        if split_prompt:
            raw_edits, unauthorized_edits = restrict_edits_to_candidates(
                raw_edits, candidates
            )
        corrected, accepted, rejected = apply_baseline_edits(
            hypothesis, raw_edits, max_edits=400
        )
        rejected = unauthorized_edits + rejected
        prior_rounds = list(parent.get("rounds", [])) if parent else []
        original_hypothesis = str(parent.get("original_hypothesis", row["pred"])) if parent else str(row["pred"])
        return {
            "id": row["id"],
            "source_dataset": row["source_dataset"],
            "source_segment_id": row["source_segment_id"],
            "status": "ok",
            "target": row["target"],
            "original_hypothesis": original_hypothesis,
            "corrected_hypothesis": corrected,
            "rounds": prior_rounds + [{
                "round_index": len(prior_rounds) + 1,
                "input_hypothesis": hypothesis,
                "raw_candidates": raw_candidates,
                "rubric_candidates": candidates,
                "rejected_candidates": rejected_candidates,
                "raw_edits": raw.get("edits", []),
                "applied_edits": accepted,
                "rejected_edits": rejected,
                "output_hypothesis": corrected,
                "detector_api": detector_meta,
                "corrector_api": corrector_meta,
            }],
            "stop_reason": "edits_applied" if accepted else "no_changes",
            "seen_hypotheses": (list(parent.get("seen_hypotheses", [])) if parent else [hypothesis]) + ([corrected] if corrected != hypothesis else []),
            "deterministic_edits": [],
        }

    def save(result: dict[str, Any]) -> None:
        with output_lock:
            append_jsonl(args.output, result)

    if args.workers == 1:
        for index, row in enumerate(pending, 1):
            try:
                save(process(row))
            except Exception as exc:
                save({"id": row["id"], "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(pending)}] {row['id']}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, row): row for row in pending}
            for index, future in enumerate(as_completed(futures), 1):
                row = futures[future]
                try:
                    save(future.result())
                except Exception as exc:
                    save({"id": row["id"], "status": "error", "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{index}/{len(pending)}] {row['id']}", flush=True)
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    summary = evaluate_files(
        args.data_prepare_root,
        args.before,
        args.corrected,
        args.output_dir,
        require_complete=args.require_complete,
        require_llm_only=args.require_llm_only,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_headline_evaluate(args: argparse.Namespace) -> int:
    summary = evaluate_headline(
        args.data_prepare_root,
        args.model_key,
        args.output_dir,
        corrected_path=args.corrected,
        require_llm_only=args.require_llm_only,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_prepare_headline_cseer(args: argparse.Namespace) -> int:
    rows = load_headline_rows(args.data_prepare_root, args.model_key)
    if args.sample_count:
        rows = [rows[index] for index in systematic_indices(len(rows), args.sample_count)]
    results = {
        str(row["id"]): row
        for row in read_jsonl(args.corrected)
        if row.get("status") == "ok"
    }
    missing = [row["id"] for row in rows if row["id"] not in results]
    if missing:
        raise ValueError(f"corrected output is missing {len(missing)} headline rows")
    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_path = model_dir / "pred.raw.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            result = results[row["id"]]
            handle.write(json.dumps({
                "id": row["id"],
                "target": row["target"],
                "pred": prediction_text({"pred": result["corrected_hypothesis"]}),
                "dataset": row["id"].split(":", 1)[0],
                "source_id": row["source_segment_id"],
                "model": args.model_key,
            }, ensure_ascii=False) + "\n")
    data_root = Path(args.data_prepare_root).resolve()
    mapping_path = model_dir / "segment_mapping.json"
    subprocess.run([
        sys.executable,
        str(data_root / "scripts" / "build_segment_mapping.py"),
        "--input", str(pred_path),
        "--output", str(mapping_path),
        "--dataset", "benchmark",
        "--normalize-for-mapping",
        "--normalizer-engine", "rule",
        "--normalizer-fallback", "rule",
        "--normalizer-spacing", "smart",
    ], check=True)
    gt_path = model_dir / "gt.jsonl"
    with gt_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({
                "id": row["id"], "ground_truth": row["target"]
            }, ensure_ascii=False) + "\n")
    subprocess.run([
        sys.executable,
        str(data_root / "scripts" / "build_llm_classification_input.py"),
        "--input", str(mapping_path),
        "--output", str(model_dir / "llm_classification_input.jsonl"),
        "--history", "0", "--suffix", "0", "--gt-file", str(gt_path),
    ], check=True)
    new_inputs = read_jsonl(model_dir / "llm_classification_input.jsonl")
    fingerprints = {
        (
            str(row.get("id", "")), str(row.get("focus_id", "")),
            str(row.get("target", "")), str(row.get("pred", "")),
            str(row.get("reference_segment", "")), str(row.get("hypothesis_segment", "")),
            str(row.get("op_segment", "")),
        )
        for row in new_inputs
    }
    historical_path = (
        data_root / "segments-mapping" / "benchmark" / args.model_key
        / "results" / "classification_results.full.jsonl"
    )
    reusable_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if historical_path.exists():
        new_by_key = {
            (str(row.get("id", "")), str(row.get("focus_id", ""))): row
            for row in new_inputs
        }
        for row in read_jsonl(historical_path):
            key = (str(row.get("id", "")), str(row.get("focus_id", "")))
            fingerprint = (
                str(row.get("id", "")), str(row.get("focus_id", "")),
                str(row.get("target", "")), str(row.get("pred", "")),
                str(row.get("reference_segment", "")), str(row.get("hypothesis_segment", "")),
                str(row.get("op_segment", "")),
            )
            if fingerprint in fingerprints:
                reusable_by_key[key] = row
                continue
            result = results.get(key[0])
            if (
                key in new_by_key and result
                and str(result.get("original_hypothesis", ""))
                == str(result.get("corrected_hypothesis", ""))
            ):
                reusable_by_key[key] = {**row, **new_by_key[key]}
    reusable = list(reusable_by_key.values())
    results_dir = model_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "classification_results.rubric.jsonl").open("w", encoding="utf-8") as handle:
        for row in reusable:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "model_dir": str(model_dir),
        "rows": len(rows),
        "classification_inputs": len(new_inputs),
        "reused_historical_classifications": len(reusable),
        "pending_classifications": len(new_inputs) - len(reusable),
        "next_command": (
            f"{sys.executable} {data_root / 'scripts' / 'run_llm_classification.py'} "
            f"--input {model_dir / 'llm_classification_input.jsonl'} --classifier rubric"
        ),
    }, ensure_ascii=False, indent=2))
    return 0


def command_finalize_headline_cseer(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir).resolve()
    rubric_results = model_dir / "results" / "classification_results.rubric.jsonl"
    canonical_results = model_dir / "results" / "classification_results.jsonl"
    if not rubric_results.exists():
        raise ValueError(f"missing rubric classification output: {rubric_results}")
    shutil.copyfile(rubric_results, canonical_results)
    result = evaluate_headline_cseer(args.data_prepare_root, model_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_export(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.corrected)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row.get("status") != "ok":
                continue
            # c-SEER's mapping/context utilities use dataset-local IDs.  Keep
            # converted model outputs such as ``canvas-speech:0000001``
            # addressable as ``0000001`` in the exported evaluation input.
            exported = {"id": source_id(row), "target": str(row.get("target", "")),
                        "pred": str(row["corrected_hypothesis"])}
            handle.write(json.dumps(exported, ensure_ascii=False) + "\n")
    return 0


def command_sample(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.input)
    indices = systematic_indices(len(rows), args.count)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index in indices:
            handle.write(json.dumps(rows[index], ensure_ascii=False) + "\n")
    print(json.dumps({"source_rows": len(rows), "sample_rows": len(indices), "indices": indices}))
    return 0


def command_normalize(args: argparse.Namespace) -> int:
    """Apply configured deterministic postprocessors without an LLM call."""
    rows = read_jsonl(args.input)
    prompts = load_prompt_bundle(args.prompt)
    engine = CorrectionEngine(None, prompts)  # type: ignore[arg-type]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    changed = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            row_id = str(row.get("id"))
            hypothesis = prediction_text(row)
            corrected, edits = engine.apply_postprocessors(hypothesis)
            changed += bool(edits)
            result = {
                "id": row_id,
                "status": "ok",
                "target": target_text(row),
                "original_hypothesis": hypothesis,
                "corrected_hypothesis": corrected,
                "rounds": [],
                "stop_reason": "deterministic_postprocessor" if edits else "no_changes",
                "seen_hypotheses": [hypothesis, corrected],
                "deterministic_edits": edits,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    manifest = {
        "input": str(Path(args.input).resolve()),
        "output": str(output.resolve()),
        "prompt_version": prompts["version"],
        "prompt_sha256": prompts["sha256"],
        "deterministic_postprocessors": list(prompts.get("deterministic_postprocessors", [])),
        "test_target_visible_to_engine": False,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "changed": changed,
                      "output": str(output)}, ensure_ascii=False))
    return 0


def command_alternate_local(args: argparse.Namespace) -> int:
    """Run the opt-in content-only local alternate selector."""
    rows = read_jsonl(args.input)
    alternate_rows = read_jsonl(args.alternate_input)
    alternate_by_exact_id = {
        str(row.get("id")): prediction_text(row)
        for row in alternate_rows if prediction_text(row)
    }
    alternate_by_source_id = {
        source_id(row): prediction_text(row)
        for row in alternate_rows if prediction_text(row)
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            row_id = str(row.get("id"))
            primary = prediction_text(row)
            alternate = alternate_by_exact_id.get(
                row_id, alternate_by_source_id.get(source_id(row), "")
            )
            corrected, edits = propose_local_edits(
                primary,
                alternate,
                max_ngram=args.max_ngram,
                max_span_chars=args.max_span_chars,
                max_token_ratio=getattr(args, "max_token_ratio", 4.0),
            ) if alternate else (primary, [])
            result = {
                "id": row_id,
                "status": "ok",
                "target": target_text(row),
                "original_hypothesis": primary,
                "corrected_hypothesis": corrected,
                "rounds": [{
                    "round_index": 1,
                    "input_hypothesis": primary,
                    "segments": [],
                    "candidates": [],
                    "suggestions": [],
                    "verified_suggestions": [],
                    "verification_decisions": [],
                    "output_hypothesis": corrected,
                    "applied_edits": edits,
                    "rejected_edits": [],
                }],
                "stop_reason": "alternate_local_selector" if edits else "no_changes",
                "seen_hypotheses": [primary, corrected],
                "deterministic_edits": [],
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    manifest = {
        "input": str(Path(args.input).resolve()),
        "alternate_input": str(Path(args.alternate_input).resolve()),
        "output": str(output.resolve()),
        "selector": "alternate_local_selector_content_only",
        "safety_gate": "span_and_token_ratio_and_new_repeated_ngram_rejected",
        "max_ngram": args.max_ngram,
        "max_span_chars": args.max_span_chars,
        "max_token_ratio": getattr(args, "max_token_ratio", 4.0),
        "code_sha256": code_sha256(),
        "llm_only": True,
        "normalization_only_edits_forbidden": True,
        "deterministic_postprocessors": [],
        "test_target_visible_to_engine": False,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def command_alternate_consensus(args: argparse.Namespace) -> int:
    """Run the opt-in multi-alternate content-consensus selector."""
    rows = read_jsonl(args.input)
    alternate_maps = []
    for path in args.alternate_input:
        alternate_rows = read_jsonl(path)
        alternate_maps.append((
            {str(row.get("id")): prediction_text(row)
             for row in alternate_rows if prediction_text(row)},
            {source_id(row): prediction_text(row)
             for row in alternate_rows if prediction_text(row)},
        ))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            row_id = str(row.get("id"))
            primary = prediction_text(row)
            alternates = [
                exact.get(row_id, by_source.get(source_id(row), ""))
                for exact, by_source in alternate_maps
            ]
            corrected, edits = propose_consensus_edits(
                primary,
                alternates,
                min_agreement=args.min_agreement,
                max_ngram=args.max_ngram,
                max_span_chars=args.max_span_chars,
                max_token_ratio=getattr(args, "max_token_ratio", 4.0),
            )
            handle.write(json.dumps({
                "id": row_id,
                "status": "ok",
                "target": target_text(row),
                "original_hypothesis": primary,
                "corrected_hypothesis": corrected,
                "rounds": [{
                    "round_index": 1,
                    "input_hypothesis": primary,
                    "segments": [],
                    "candidates": [],
                    "suggestions": [],
                    "verified_suggestions": [],
                    "verification_decisions": [],
                    "output_hypothesis": corrected,
                    "applied_edits": edits,
                    "rejected_edits": [],
                }],
                "stop_reason": "alternate_consensus_selector" if edits else "no_changes",
                "seen_hypotheses": [primary, corrected],
                "deterministic_edits": [],
            }, ensure_ascii=False) + "\n")
    manifest = {
        "input": str(Path(args.input).resolve()),
        "alternate_inputs": [str(Path(path).resolve()) for path in args.alternate_input],
        "output": str(output.resolve()),
        "selector": "alternate_consensus_selector_content_only",
        "min_agreement": args.min_agreement,
        "safety_gate": "multi_alternate_content_agreement_and_span_token_ratio_and_new_repeated_ngram_rejected",
        "max_ngram": args.max_ngram,
        "max_span_chars": args.max_span_chars,
        "max_token_ratio": getattr(args, "max_token_ratio", 4.0),
        "code_sha256": code_sha256(),
        "llm_only": True,
        "normalization_only_edits_forbidden": True,
        "deterministic_postprocessors": [],
        "test_target_visible_to_engine": False,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def command_cseer(args: argparse.Namespace) -> int:
    import sys
    root = Path(args.data_prepare_root).resolve()
    metric_scripts = root / "scripts" / "26.8.2"
    testing_scripts = root / "benchmark" / "testing" / "scripts"
    sys.path[:0] = [str(metric_scripts), str(testing_scripts)]
    spec = importlib.util.spec_from_file_location("post_correction_ser_metrics", metric_scripts / "_ser_metrics.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load _ser_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.compute_ser_variants(Path(args.model_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_whisper_evidence(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[:args.limit]
    messages = read_jsonl(args.message_file)
    message_by_id = {str(row.get("id")): row for row in messages}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ids(output)
    pending = [row for row in rows if str(row.get("id")) not in completed]
    transcriber = LocalWhisperTranscriber(args.model_dir, device=args.device, dtype=args.dtype)
    for index, row in enumerate(pending, 1):
        row_id = str(row.get("id"))
        audio_path = resolve_audio_path(row, message_by_id, args.audio_root)
        try:
            prediction, elapsed = transcriber.transcribe(audio_path, language=args.language)
            if not prediction:
                raise ValueError("empty Whisper prediction")
            append_jsonl(output, {
                "id": row_id,
                "source_id": str(row.get("source_id") or row_id.rsplit(":", 1)[-1]),
                "status": "ok",
                "target": target_text(row),
                "pred": prediction,
                "audio": str(audio_path.resolve()),
                "infer_time_sec": elapsed,
                "model": "whisper",
            })
        except Exception as exc:
            append_jsonl(output, {
                "id": row_id,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "audio": str(audio_path),
            })
        print(f"[{index}/{len(pending)}] {row_id}", flush=True)
    manifest = {
        "input": str(Path(args.input).resolve()),
        "message_file": str(Path(args.message_file).resolve()),
        "audio_root": str(Path(args.audio_root).resolve()),
        "output": str(output.resolve()),
        "model_dir": str(Path(args.model_dir).resolve()),
        "device": args.device,
        "dtype": args.dtype,
        "language": args.language,
        "test_target_visible_to_engine": False,
        "purpose": "non-GT alternate ASR evidence",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def command_qwen2_audio_evidence(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[:args.limit]
    messages = read_jsonl(args.message_file)
    message_by_id = {str(row.get("id")): row for row in messages}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ids(output)
    pending = [row for row in rows if str(row.get("id")) not in completed]
    transcriber = LocalQwen2AudioTranscriber(
        args.model_dir, dtype=args.dtype, max_new_tokens=args.max_new_tokens
    )
    for index, row in enumerate(pending, 1):
        row_id = str(row.get("id"))
        audio_path = resolve_audio_path(row, message_by_id, args.audio_root)
        try:
            prediction, elapsed = transcriber.transcribe(audio_path, language=args.language)
            if not prediction:
                raise ValueError("empty Qwen2-Audio prediction")
            append_jsonl(output, {
                "id": row_id,
                "source_id": str(row.get("source_id") or row_id.rsplit(":", 1)[-1]),
                "status": "ok",
                "target": target_text(row),
                "pred": prediction,
                "audio": str(audio_path.resolve()),
                "infer_time_sec": elapsed,
                "model": "qwen2-audio-7b",
            })
        except Exception as exc:
            append_jsonl(output, {
                "id": row_id,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "audio": str(audio_path),
            })
        print(f"[{index}/{len(pending)}] {row_id}", flush=True)
    manifest = {
        "input": str(Path(args.input).resolve()),
        "message_file": str(Path(args.message_file).resolve()),
        "audio_root": str(Path(args.audio_root).resolve()),
        "output": str(output.resolve()),
        "model_dir": str(Path(args.model_dir).resolve()),
        "dtype": args.dtype,
        "language": args.language,
        "max_new_tokens": args.max_new_tokens,
        "test_target_visible_to_engine": False,
        "purpose": "non-GT acoustic transcription evidence",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def command_acoustic_rerank(args: argparse.Namespace) -> int:
    from .io import source_id

    primary_rows = read_jsonl(args.input)
    if args.limit:
        primary_rows = primary_rows[:args.limit]
    messages = read_jsonl(args.message_file)
    message_by_id = {str(row.get("id")): row for row in messages}
    sources: list[tuple[str, dict[str, dict[str, Any]]]] = []
    for specification in args.candidate_input or []:
        if "=" not in specification:
            raise ValueError("--candidate-input must be NAME=PATH")
        name, path = specification.split("=", 1)
        indexed = {source_id(row): row for row in read_jsonl(path)}
        sources.append((name, indexed))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ids(output)
    pending = [row for row in primary_rows if str(row.get("id")) not in completed]
    scorer = LocalQwen2AudioTranscriber(
        args.model_dir, dtype=args.dtype, max_new_tokens=1
    )
    for index, row in enumerate(pending, 1):
        row_id = str(row.get("id"))
        local_id = source_id(row)
        original = prediction_text(row)
        candidates: list[dict[str, str]] = [{"source": "primary", "text": original}]
        seen = {original}
        for name, indexed in sources:
            candidate_row = indexed.get(local_id)
            if candidate_row is None or candidate_row.get("status") == "error":
                continue
            text = prediction_text(candidate_row)
            if text and text not in seen:
                seen.add(text)
                candidates.append({"source": name, "text": text})
        audio_path = resolve_audio_path(row, message_by_id, args.audio_root)
        try:
            scores, elapsed = scorer.score_transcripts(
                audio_path,
                [candidate["text"] for candidate in candidates],
                language=args.language,
            )
            ranked = [
                {**candidate, **score}
                for candidate, score in zip(candidates, scores)
            ]
            ranked.sort(key=lambda item: float(item["mean_log_probability"]), reverse=True)
            selected = ranked[0]
            if CorrectionEngine._is_normalization_only_edit(original, str(selected["text"])):
                selected = next(item for item in ranked if item["source"] == "primary")
                selection_reason = "format_only_winner_blocked"
            else:
                selection_reason = "highest_acoustic_mean_log_probability"
            corrected = str(selected["text"])
            append_jsonl(output, {
                "id": row_id,
                "status": "ok",
                "target": target_text(row),
                "original_hypothesis": original,
                "corrected_hypothesis": corrected,
                "rounds": [],
                "stop_reason": "acoustic_rerank" if corrected != original else "no_changes",
                "seen_hypotheses": [original] if corrected == original else [original, corrected],
                "selected_source": selected["source"],
                "selection_reason": selection_reason,
                "acoustic_candidates": ranked,
                "audio": str(audio_path.resolve()),
                "infer_time_sec": elapsed,
            })
        except Exception as exc:
            append_jsonl(output, {
                "id": row_id,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "audio": str(audio_path),
            })
        print(f"[{index}/{len(pending)}] {row_id}", flush=True)
    manifest = {
        "input": str(Path(args.input).resolve()),
        "candidate_inputs": list(args.candidate_input or []),
        "message_file": str(Path(args.message_file).resolve()),
        "audio_root": str(Path(args.audio_root).resolve()),
        "output": str(output.resolve()),
        "model_dir": str(Path(args.model_dir).resolve()),
        "dtype": args.dtype,
        "language": args.language,
        "selection_strategy": "Qwen2-Audio conditional mean token log probability",
        "format_only_winners_blocked": True,
        "deterministic_postprocessors": [],
        "test_target_visible_to_engine": False,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rubric-guided ASR post-correction")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--input", required=True)
    run.add_argument("--message-file")
    run.add_argument("--context-input", help="Full prediction JSONL used only to build preceding-hypothesis context")
    run.add_argument("--alternate-input", action="append",
                     help="An alternate ASR transcript used as non-GT evidence; repeat for multiple systems")
    run.add_argument("--ocr-file", help="Slide OCR JSONL used as non-GT evidence")
    run.add_argument("--output", required=True)
    run.add_argument("--prompt", required=True)
    run.add_argument("--mode", choices=("single", "multi"), default="single")
    run.add_argument("--model", default="deepseek-v4-pro")
    run.add_argument("--base-url", default="https://apifusion.aispeech.com.cn/v1")
    run.add_argument("--cache-dir", default="cache")
    run.add_argument("--workers", type=int, default=2)
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--max-rounds", type=int, default=4)
    run.add_argument("--history-size", type=int, default=10)
    run.add_argument("--max-segment-chars", type=int, default=80)
    run.add_argument("--detector-threshold", type=float, default=0.72)
    run.add_argument("--corrector-threshold", type=float, default=0.82)
    run.add_argument("--verifier-threshold", type=float, default=0.82)
    run.add_argument("--limit", type=int)
    run.add_argument("--llm-only", action="store_true",
                     help="Reject prompts containing deterministic postprocessors")
    run.set_defaults(func=command_run)

    baseline = sub.add_parser(
        "gt-context-baseline",
        help="Single-pass correction on the non-AISHELL benchmark using stored GT history",
    )
    baseline.add_argument("--data-prepare-root", required=True)
    baseline.add_argument("--model-key", choices=("qwen3-asr", "sensevoice"), required=True)
    baseline.add_argument("--output", required=True)
    baseline.add_argument("--prompt", default="prompts/gt_context_baseline_v1.json")
    baseline.add_argument("--model", default="deepseek-v4-pro")
    baseline.add_argument("--base-url", default="https://apifusion.aispeech.com.cn/v1")
    baseline.add_argument("--cache-dir", default="cache")
    baseline.add_argument("--workers", type=int, default=4)
    baseline.add_argument("--max-attempts", type=int, default=3)
    baseline.add_argument("--limit", type=int)
    baseline.add_argument("--sample-count", type=int)
    baseline.add_argument("--parent-corrected")
    baseline.set_defaults(func=command_gt_context_baseline)

    evaluate = sub.add_parser(
        "evaluate", help="Evaluate source rows with data-prepare's WER implementation"
    )
    evaluate.add_argument("--data-prepare-root", required=True)
    evaluate.add_argument("--before", required=True)
    evaluate.add_argument("--corrected", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--require-complete", action="store_true")
    evaluate.add_argument("--require-llm-only", action="store_true")
    evaluate.set_defaults(func=command_evaluate)

    headline = sub.add_parser(
        "headline-evaluate",
        help="Reproduce the non-AISHELL benchmark headline or evaluate corrections",
    )
    headline.add_argument("--data-prepare-root", required=True)
    headline.add_argument("--model-key", choices=("qwen3-asr", "sensevoice"), required=True)
    headline.add_argument("--corrected")
    headline.add_argument("--output-dir", required=True)
    headline.add_argument("--require-llm-only", action="store_true")
    headline.set_defaults(func=command_headline_evaluate)

    prepare_cseer = sub.add_parser(
        "prepare-headline-cseer",
        help="Build authoritative data-prepare mapping/classification inputs for corrected headline rows",
    )
    prepare_cseer.add_argument("--data-prepare-root", required=True)
    prepare_cseer.add_argument("--model-key", choices=("qwen3-asr", "sensevoice"), required=True)
    prepare_cseer.add_argument("--corrected", required=True)
    prepare_cseer.add_argument("--model-dir", required=True)
    prepare_cseer.add_argument("--sample-count", type=int)
    prepare_cseer.set_defaults(func=command_prepare_headline_cseer)

    finalize_cseer = sub.add_parser(
        "finalize-headline-cseer",
        help="Promote rubric classifications and compute authoritative non-AISHELL c-SEER2",
    )
    finalize_cseer.add_argument("--data-prepare-root", required=True)
    finalize_cseer.add_argument("--model-dir", required=True)
    finalize_cseer.add_argument("--output", required=True)
    finalize_cseer.set_defaults(func=command_finalize_headline_cseer)

    export = sub.add_parser("export")
    export.add_argument("--corrected", required=True)
    export.add_argument("--output", required=True)
    export.set_defaults(func=command_export)
    sample = sub.add_parser("sample", help="Deterministic content-blind systematic sample")
    sample.add_argument("--input", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--count", type=int, required=True)
    sample.set_defaults(func=command_sample)
    normalize = sub.add_parser("normalize", help="Apply deterministic prompt postprocessors without an API call")
    normalize.add_argument("--input", required=True)
    normalize.add_argument("--output", required=True)
    normalize.add_argument("--prompt", required=True)
    normalize.set_defaults(func=command_normalize)
    alternate_local = sub.add_parser(
        "alternate-local",
        help="Experimental content-only local edits from one alternate ASR transcript",
    )
    alternate_local.add_argument("--input", required=True)
    alternate_local.add_argument("--alternate-input", required=True)
    alternate_local.add_argument("--output", required=True)
    alternate_local.add_argument("--max-ngram", type=int, default=4)
    alternate_local.add_argument("--max-span-chars", type=int, default=15)
    alternate_local.add_argument("--max-token-ratio", type=float, default=4.0)
    alternate_local.set_defaults(func=command_alternate_local)
    alternate_consensus = sub.add_parser(
        "alternate-consensus",
        help="Experimental local edits supported by multiple alternate ASR transcripts",
    )
    alternate_consensus.add_argument("--input", required=True)
    alternate_consensus.add_argument("--alternate-input", action="append", required=True)
    alternate_consensus.add_argument("--output", required=True)
    alternate_consensus.add_argument("--min-agreement", type=int, default=2)
    alternate_consensus.add_argument("--max-ngram", type=int, default=4)
    alternate_consensus.add_argument("--max-span-chars", type=int, default=15)
    alternate_consensus.add_argument("--max-token-ratio", type=float, default=4.0)
    alternate_consensus.set_defaults(func=command_alternate_consensus)
    cseer = sub.add_parser("cseer", help="Reuse data-prepare's authoritative c-SEER implementation")
    cseer.add_argument("--data-prepare-root", required=True)
    cseer.add_argument("--model-dir", required=True,
                       help="Directory containing pred.raw.jsonl, llm_classification_input.jsonl and results/")
    cseer.set_defaults(func=command_cseer)
    whisper = sub.add_parser("whisper-evidence", help="Generate offline Whisper alternate transcripts")
    whisper.add_argument("--input", required=True)
    whisper.add_argument("--message-file", required=True)
    whisper.add_argument("--audio-root", required=True)
    whisper.add_argument("--model-dir", required=True)
    whisper.add_argument("--output", required=True)
    whisper.add_argument("--device", default="cpu")
    whisper.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    whisper.add_argument("--language", default="auto")
    whisper.add_argument("--limit", type=int)
    whisper.set_defaults(func=command_whisper_evidence)
    qwen2_audio = sub.add_parser("qwen2-audio-evidence", help="Generate offline Qwen2-Audio evidence")
    qwen2_audio.add_argument("--input", required=True)
    qwen2_audio.add_argument("--message-file", required=True)
    qwen2_audio.add_argument("--audio-root", required=True)
    qwen2_audio.add_argument("--model-dir", required=True)
    qwen2_audio.add_argument("--output", required=True)
    qwen2_audio.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    qwen2_audio.add_argument("--language", default="Chinese")
    qwen2_audio.add_argument("--max-new-tokens", type=int, default=256)
    qwen2_audio.add_argument("--limit", type=int)
    qwen2_audio.set_defaults(func=command_qwen2_audio_evidence)
    rerank = sub.add_parser("acoustic-rerank", help="Rerank raw ASR candidates with Qwen2-Audio")
    rerank.add_argument("--input", required=True)
    rerank.add_argument("--candidate-input", action="append", help="Repeatable NAME=PATH prediction source")
    rerank.add_argument("--message-file", required=True)
    rerank.add_argument("--audio-root", required=True)
    rerank.add_argument("--model-dir", required=True)
    rerank.add_argument("--output", required=True)
    rerank.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    rerank.add_argument("--language", default="Chinese")
    rerank.add_argument("--limit", type=int)
    rerank.set_defaults(func=command_acoustic_rerank)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
