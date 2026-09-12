from __future__ import annotations

import importlib.util
import json
import sys
import unicodedata
from pathlib import Path
from types import ModuleType
from typing import Any

from .io import prediction_text, read_jsonl, target_text


HEADLINE_UTTERANCES = 467
HEADLINE_REFERENCE_TOKENS = 18_284
HEADLINE_BASELINES = {
    "qwen3-asr": 12.97309122730256,
    "sensevoice": 11.868300153139357,
}


def _load_data_prepare(data_prepare_root: str | Path) -> tuple[ModuleType, ModuleType]:
    root = Path(data_prepare_root).resolve()
    testing_scripts = root / "benchmark" / "testing" / "scripts"
    report_scripts = root / "scripts" / "26.8.2"
    for path in (testing_scripts, report_scripts):
        if not path.is_dir():
            raise ValueError(f"missing data-prepare script directory: {path}")
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    from asr_testing import wer

    module_path = report_scripts / "generate_metrics_table_t1.py"
    spec = importlib.util.spec_from_file_location(
        "asr_post_correction_data_prepare_t1", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load data-prepare evaluator: {module_path}")
    t1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t1)
    return wer, t1


def _score_dict(stats: Any) -> dict[str, Any]:
    return {
        "wer": stats.wer,
        "errors": stats.errors,
        "reference_tokens": stats.ref_chars,
        "substitutions": stats.substitutions,
        "insertions": stats.insertions,
        "deletions": stats.deletions,
        "evaluated": stats.evaluated,
        "missing_predictions": stats.missing_gt,
    }


def _write_analysis(
    summary: dict[str, Any], cases: list[dict[str, Any]], output_dir: str | Path
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for outcome in ("improved", "regressed", "neutral"):
        with (root / f"{outcome}.jsonl").open("w", encoding="utf-8") as handle:
            for case in cases:
                if case["outcome"] == outcome:
                    handle.write(json.dumps(case, ensure_ascii=False) + "\n")


def validate_llm_only_output(corrected_path: str | Path) -> dict[str, Any]:
    """Audit correction provenance without implementing an evaluation metric."""
    corrected_path = Path(corrected_path)
    manifest_path = corrected_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise ValueError(f"LLM-only output is missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("llm_only") is not True:
        raise ValueError("manifest does not explicitly declare llm_only=true")
    if manifest.get("normalization_only_edits_forbidden") is not True:
        raise ValueError("manifest does not declare normalization-only edits forbidden")
    postprocessors = manifest.get("deterministic_postprocessors", [])
    if postprocessors:
        raise ValueError(
            f"LLM-only output configures deterministic postprocessors: {postprocessors}"
        )

    deterministic_edits = 0
    normalization_only_edits = 0

    def content_signature(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        return "".join(character for character in normalized if character.isalnum())

    for row in read_jsonl(corrected_path):
        deterministic_edits += len(row.get("deterministic_edits") or [])
        for round_result in row.get("rounds") or []:
            for edit in round_result.get("applied_edits") or []:
                deterministic_edits += edit.get("source") == "deterministic_postprocessor"
                original = str(edit.get("original", ""))
                replacement = str(edit.get("replacement", ""))
                normalization_only_edits += (
                    original != replacement
                    and content_signature(original) == content_signature(replacement)
                )
    if deterministic_edits:
        raise ValueError(f"LLM-only output contains {deterministic_edits} deterministic edits")
    if normalization_only_edits:
        raise ValueError(
            f"LLM-only output contains {normalization_only_edits} normalization-only edits"
        )
    return {
        "manifest": str(manifest_path.resolve()),
        "llm_only": True,
        "deterministic_postprocessors": [],
        "deterministic_edits": 0,
        "normalization_only_edits": 0,
    }


def _compare_maps(
    wer: ModuleType,
    ground_truth: dict[tuple[str, str], str],
    before: dict[tuple[str, str], str],
    after: dict[tuple[str, str], str],
    traces: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    before_stats = wer.compute_wer(before, ground_truth)
    after_stats = wer.compute_wer(after, ground_truth)
    cases: list[dict[str, Any]] = []
    for key in ground_truth:
        original = before[key]
        corrected = after[key]
        if original == corrected:
            continue
        row_gt = {key: ground_truth[key]}
        before_row = wer.compute_wer({key: original}, row_gt)
        after_row = wer.compute_wer({key: corrected}, row_gt)
        outcome = (
            "improved"
            if after_row.errors < before_row.errors
            else "regressed"
            if after_row.errors > before_row.errors
            else "neutral"
        )
        cases.append(
            {
                "id": f"{key[0]}:{key[1]}",
                "target": ground_truth[key],
                "before": original,
                "after": corrected,
                "before_errors": before_row.errors,
                "after_errors": after_row.errors,
                "outcome": outcome,
                "trace": traces.get(key, []),
            }
        )
    before_score = _score_dict(before_stats)
    after_score = _score_dict(after_stats)
    reduction = (
        100.0 * (before_stats.wer - after_stats.wer) / before_stats.wer
        if before_stats.wer
        else None
    )
    return {
        "metric_implementation": str(Path(wer.__file__).resolve()),
        "before": before_score,
        "after": after_score,
        "relative_wer_reduction_pct": reduction,
        "changed": len(cases),
        "improved": sum(case["outcome"] == "improved" for case in cases),
        "regressed": sum(case["outcome"] == "regressed" for case in cases),
        "neutral": sum(case["outcome"] == "neutral" for case in cases),
    }, cases


def _source_dataset_breakdown(
    wer: ModuleType,
    composite_source_map: dict[str, tuple[str, str]],
    ground_truth: dict[tuple[str, str], str],
    before: dict[tuple[str, str], str],
    after: dict[tuple[str, str], str] | None = None,
) -> dict[str, Any]:
    datasets = sorted(
        {
            composite_source_map[f"{key[0]}:{key[1]}"][0]
            for key in ground_truth
        }
    )
    breakdown: dict[str, Any] = {}
    for dataset in datasets:
        keys = {
            key
            for key in ground_truth
            if composite_source_map[f"{key[0]}:{key[1]}"][0] == dataset
        }
        group_gt = {key: ground_truth[key] for key in keys}
        before_stats = wer.compute_wer(
            {key: before[key] for key in keys}, group_gt
        )
        item: dict[str, Any] = {
            "utterances": len(keys),
            "before": _score_dict(before_stats),
        }
        if after is not None:
            after_stats = wer.compute_wer(
                {key: after[key] for key in keys}, group_gt
            )
            item["after"] = _score_dict(after_stats)
            item["relative_wer_reduction_pct"] = (
                100.0 * (before_stats.wer - after_stats.wer) / before_stats.wer
                if before_stats.wer
                else None
            )
        breakdown[dataset] = item
    return breakdown


def _headline_corpus_breakdown(
    wer: ModuleType,
    composite_source_map: dict[str, tuple[str, str]],
    ground_truth: dict[tuple[str, str], str],
    before: dict[tuple[str, str], str],
    after: dict[tuple[str, str], str] | None = None,
) -> dict[str, Any]:
    grouped_map = {
        composite_id: (
            "TEDLIUM_release-3"
            if source[0].startswith("TEDLIUM_release-3")
            else source[0],
            source[1],
        )
        for composite_id, source in composite_source_map.items()
    }
    return _source_dataset_breakdown(
        wer, grouped_map, ground_truth, before, after
    )


def evaluate_files(
    data_prepare_root: str | Path,
    before_path: str | Path,
    corrected_path: str | Path,
    output_dir: str | Path,
    *,
    require_complete: bool = False,
    require_llm_only: bool = False,
) -> dict[str, Any]:
    wer, _ = _load_data_prepare(data_prepare_root)
    source_rows = {str(row["id"]): row for row in read_jsonl(before_path)}
    all_results = read_jsonl(corrected_path)
    results = {
        str(row["id"]): row for row in all_results if row.get("status") == "ok"
    }
    failed = sum(row.get("status") != "ok" for row in all_results)
    missing = sorted(source_rows.keys() - results.keys())
    extra = sorted(results.keys() - source_rows.keys())
    if require_complete and (missing or extra or failed):
        raise ValueError(
            f"incomplete result: missing={len(missing)}, extra={len(extra)}, failed={failed}"
        )
    ids = sorted(source_rows.keys() & results.keys())
    keys = {row_id: ("input", row_id) for row_id in ids}
    ground_truth = {keys[row_id]: target_text(source_rows[row_id]) for row_id in ids}
    before = {keys[row_id]: prediction_text(source_rows[row_id]) for row_id in ids}
    after = {
        keys[row_id]: str(results[row_id].get("corrected_hypothesis", ""))
        for row_id in ids
    }
    mismatches = sum(
        prediction_text({"pred": results[row_id].get("original_hypothesis", "")})
        != before[keys[row_id]]
        for row_id in ids
    )
    if require_llm_only and mismatches:
        raise ValueError(
            f"{mismatches} original_hypothesis values do not match raw input predictions"
        )
    traces = {keys[row_id]: results[row_id].get("rounds", []) for row_id in ids}
    summary, cases = _compare_maps(wer, ground_truth, before, after, traces)
    summary.update(
        {
            "scope": "provided source rows",
            "utterances": len(ids),
            "source_rows": len(source_rows),
            "successful_rows": len(results),
            "failed_rows": failed,
            "missing_rows": len(missing),
            "extra_rows": len(extra),
            "raw_input_mismatches": mismatches,
        }
    )
    if require_llm_only:
        summary["llm_only_audit"] = validate_llm_only_output(corrected_path)
    _write_analysis(summary, cases, output_dir)
    return summary


def evaluate_headline(
    data_prepare_root: str | Path,
    model_key: str,
    output_dir: str | Path,
    *,
    corrected_path: str | Path | None = None,
    require_llm_only: bool = False,
) -> dict[str, Any]:
    if model_key not in HEADLINE_BASELINES:
        raise ValueError(f"unsupported headline model: {model_key}")
    root = Path(data_prepare_root).resolve()
    wer, t1 = _load_data_prepare(root)
    benchmark_root = root / "benchmark" / "benchmark"
    prediction_root = root / "benchmark" / "testing" / "reports" / "asr_full" / "predictions"
    composite_source_map = t1.load_composite_source_map(benchmark_root)
    allowed = t1.build_allowed_composites(
        composite_source_map, source_dedupe=False, exclude_aishell4=True
    )
    if len(allowed) != HEADLINE_UTTERANCES:
        raise ValueError(
            f"headline scope drifted: expected {HEADLINE_UTTERANCES} rows, got {len(allowed)}"
        )
    allowed_keys = {tuple(composite_id.split(":", 1)) for composite_id in allowed}
    ground_truth = {
        key: value
        for key, value in wer.load_benchmark_ground_truth(benchmark_root).items()
        if key in allowed_keys
    }
    before = wer.load_model_predictions(
        prediction_root, model_key, benchmark_root=benchmark_root
    )
    before = {key: before[key] for key in ground_truth if key in before}
    if set(before) != set(ground_truth):
        raise ValueError(
            f"incomplete baseline: expected={len(ground_truth)}, predictions={len(before)}"
        )
    baseline_stats = wer.compute_wer(before, ground_truth)
    if baseline_stats.ref_chars != HEADLINE_REFERENCE_TOKENS:
        raise ValueError(
            "headline denominator drifted: "
            f"expected {HEADLINE_REFERENCE_TOKENS}, got {baseline_stats.ref_chars}"
        )
    expected_wer = HEADLINE_BASELINES[model_key]
    if abs(baseline_stats.wer - expected_wer) > 1e-12:
        raise ValueError(
            f"headline baseline drifted: expected {expected_wer}, got {baseline_stats.wer}"
        )

    if corrected_path is None:
        summary = {
            "scope": "benchmark composites excluding AISHELL-4, without source dedupe",
            "model_key": model_key,
            "utterances": len(ground_truth),
            "source_dedupe": False,
            "exclude_aishell4": True,
            "metric_implementation": str(Path(wer.__file__).resolve()),
            "baseline": _score_dict(baseline_stats),
            "by_source_dataset": _source_dataset_breakdown(
                wer, composite_source_map, ground_truth, before
            ),
            "by_corpus": _headline_corpus_breakdown(
                wer, composite_source_map, ground_truth, before
            ),
        }
        _write_analysis(summary, [], output_dir)
        return summary

    all_results = read_jsonl(corrected_path)
    results = {
        str(row["id"]): row for row in all_results if row.get("status") == "ok"
    }
    failed = sum(row.get("status") != "ok" for row in all_results)
    required_ids = {f"{key[0]}:{key[1]}" for key in ground_truth}
    missing = sorted(required_ids - results.keys())
    if missing or failed:
        raise ValueError(
            f"incomplete headline result: missing={len(missing)}, failed={failed}"
        )

    after: dict[tuple[str, str], str] = {}
    traces: dict[tuple[str, str], list[dict[str, Any]]] = {}
    raw_mismatches = 0
    normalized_mismatches = 0
    token_mismatches = 0
    for key in ground_truth:
        composite_id = f"{key[0]}:{key[1]}"
        result = results[composite_id]
        original = str(result.get("original_hypothesis", ""))
        raw_mismatches += original != before[key]
        normalized_mismatches += (
            wer.normalize_asr_text(original) != wer.normalize_asr_text(before[key])
        )
        token_mismatches += (
            wer.tokenize_asr_text(original) != wer.tokenize_asr_text(before[key])
        )
        after[key] = str(result.get("corrected_hypothesis", ""))
        traces[key] = result.get("rounds", [])
    if token_mismatches:
        raise ValueError(
            f"{token_mismatches} original_hypothesis values do not match "
            "data-prepare predictions under official WER tokenization"
        )

    summary, cases = _compare_maps(wer, ground_truth, before, after, traces)
    summary.update(
        {
            "scope": "benchmark composites excluding AISHELL-4, without source dedupe",
            "model_key": model_key,
            "utterances": len(ground_truth),
            "source_dedupe": False,
            "exclude_aishell4": True,
            "input_rows": len(all_results),
            "evaluated_rows": len(ground_truth),
            "failed_rows": failed,
            "missing_rows": len(missing),
            "raw_string_mismatches": raw_mismatches,
            "official_normalized_string_mismatches": normalized_mismatches,
            "official_token_input_mismatches": token_mismatches,
            "by_source_dataset": _source_dataset_breakdown(
                wer, composite_source_map, ground_truth, before, after
            ),
            "by_corpus": _headline_corpus_breakdown(
                wer, composite_source_map, ground_truth, before, after
            ),
        }
    )
    if require_llm_only:
        summary["llm_only_audit"] = validate_llm_only_output(corrected_path)
    _write_analysis(summary, cases, output_dir)
    return summary


def evaluate_headline_cseer(
    data_prepare_root: str | Path, model_dir: str | Path
) -> dict[str, Any]:
    root = Path(data_prepare_root).resolve()
    _, t1 = _load_data_prepare(root)
    benchmark_root = root / "benchmark" / "benchmark"
    composite_source_map = t1.load_composite_source_map(benchmark_root)
    headline_allowed = t1.build_allowed_composites(
        composite_source_map, source_dedupe=False, exclude_aishell4=True
    )
    if len(headline_allowed) != HEADLINE_UTTERANCES:
        raise ValueError(
            f"headline scope drifted: expected {HEADLINE_UTTERANCES}, got {len(headline_allowed)}"
        )
    model_ids = {
        str(row.get("id")) for row in read_jsonl(Path(model_dir) / "pred.raw.jsonl")
    }
    allowed = headline_allowed & model_ids
    if not allowed or allowed != model_ids:
        raise ValueError("model-dir predictions are outside the non-AISHELL headline scope")
    result = t1.compute_ser_variants(
        Path(model_dir),
        allowed_composites=allowed,
    )
    return {
        "scope": "provided non-AISHELL benchmark composites, without source dedupe",
        "metric_implementation": str(
            (root / "scripts" / "26.8.2" / "_ser_metrics.py").resolve()
        ),
        **result,
    }
