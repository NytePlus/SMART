"""End-to-end metrics against existing human consequence labels."""

from __future__ import annotations

from collections import Counter


FINAL_LABELS = ("CONSEQUENTIAL", "INCONSEQUENTIAL")


def _human_label(record: dict) -> str:
    value = record.get("human_annotation", {}).get("is_semantic_error")
    if value is True:
        return "CONSEQUENTIAL"
    if value is False:
        return "INCONSEQUENTIAL"
    raise ValueError(f"{record.get('sample_id')}: missing boolean human label")


def _predicted_label(record: dict) -> str:
    if record.get("inference_status") != "SUCCESS":
        return "INFERENCE_FAILED"
    result = record.get("aggregated_result")
    if not isinstance(result, dict):
        return "INFERENCE_FAILED"
    return str(result.get("final_label", "INFERENCE_FAILED"))


def _divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _binary_metrics(pairs: list[tuple[str, str]]) -> dict:
    per_label: dict[str, dict] = {}
    for label in FINAL_LABELS:
        tp = sum(1 for human, pred in pairs if human == label and pred == label)
        fp = sum(1 for human, pred in pairs if human != label and pred == label)
        fn = sum(1 for human, pred in pairs if human == label and pred != label)
        precision = _divide(tp, tp + fp)
        recall = _divide(tp, tp + fn)
        f1 = _divide(2 * precision * recall, precision + recall)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for human, _ in pairs if human == label),
        }

    accuracy = _divide(sum(human == pred for human, pred in pairs), len(pairs))
    balanced_accuracy = (
        sum(per_label[label]["recall"] for label in FINAL_LABELS) / len(FINAL_LABELS)
    )
    macro_f1 = sum(per_label[label]["f1"] for label in FINAL_LABELS) / len(FINAL_LABELS)

    if pairs:
        human_counts = Counter(human for human, _ in pairs)
        pred_counts = Counter(pred for _, pred in pairs)
        expected = sum(
            _divide(human_counts[label], len(pairs)) * _divide(pred_counts[label], len(pairs))
            for label in FINAL_LABELS
        )
        kappa = _divide(accuracy - expected, 1.0 - expected) if expected < 1.0 else 1.0
    else:
        kappa = 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "cohens_kappa": kappa,
        "per_label": per_label,
    }


def evaluate_records(records: list[dict], include_groups: bool = True) -> dict:
    latest: dict[str, dict] = {}
    for record in records:
        latest[str(record.get("sample_id"))] = record
    deduplicated = list(latest.values())

    rows = [(_human_label(record), _predicted_label(record), record) for record in deduplicated]
    scorable = [(human, pred) for human, pred, _ in rows if pred in FINAL_LABELS]
    confusion: dict[str, dict[str, int]] = {
        human: dict(Counter(pred for h, pred, _ in rows if h == human))
        for human in FINAL_LABELS
    }
    total = len(rows)
    successful = sum(pred != "INFERENCE_FAILED" for _, pred, _ in rows)
    unscorable = sum(pred == "UNSCORABLE" for _, pred, _ in rows)
    correct_all = sum(human == pred for human, pred, _ in rows)
    rule_distribution = Counter(
        record.get("aggregated_result", {}).get("aggregation_rule")
        for _, pred, record in rows
        if pred != "INFERENCE_FAILED"
    )
    unscorable_reasons = Counter(
        record.get("aggregated_result", {}).get("unscorable_reason")
        for _, pred, record in rows
        if pred == "UNSCORABLE"
    )

    report = {
        "total_records": total,
        "successful_inference": successful,
        "inference_failures": total - successful,
        "scorable_records": len(scorable),
        "unscorable_records": unscorable,
        "inference_success_rate": _divide(successful, total),
        "scorable_coverage": _divide(len(scorable), total),
        "unscorable_rate": _divide(unscorable, total),
        "accuracy_all": _divide(correct_all, total),
        "scorable_metrics": _binary_metrics(scorable),
        "confusion_matrix": confusion,
        "aggregation_rule_distribution": dict(rule_distribution),
        "unscorable_reason_distribution": dict(unscorable_reasons),
    }
    if include_groups:
        datasets = sorted({str(record.get("dataset", "UNKNOWN")) for _, _, record in rows})
        report["by_dataset"] = {
            dataset: evaluate_records(
                [record for _, _, record in rows if str(record.get("dataset", "UNKNOWN")) == dataset],
                include_groups=False,
            )
            for dataset in datasets
        }
    return report


def disagreement_records(records: list[dict]) -> list[dict]:
    latest: dict[str, dict] = {}
    for record in records:
        latest[str(record.get("sample_id"))] = record
    disagreements: list[dict] = []
    for record in latest.values():
        human = _human_label(record)
        predicted = _predicted_label(record)
        if human == predicted:
            continue
        disagreements.append({
            "sample_id": record.get("sample_id"),
            "dataset": record.get("dataset"),
            "target": record.get("target"),
            "pred": record.get("pred"),
            "reference_segment": record.get("reference_segment"),
            "hypothesis_segment": record.get("hypothesis_segment"),
            "human_label": human,
            "predicted_label": predicted,
            "aggregated_result": record.get("aggregated_result"),
            "rubric_judgment": record.get("rubric_judgment"),
        })
    return disagreements


def render_markdown(report: dict) -> str:
    metrics = report["scorable_metrics"]
    consequential = metrics["per_label"]["CONSEQUENTIAL"]
    lines = [
        "# Rubric Evaluation Report",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total records | {report['total_records']} |",
        f"| Inference success rate | {report['inference_success_rate']:.4f} |",
        f"| Scorable coverage | {report['scorable_coverage']:.4f} |",
        f"| Unscorable rate | {report['unscorable_rate']:.4f} |",
        f"| Accuracy, all records | {report['accuracy_all']:.4f} |",
        f"| Accuracy, scorable records | {metrics['accuracy']:.4f} |",
        f"| Balanced accuracy, scorable records | {metrics['balanced_accuracy']:.4f} |",
        f"| Macro-F1, scorable records | {metrics['macro_f1']:.4f} |",
        f"| Cohen's kappa, scorable records | {metrics['cohens_kappa']:.4f} |",
        f"| Consequential precision | {consequential['precision']:.4f} |",
        f"| Consequential recall | {consequential['recall']:.4f} |",
        f"| Consequential F1 | {consequential['f1']:.4f} |",
        "",
        "## Confusion Matrix",
        "",
        "```json",
        __import__("json").dumps(report["confusion_matrix"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Important Limitation",
        "",
        "The source file contains holistic human final labels but no human item-level rubric annotations. This report measures end-to-end agreement only; it cannot yet distinguish rubric-design errors from LLM item-execution errors.",
    ]
    return "\n".join(lines) + "\n"
