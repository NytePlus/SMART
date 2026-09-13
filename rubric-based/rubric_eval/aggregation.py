"""Deterministic, configuration-driven rubric aggregation."""

from __future__ import annotations


def _answer(
    items: dict,
    question_id: str,
    answer_value_mappings: dict[str, dict[str, str]] | None = None,
) -> str | None:
    item = items.get(question_id, {})
    answer = item.get("answer") if isinstance(item, dict) else None
    if answer is None or not answer_value_mappings:
        return answer
    return answer_value_mappings.get(question_id, {}).get(answer, answer)


def _evaluate_condition(
    condition: dict,
    items: dict,
    semantic_ids: list[str],
    answer_value_mappings: dict[str, dict[str, str]] | None = None,
) -> bool:
    if "all" in condition:
        return all(
            _evaluate_condition(child, items, semantic_ids, answer_value_mappings)
            for child in condition["all"]
        )
    if "any" in condition:
        return any(
            _evaluate_condition(child, items, semantic_ids, answer_value_mappings)
            for child in condition["any"]
        )
    if "not" in condition:
        return not _evaluate_condition(
            condition["not"], items, semantic_ids, answer_value_mappings
        )

    predicate = condition.get("predicate")
    values = set(condition.get("values", []))
    if predicate == "always":
        return True
    if predicate == "item_in":
        return _answer(items, condition["question_id"], answer_value_mappings) in values
    if predicate == "item_not_in":
        return _answer(items, condition["question_id"], answer_value_mappings) not in values
    if predicate == "semantic_any":
        return any(
            _answer(items, question_id, answer_value_mappings) in values
            for question_id in semantic_ids
        )
    if predicate == "semantic_all":
        return all(
            _answer(items, question_id, answer_value_mappings) in values
            for question_id in semantic_ids
        )
    raise ValueError(f"unsupported aggregation predicate: {predicate!r}")


def _severity(label: str, impact: str | None, config: dict) -> str:
    if label == "UNSCORABLE":
        return "UNDETERMINED"
    configured = config.get("aggregation", {}).get("severity_by_impact", {})
    if impact in configured:
        return configured[impact]
    if label == "INCONSEQUENTIAL":
        return "MINOR" if impact == "MINOR" else "NONE"
    return "CRITICAL" if impact == "CRITICAL" else "MAJOR"


def aggregate_rubric_result(judgment: dict, config: dict) -> dict:
    """Aggregate validated rubric items without using an LLM final label."""
    items = judgment.get("items", {})
    aggregation = config["aggregation"]
    semantic_ids = aggregation["semantic_question_ids"]
    answer_value_mappings = aggregation.get("answer_value_mappings", {})

    matched_rule = None
    for rule in config["aggregation"]["rules"]:
        if _evaluate_condition(
            rule["when"], items, semantic_ids, answer_value_mappings
        ):
            matched_rule = rule
            break
    if matched_rule is None:
        raise ValueError("rubric aggregation config has no matching fallback rule")

    label = matched_rule["label"]
    alignment_question_id = aggregation.get(
        "alignment_question_id", "G1" if "G1" in items else None
    )
    alignment = (
        _answer(items, alignment_question_id, answer_value_mappings)
        if alignment_question_id
        else None
    )
    alignment_error_answers = set(aggregation.get("alignment_error_answers", ["INVALID"]))
    if label == "UNSCORABLE" and alignment in alignment_error_answers:
        eval_status = "ALIGNMENT_ERROR"
    elif label == "UNSCORABLE":
        eval_status = "UNSCORABLE"
    else:
        eval_status = "SCORABLE"

    semantic_change_answer = aggregation.get("semantic_change_answer", "SEMANTIC_CHANGE")
    semantic_changes = [
        question_id
        for question_id in semantic_ids
        if _answer(items, question_id, answer_value_mappings)
        == semantic_change_answer
    ]
    impact_question_id = aggregation.get("impact_question_id", "C2")
    impact = _answer(items, impact_question_id, answer_value_mappings)
    return {
        "final_label": label,
        "eval_status": eval_status,
        "aggregation_rule": matched_rule["id"],
        "unscorable_reason": matched_rule.get("unscorable_reason"),
        "severity": _severity(label, impact, config),
        "semantic_change_exists": bool(semantic_changes),
        "semantic_change_items": semantic_changes,
        "change_types": judgment.get("change_types", []),
    }
