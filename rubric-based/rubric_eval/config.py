"""Load and validate rubric configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def config_hash(config: dict) -> str:
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_rubric(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_rubric_config(config)
    config["config_hash"] = config_hash(config)
    return config


def validate_rubric_config(config: dict) -> None:
    if not isinstance(config.get("rubric_version"), str) or not config["rubric_version"]:
        raise ValueError("rubric_version must be a non-empty string")
    questions = config.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions must be a non-empty list")

    ids = [question.get("id") for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("rubric question IDs must be unique")
    required_ids = set(config.get("required_question_ids", ids))
    missing = required_ids - set(ids)
    if missing:
        raise ValueError(f"missing required rubric questions: {sorted(missing)}")

    for question in questions:
        question_id = question.get("id", "<unknown>")
        for field in ("key", "dimension", "question"):
            if not isinstance(question.get(field), str) or not question[field]:
                raise ValueError(f"{question_id}: {field} must be a non-empty string")
        answers = question.get("allowed_answers")
        if not isinstance(answers, list) or not answers or len(answers) != len(set(answers)):
            raise ValueError(f"{question_id}: allowed_answers must be a unique non-empty list")
        definitions = question.get("definitions", {})
        if set(definitions) != set(answers):
            raise ValueError(f"{question_id}: every allowed answer needs exactly one definition")
        reason_codes = question.get("reason_codes")
        if not isinstance(reason_codes, list) or not reason_codes:
            raise ValueError(f"{question_id}: reason_codes must be non-empty")
        reason_codes_by_answer = question.get("reason_codes_by_answer")
        if reason_codes_by_answer is not None:
            if set(reason_codes_by_answer) != set(answers):
                raise ValueError(f"{question_id}: reason_codes_by_answer must cover every answer")
            configured_codes = {
                code
                for answer_codes in reason_codes_by_answer.values()
                for code in answer_codes
            }
            if configured_codes != set(reason_codes):
                raise ValueError(
                    f"{question_id}: reason_codes must equal the union of reason_codes_by_answer"
                )
        evidence_sources = question.get("allowed_evidence_sources")
        if (
            not isinstance(evidence_sources, list)
            or not evidence_sources
            or len(evidence_sources) != len(set(evidence_sources))
        ):
            raise ValueError(
                f"{question_id}: allowed_evidence_sources must be a unique non-empty list"
            )
        evidence_required_for = question.get("evidence_required_for")
        if (
            not isinstance(evidence_required_for, list)
            or not set(evidence_required_for).issubset(set(answers))
        ):
            raise ValueError(
                f"{question_id}: evidence_required_for must contain configured answers"
            )
        allow_empty_sources = question.get("allow_empty_evidence_sources", [])
        if (
            not isinstance(allow_empty_sources, list)
            or not set(allow_empty_sources).issubset(set(evidence_sources))
        ):
            raise ValueError(
                f"{question_id}: allow_empty_evidence_sources must contain allowed sources"
            )
        grouped_requirements = question.get(
            "required_evidence_source_groups_by_answer", {}
        )
        if not isinstance(grouped_requirements, dict) or not set(
            grouped_requirements
        ).issubset(set(answers)):
            raise ValueError(
                f"{question_id}: evidence source group requirements must reference configured answers"
            )
        for answer, groups in grouped_requirements.items():
            if not isinstance(groups, list) or not groups:
                raise ValueError(
                    f"{question_id}: evidence source groups for {answer!r} must be non-empty"
                )
            for group in groups:
                if (
                    not isinstance(group, list)
                    or not group
                    or not set(group).issubset(set(evidence_sources))
                ):
                    raise ValueError(
                        f"{question_id}: invalid evidence source group for {answer!r}"
                    )

    aggregation = config.get("aggregation", {})
    semantic_ids = aggregation.get("semantic_question_ids", [])
    if not isinstance(semantic_ids, list) or not set(semantic_ids).issubset(set(ids)):
        raise ValueError("aggregation.semantic_question_ids must reference configured questions")
    questions_by_id = {question["id"]: question for question in questions}
    answer_value_mappings = aggregation.get("answer_value_mappings", {})
    if not isinstance(answer_value_mappings, dict):
        raise ValueError("aggregation.answer_value_mappings must be an object")
    unknown_mapping_questions = set(answer_value_mappings) - set(questions_by_id)
    if unknown_mapping_questions:
        raise ValueError(
            "aggregation.answer_value_mappings references unknown questions: "
            f"{sorted(unknown_mapping_questions)}"
        )
    effective_answers: dict[str, set[str]] = {}
    for question_id, question in questions_by_id.items():
        mapping = answer_value_mappings.get(question_id, {})
        if not isinstance(mapping, dict):
            raise ValueError(
                f"aggregation.answer_value_mappings.{question_id} must be an object"
            )
        unknown_source_answers = set(mapping) - set(question["allowed_answers"])
        if unknown_source_answers:
            raise ValueError(
                f"aggregation.answer_value_mappings.{question_id} has unknown source "
                f"answers: {sorted(unknown_source_answers)}"
            )
        if any(not isinstance(value, str) or not value for value in mapping.values()):
            raise ValueError(
                f"aggregation.answer_value_mappings.{question_id} values must be "
                "non-empty strings"
            )
        effective_answers[question_id] = {
            mapping.get(answer, answer) for answer in question["allowed_answers"]
        }
    rules = aggregation.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("aggregation.rules must be a non-empty ordered list")
    rule_ids = [rule.get("id") for rule in rules]
    if any(not isinstance(rule_id, str) or not rule_id for rule_id in rule_ids):
        raise ValueError("every aggregation rule needs a non-empty string ID")
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("aggregation rule IDs must be unique")
    for rule in rules:
        if rule.get("label") not in {
            "CONSEQUENTIAL",
            "INCONSEQUENTIAL",
            "UNSCORABLE",
        }:
            raise ValueError(f"{rule['id']}: invalid aggregation label")
        _validate_aggregation_condition(
            rule.get("when"),
            questions_by_id,
            semantic_ids,
            rule["id"],
            effective_answers,
        )
    analysis_tags = config.get("analysis_tags", [])
    if not isinstance(analysis_tags, list) or len(analysis_tags) != len(set(analysis_tags)):
        raise ValueError("analysis_tags must be a unique list")


def question_map(config: dict) -> dict[str, dict]:
    return {question["id"]: question for question in config["questions"]}


def _validate_aggregation_condition(
    condition: object,
    questions: dict[str, dict],
    semantic_ids: list[str],
    rule_id: str,
    effective_answers: dict[str, set[str]] | None = None,
) -> None:
    if not isinstance(condition, dict):
        raise ValueError(f"{rule_id}: aggregation condition must be an object")
    for operator in ("all", "any"):
        if operator in condition:
            children = condition[operator]
            if not isinstance(children, list) or not children:
                raise ValueError(f"{rule_id}: {operator} condition must be non-empty")
            for child in children:
                _validate_aggregation_condition(
                    child, questions, semantic_ids, rule_id, effective_answers
                )
            return
    if "not" in condition:
        _validate_aggregation_condition(
            condition["not"], questions, semantic_ids, rule_id, effective_answers
        )
        return

    predicate = condition.get("predicate")
    if predicate == "always":
        return
    values = condition.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{rule_id}: predicate {predicate!r} needs non-empty values")
    if predicate in {"item_in", "item_not_in"}:
        question_id = condition.get("question_id")
        if question_id not in questions:
            raise ValueError(f"{rule_id}: aggregation references unknown question {question_id!r}")
        allowed_answers = (
            effective_answers[question_id]
            if effective_answers is not None
            else set(questions[question_id]["allowed_answers"])
        )
        invalid_values = set(values) - allowed_answers
        if invalid_values:
            raise ValueError(
                f"{rule_id}: invalid answers for {question_id}: {sorted(invalid_values)}"
            )
        return
    if predicate in {"semantic_any", "semantic_all"}:
        semantic_answers = {
            answer
            for question_id in semantic_ids
            for answer in (
                effective_answers[question_id]
                if effective_answers is not None
                else set(questions[question_id]["allowed_answers"])
            )
        }
        invalid_values = set(values) - semantic_answers
        if invalid_values:
            raise ValueError(
                f"{rule_id}: invalid semantic answers: {sorted(invalid_values)}"
            )
        return
    raise ValueError(f"{rule_id}: unsupported aggregation predicate {predicate!r}")
