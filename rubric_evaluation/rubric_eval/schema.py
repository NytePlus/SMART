"""Input, rubric-response, and extractive-evidence validation."""

from __future__ import annotations

from .config import question_map


REQUIRED_INPUT_FIELDS = (
    "sample_id",
    "dataset",
    "target",
    "pred",
    "reference_segment",
    "hypothesis_segment",
    "op_segment",
)

EVIDENCE_SOURCE_TO_FIELD = {
    "reference_span": "reference_segment",
    "hypothesis_span": "hypothesis_segment",
    "reference_text": "target",
    "hypothesis_text": "pred",
    "history": "history",
    "suffix": "suffix",
}


def validate_input_record(record: dict) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_INPUT_FIELDS:
        if field not in record:
            errors.append(f"missing required field: {field}")
        elif not isinstance(record[field], (str, int)):
            errors.append(f"{field} must be a string or integer")
    if "sample_id" in record and not str(record["sample_id"]).strip():
        errors.append("sample_id must not be empty")
    if not str(record.get("reference_segment", "")) and not str(record.get("hypothesis_segment", "")):
        errors.append("reference_segment and hypothesis_segment cannot both be empty")
    for field in ("history", "suffix", "structured_prediction"):
        if field in record and record[field] is not None and not isinstance(record[field], str):
            errors.append(f"{field} must be a string when provided")
    annotation = record.get("human_annotation")
    if annotation is not None:
        if not isinstance(annotation, dict):
            errors.append("human_annotation must be an object")
        elif not isinstance(annotation.get("is_semantic_error"), bool):
            errors.append("human_annotation.is_semantic_error must be boolean")
    return errors


def validate_dataset(records: list[dict]) -> list[dict]:
    issues: list[dict] = []
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        sample_id = str(record.get("sample_id", ""))
        errors = validate_input_record(record)
        if sample_id in seen:
            errors.append("duplicate sample_id")
        seen.add(sample_id)
        if errors:
            issues.append({"line": index, "sample_id": sample_id, "errors": errors})
    return issues


def _validate_evidence(item: dict, question: dict, sample: dict) -> list[str]:
    errors: list[str] = []
    evidence = item.get("evidence")
    if not isinstance(evidence, list):
        return ["evidence must be a list"]

    allowed_sources = set(question.get("allowed_evidence_sources", []))
    allow_empty_sources = set(question.get("allow_empty_evidence_sources", []))
    valid_count = 0
    valid_sources: set[str] = set()
    for index, entry in enumerate(evidence):
        if not isinstance(entry, dict):
            errors.append(f"evidence[{index}] must be an object")
            continue
        source = entry.get("source")
        quote = entry.get("quote")
        if source not in allowed_sources:
            errors.append(f"evidence[{index}].source {source!r} is not allowed for {question['id']}")
            continue
        if not isinstance(quote, str):
            errors.append(f"evidence[{index}].quote must be a string")
            continue
        input_field = EVIDENCE_SOURCE_TO_FIELD.get(source)
        source_text = str(sample.get(input_field, ""))
        if not quote:
            if source in allow_empty_sources and not source_text:
                valid_count += 1
                valid_sources.add(source)
                continue
            errors.append(
                f"evidence[{index}].quote may be empty only when source is an allowed empty span"
            )
            continue
        if quote not in source_text:
            candidate_sources = [
                candidate
                for candidate in question.get("allowed_evidence_sources", [])
                if quote in str(sample.get(EVIDENCE_SOURCE_TO_FIELD.get(candidate), ""))
            ]
            hint = (
                f"; the exact quote exists in allowed source(s) {candidate_sources}, so change the source field"
                if candidate_sources
                else ""
            )
            errors.append(f"evidence[{index}] quote is not present in input source {source}{hint}")
            continue
        valid_count += 1
        valid_sources.add(source)

    if item.get("answer") in question.get("evidence_required_for", []) and valid_count == 0:
        errors.append(f"at least one valid evidence span is required for answer {item.get('answer')}")
    grouped_requirements = question.get(
        "required_evidence_source_groups_by_answer", {}
    ).get(item.get("answer"), [])
    for group in grouped_requirements:
        if not valid_sources.intersection(group):
            errors.append(
                "at least one valid evidence span is required from source group "
                f"{group} for answer {item.get('answer')}"
            )
    return errors


def validate_rubric_judgment(judgment: dict, sample: dict, config: dict) -> list[str]:
    errors: list[str] = []
    allowed_top_level = {"rubric_version", "items"}
    if config.get("analysis_tags"):
        allowed_top_level.add("change_types")
    unknown_top_level = set(judgment) - allowed_top_level
    if unknown_top_level:
        errors.append(f"unknown top-level fields: {sorted(unknown_top_level)}")
    if judgment.get("rubric_version") != config["rubric_version"]:
        errors.append(f"rubric_version must be {config['rubric_version']!r}")

    items = judgment.get("items")
    if not isinstance(items, dict):
        return errors + ["items must be an object"]

    if config.get("analysis_tags"):
        change_types = judgment.get("change_types")
        if not isinstance(change_types, list):
            errors.append("change_types must be a list")
        else:
            if len(change_types) != len(set(change_types)):
                errors.append("change_types must not contain duplicates")
            invalid_tags = set(change_types) - set(config["analysis_tags"])
            if invalid_tags:
                errors.append(f"invalid change_types: {sorted(invalid_tags)}")

    questions = question_map(config)
    missing = set(questions) - set(items)
    unknown = set(items) - set(questions)
    if missing:
        errors.append(f"missing rubric items: {sorted(missing)}")
    if unknown:
        errors.append(f"unknown rubric items: {sorted(unknown)}")

    for question_id, question in questions.items():
        item = items.get(question_id)
        if not isinstance(item, dict):
            continue
        allowed_item_fields = {"answer", "reason_code", "confidence", "evidence"}
        unknown_fields = set(item) - allowed_item_fields
        if unknown_fields:
            errors.append(f"{question_id}: unknown fields: {sorted(unknown_fields)}")
        answer = item.get("answer")
        if answer not in question["allowed_answers"]:
            errors.append(f"{question_id}: invalid answer {answer!r}")
        reason_code = item.get("reason_code")
        if reason_code not in question["reason_codes"]:
            errors.append(f"{question_id}: invalid reason_code {reason_code!r}")
        elif question.get("reason_codes_by_answer"):
            answer_codes = question["reason_codes_by_answer"].get(answer, [])
            if reason_code not in answer_codes:
                errors.append(
                    f"{question_id}: reason_code {reason_code!r} is not valid for answer {answer!r}"
                )
        confidence = item.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            errors.append(f"{question_id}: confidence must be a number")
        elif not 0.0 <= float(confidence) <= 1.0:
            errors.append(f"{question_id}: confidence must be between 0 and 1")
        errors.extend(f"{question_id}: {error}" for error in _validate_evidence(item, question, sample))
    return errors
