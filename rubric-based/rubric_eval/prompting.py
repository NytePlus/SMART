"""Build a structured rubric prompt from configuration and one input sample."""

from __future__ import annotations

import json


SYSTEM_PROMPT = """You are evaluating the semantic consequence of one aligned ASR error segment.

Follow the supplied rubric exactly. Your job is to answer the fixed rubric items, not to choose the formal final label. Return one JSON object and no Markdown or commentary.

Evidence is extractive: every evidence quote must be copied exactly from the named input field. Never invent, normalize, translate, or paraphrase an evidence quote. For contextual recoverability, act as a reasonable reader who can see only the ASR hypothesis and the available history/suffix. Seeing the reference as a judge does not allow you to assume that the reader can reconstruct reference-only information.
"""


def _render_questions(config: dict) -> str:
    chinese = config.get("prompt_language") == "zh"
    sections: list[str] = []
    for question in config["questions"]:
        definitions = "\n".join(
            f"  - {answer}: {definition}"
            for answer, definition in question["definitions"].items()
        )
        if question.get("reason_codes_by_answer"):
            reason_codes = "\n".join(
                f"  - {answer}: {', '.join(codes)}"
                for answer, codes in question["reason_codes_by_answer"].items()
            )
        else:
            reason_codes = ", ".join(question["reason_codes"])
        evidence_sources = ", ".join(question["allowed_evidence_sources"])
        target = question.get("evaluation_target")
        grouped_requirements = question.get(
            "required_evidence_source_groups_by_answer", {}
        )
        empty_evidence_sources = question.get("allow_empty_evidence_sources", [])
        group_lines = "\n".join(
            f"  - {answer}: "
            + "；并且".join(f"至少引用 {', '.join(group)} 之一" for group in groups)
            for answer, groups in grouped_requirements.items()
        )
        examples = "\n".join(
            f"  - {example}" for example in question.get("boundary_examples", [])
        )
        if chinese:
            examples_block = f"\n边界示例：\n{examples}" if examples else ""
            target_block = f"判定对象：{target}\n" if target else ""
            groups_block = f"\n分侧证据要求：\n{group_lines}" if group_lines else ""
            empty_block = (
                "\n空 span 规则：仅当输入字段本身为空时，"
                f"{', '.join(empty_evidence_sources)} 可使用空 quote 表示缺位。"
                if empty_evidence_sources
                else ""
            )
            sections.append(
                f"{question['id']} [{question['key']}]：{question['question']}\n"
                f"{target_block}"
                f"可选答案：\n{definitions}\n"
                f"原因代码：\n{reason_codes}\n"
                f"允许引用的证据来源：{evidence_sources}\n"
                f"以下答案必须提供证据：{', '.join(question['evidence_required_for']) or '无'}"
                f"{groups_block}"
                f"{empty_block}"
                f"{examples_block}"
            )
        else:
            examples_block = f"\nBoundary examples:\n{examples}" if examples else ""
            sections.append(
                f"{question['id']} [{question['key']}]: {question['question']}\n"
                f"Answers:\n{definitions}\n"
                f"Reason codes: {reason_codes}\n"
                f"Allowed evidence sources: {evidence_sources}\n"
                f"Evidence required for: {', '.join(question['evidence_required_for']) or 'none'}"
                f"{examples_block}"
            )
    return "\n\n".join(sections)


def _output_template(config: dict) -> dict:
    chinese = config.get("prompt_language") == "zh"
    output = {
        "rubric_version": config["rubric_version"],
        "items": {
            question["id"]: {
                "answer": (
                    f"以下选项之一：{' | '.join(question['allowed_answers'])}"
                    if chinese
                    else f"one of: {' | '.join(question['allowed_answers'])}"
                ),
                "reason_code": (
                    f"与答案匹配的原因代码：{' | '.join(question['reason_codes'])}"
                    if chinese
                    else f"one of: {' | '.join(question['reason_codes'])}"
                ),
                "confidence": "0 到 1 之间的数字" if chinese else "number from 0 to 1",
                "evidence": [
                    {
                        "source": (
                            f"以下来源之一：{' | '.join(question['allowed_evidence_sources'])}"
                            if chinese
                            else f"one of: {' | '.join(question['allowed_evidence_sources'])}"
                        ),
                        "quote": (
                            (
                                "从该来源逐字复制的子串；仅当允许的 span 来源本身为空时可用空字符串"
                                if question.get("allow_empty_evidence_sources")
                                else "从该来源逐字复制的非空子串"
                            )
                            if chinese
                            else "exact non-empty substring from that source"
                        ),
                    }
                ],
            }
            for question in config["questions"]
        },
    }
    if config.get("analysis_tags"):
        output["change_types"] = (
            [f"从以下标签中选择零个或多个：{' | '.join(config['analysis_tags'])}"]
            if chinese
            else [f"zero or more of: {' | '.join(config['analysis_tags'])}"]
        )
    return output


def build_messages(sample: dict, config: dict) -> list[dict]:
    chinese = config.get("prompt_language") == "zh"
    principles = "\n".join(f"- {value}" for value in config["principles"])
    input_payload = {
        "sample_id": str(sample["sample_id"]),
        "dataset": str(sample["dataset"]),
        "history": str(sample.get("history", "")),
        "suffix": str(sample.get("suffix", "")),
        "reference_text": str(sample["target"]),
        "hypothesis_text": str(sample["pred"]),
        "reference_span": str(sample["reference_segment"]),
        "hypothesis_span": str(sample["hypothesis_segment"]),
        "edit_operations": str(sample["op_segment"]),
        "structured_prediction": str(sample.get("structured_prediction", "")),
    }
    if chinese:
        user_prompt = (
            f"Rubric 版本：{config['rubric_version']}\n\n"
            f"评估原则：\n{principles}\n\n"
            f"输入字段：\n{json.dumps(input_payload, ensure_ascii=False, indent=2)}\n\n"
            f"Rubric 问题：\n{_render_questions(config)}\n\n"
            "必须严格按照以下结构输出：\n"
            f"{json.dumps(_output_template(config), ensure_ascii=False, indent=2)}\n\n"
            f"请独立回答全部 {len(config['questions'])} 个问题。若某题不强制证据且没有有用的逐字证据，请使用空 evidence 列表。"
            "不要添加 final_label，也不要在 JSON 之外输出任何内容。"
        )
    else:
        user_prompt = (
            f"Rubric version: {config['rubric_version']}\n\n"
            f"Principles:\n{principles}\n\n"
            f"Input fields:\n{json.dumps(input_payload, ensure_ascii=False, indent=2)}\n\n"
            f"Rubric questions:\n{_render_questions(config)}\n\n"
            "Required output structure:\n"
            f"{json.dumps(_output_template(config), ensure_ascii=False, indent=2)}\n\n"
            f"Answer all {len(config['questions'])} items independently. Use an empty evidence list when evidence is not required and no useful exact quote exists. Do not add a final_label field."
        )
    return [
        {"role": "system", "content": config.get("system_prompt", SYSTEM_PROMPT)},
        {"role": "user", "content": user_prompt},
    ]
