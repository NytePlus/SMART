from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rubric_eval.aggregation import aggregate_rubric_result
from rubric_eval.config import load_rubric
from rubric_eval.evaluation import evaluate_records
from rubric_eval.prompting import build_messages
from rubric_eval.response_parser import parse_json_response
from rubric_eval.schema import validate_dataset, validate_input_record, validate_rubric_judgment
from scripts.build_rubric_viewer import build_html


CONFIG = load_rubric(ROOT / "config" / "rubric_v1.json")
MINIMAL_CONFIG = CONFIG
THEORY_CONFIG = CONFIG


def sample() -> dict:
    return {
        "sample_id": "test:1:0",
        "dataset": "test",
        "id": "1",
        "history": "Earlier context.",
        "suffix": "Later context.",
        "target": "The model did not improve accuracy.",
        "pred": "The model improved accuracy.",
        "structured_prediction": "The model <focus><del>did not </del>improve</focus> accuracy.",
        "reference_segment": "did not improve",
        "hypothesis_segment": "improved",
        "op_segment": "+++++++========",
        "human_annotation": {"is_semantic_error": True},
    }


def item(answer: str, reason_code: str, evidence: list[dict] | None = None) -> dict:
    return {
        "answer": answer,
        "reason_code": reason_code,
        "confidence": 0.9,
        "evidence": evidence or [],
    }


def judgment(
    *,
    g1: str = "VALID",
    g2: str = "SCORABLE",
    s1: str = "SEMANTIC_CHANGE",
    s2: str = "NO_CHANGE",
    s3: str = "SEMANTIC_CHANGE",
    s4: str = "NO_CHANGE",
    c1: str = "NOT_RECOVERABLE",
    c2: str = "MATERIAL",
) -> dict:
    return {
        "rubric_version": CONFIG["rubric_version"],
        "items": {
            "G1": item(g1, {
                "VALID": "ALIGNED_LOCAL_UNIT",
                "PARTIALLY_VALID": "BOUNDARY_OVERLAP",
                "INVALID": "MISMATCHED_SPANS",
                "UNSCORABLE": "INSUFFICIENT_ALIGNMENT_EVIDENCE",
            }[g1], [{"source": "reference_span", "quote": "did not improve"}] if g1 != "UNSCORABLE" else []),
            "G2": item(g2, "SUFFICIENT_LOCAL_EVIDENCE" if g2 == "SCORABLE" else "CONTEXT_MISSING"),
            "S1": item(s1, {
                "SEMANTIC_CHANGE": "PREDICATE_CHANGED",
                "NO_CHANGE": "PROPOSITION_PRESERVED",
                "MINOR_FORM_CHANGE": "FORM_ONLY",
                "UNCERTAIN": "UNCERTAIN_EVENT",
                "N_A": "NOT_APPLICABLE",
            }[s1], [{"source": "reference_span", "quote": "did not improve"}] if s1 == "SEMANTIC_CHANGE" else []),
            "S2": item(s2, {
                "SEMANTIC_CHANGE": "ENTITY_CHANGED",
                "NO_CHANGE": "ENTITY_PRESERVED",
                "MINOR_FORM_CHANGE": "SPELLING_VARIANT",
                "UNCERTAIN": "UNCERTAIN_ENTITY",
                "N_A": "NOT_APPLICABLE",
            }[s2], [{"source": "reference_span", "quote": "did not improve"}] if s2 == "SEMANTIC_CHANGE" else []),
            "S3": item(s3, {
                "SEMANTIC_CHANGE": "NEGATION_CHANGED",
                "NO_CHANGE": "LOGIC_PRESERVED",
                "MINOR_FORM_CHANGE": "FORM_ONLY",
                "UNCERTAIN": "UNCERTAIN_LOGIC",
                "N_A": "NOT_APPLICABLE",
            }[s3], [{"source": "reference_text", "quote": "not"}] if s3 == "SEMANTIC_CHANGE" else []),
            "S4": item(s4, {
                "SEMANTIC_CHANGE": "AGENT_CHANGED",
                "NO_CHANGE": "RELATION_PRESERVED",
                "MINOR_FORM_CHANGE": "FORM_ONLY",
                "UNCERTAIN": "UNCERTAIN_RELATION",
                "N_A": "NOT_APPLICABLE",
            }[s4], [{"source": "reference_span", "quote": "did not improve"}] if s4 == "SEMANTIC_CHANGE" else []),
            "C1": item(c1, {
                "FULLY_RECOVERABLE": "UNIQUE_CONTEXTUAL_RECOVERY",
                "PARTIALLY_RECOVERABLE": "PARTIAL_MEANING_ONLY",
                "NOT_RECOVERABLE": "WRONG_INTERPRETATION_SUPPORTED",
                "UNSCORABLE": "INSUFFICIENT_CONTEXT",
            }[c1], [{"source": "hypothesis_text", "quote": "improved accuracy"}] if c1 != "UNSCORABLE" else []),
            "C2": item(c2, {
                "NONE": "NO_INTERPRETATION_CHANGE",
                "MINOR": "MINOR_LOCAL_DEVIATION",
                "MATERIAL": "LOGIC_MISUNDERSTOOD",
                "CRITICAL": "MAIN_MEANING_REVERSED",
                "UNSCORABLE": "INSUFFICIENT_EVIDENCE",
            }[c2], [{"source": "hypothesis_text", "quote": "improved accuracy"}] if c2 in {"MATERIAL", "CRITICAL"} else []),
        },
    }


def minimal_judgment(
    *,
    q0: str = "可评分",
    q1: str = "内容变化",
    q2: str = "不可恢复",
    q3: str = "实质影响",
) -> dict:
    reason_codes = {
        "Q0": {
            "可评分": "局部对齐有效",
            "不可评分": "参考文本有问题",
        },
        "Q1": {
            "无内容变化": "内容保持一致",
            "仅形式变化": "拼写或空格差异",
            "内容变化": "逻辑条件改变",
            "不确定": "内容变化无法确定",
        },
        "Q2": {
            "完全可恢复": "上下文唯一确定原意",
            "部分可恢复": "关键细节仍不确定",
            "不可恢复": "错误解释更自然",
            "不确定": "上下文证据不足",
        },
        "Q3": {
            "无影响": "最终理解一致",
            "轻微影响": "非关键信息轻微偏差",
            "实质影响": "逻辑条件可能误解",
            "不确定": "影响程度无法确定",
        },
    }
    evidence = {
        "Q0": [],
        "Q1": [{"source": "reference_text", "quote": "not"}] if q1 == "内容变化" else [],
        "Q2": (
            [{"source": "hypothesis_text", "quote": "improved accuracy"}]
            if q2 != "不确定"
            else []
        ),
        "Q3": (
            [{"source": "hypothesis_text", "quote": "improved accuracy"}]
            if q3 == "实质影响"
            else []
        ),
    }
    answers = {"Q0": q0, "Q1": q1, "Q2": q2, "Q3": q3}
    return {
        "rubric_version": MINIMAL_CONFIG["rubric_version"],
        "items": {
            question_id: item(answer, reason_codes[question_id][answer], evidence[question_id])
            for question_id, answer in answers.items()
        },
        "change_types": ["否定或模态"],
    }


def theory_judgment(
    *,
    g1: str = "有效",
    g2: str = "可评分",
    s1: str = "不适用",
    s2: str = "语义改变",
    s3: str = "不适用",
    s4: str = "语义改变",
    c1: str = "不可恢复",
    c2: str = "实质影响",
) -> dict:
    reason_codes = {
        "G1": {
            "有效": "同一局部编辑",
            "部分有效": "边界有偏差但可比较",
            "无效": "配对位置错误",
            "不可评分": "对齐证据不足",
        },
        "G2": {
            "可评分": "局部证据充分",
            "不可评分": "参考文本有问题",
        },
        "S1": {
            "保持不变": "指称保持",
            "仅形式差异": "同一对象的拼写变体",
            "语义改变": "指称对象改变",
            "不确定": "指称词义无法确定",
            "不适用": "该维度不适用",
        },
        "S2": {
            "保持不变": "事件框架保持",
            "仅形式差异": "同义表达",
            "语义改变": "状态改变",
            "不确定": "事件命题无法确定",
            "不适用": "该维度不适用",
        },
        "S3": {
            "保持不变": "参与者角色保持",
            "仅形式差异": "可恢复的代词形式",
            "语义改变": "施事改变",
            "不确定": "角色关系无法确定",
            "不适用": "该维度不适用",
        },
        "S4": {
            "保持不变": "逻辑结构保持",
            "仅形式差异": "不改变逻辑的形式变体",
            "语义改变": "否定改变",
            "不确定": "逻辑识解无法确定",
            "不适用": "该维度不适用",
        },
        "C1": {
            "完全可恢复": "上下文唯一确定原意",
            "部分可恢复": "关键细节仍缺失",
            "不可恢复": "错误解释更自然",
            "不可评分": "上下文证据不足",
        },
        "C2": {
            "无影响": "最终理解一致",
            "轻微影响": "非关键信息局部偏差",
            "实质影响": "逻辑条件可能误解",
            "关键影响": "主要结论反转",
            "不可评分": "影响程度无法确定",
        },
    }
    answers = {
        "G1": g1,
        "G2": g2,
        "S1": s1,
        "S2": s2,
        "S3": s3,
        "S4": s4,
        "C1": c1,
        "C2": c2,
    }
    evidence: dict[str, list[dict]] = {question_id: [] for question_id in answers}
    if g1 == "无效":
        evidence["G1"] = [{"source": "reference_span", "quote": "did not improve"}]
    if g2 == "不可评分":
        evidence["G2"] = [{"source": "reference_text", "quote": "did not improve"}]
    for question_id, answer in (("S1", s1), ("S2", s2), ("S3", s3), ("S4", s4)):
        if answer == "语义改变":
            evidence[question_id] = [
                {"source": "reference_span", "quote": "did not improve"},
                {"source": "hypothesis_span", "quote": "improved"},
            ]
    if c1 != "不可评分":
        evidence["C1"] = [{"source": "hypothesis_text", "quote": "improved accuracy"}]
    if c2 in {"实质影响", "关键影响"}:
        evidence["C2"] = [{"source": "hypothesis_text", "quote": "improved accuracy"}]
    return {
        "rubric_version": THEORY_CONFIG["rubric_version"],
        "items": {
            question_id: item(answer, reason_codes[question_id][answer], evidence[question_id])
            for question_id, answer in answers.items()
        },
        "change_types": ["否定"],
    }


class DatasetValidationTests(unittest.TestCase):
    def test_valid_input(self) -> None:
        self.assertEqual(validate_input_record(sample()), [])

    def test_missing_required_field(self) -> None:
        value = sample()
        del value["target"]
        self.assertIn("missing required field: target", validate_input_record(value))

    def test_duplicate_sample_id(self) -> None:
        issues = validate_dataset([sample(), sample()])
        self.assertEqual(len(issues), 1)
        self.assertIn("duplicate sample_id", issues[0]["errors"])


class ResponseValidationTests(unittest.TestCase):
    def test_valid_judgment(self) -> None:
        self.assertEqual(validate_rubric_judgment(judgment(), sample(), CONFIG), [])

    def test_invalid_enum(self) -> None:
        value = judgment()
        value["items"]["S1"]["answer"] = "MAYBE"
        errors = validate_rubric_judgment(value, sample(), CONFIG)
        self.assertTrue(any("S1: invalid answer" in error for error in errors))

    def test_fabricated_evidence_is_rejected(self) -> None:
        value = judgment()
        value["items"]["S3"]["evidence"][0]["quote"] = "never"
        errors = validate_rubric_judgment(value, sample(), CONFIG)
        self.assertTrue(any("quote is not present" in error for error in errors))

    def test_recoverability_cannot_cite_reference(self) -> None:
        value = judgment()
        value["items"]["C1"]["evidence"] = [{"source": "reference_text", "quote": "not"}]
        errors = validate_rubric_judgment(value, sample(), CONFIG)
        self.assertTrue(any("not allowed for C1" in error for error in errors))

    def test_json_code_fence(self) -> None:
        self.assertEqual(parse_json_response('```json\n{"a": 1}\n```'), {"a": 1})

    def test_invalid_json(self) -> None:
        with self.assertRaises(ValueError):
            parse_json_response("not json")


class AggregationTests(unittest.TestCase):
    def test_consequential_rule(self) -> None:
        result = aggregate_rubric_result(judgment(), CONFIG)
        self.assertEqual(result["final_label"], "CONSEQUENTIAL")
        self.assertEqual(result["aggregation_rule"], "R1_CONSEQUENTIAL")

    def test_inconsequential_no_change(self) -> None:
        value = judgment(s1="NO_CHANGE", s3="NO_CHANGE", c1="FULLY_RECOVERABLE", c2="NONE")
        result = aggregate_rubric_result(value, CONFIG)
        self.assertEqual(result["final_label"], "INCONSEQUENTIAL")

    def test_fully_recoverable_semantic_change_with_no_impact(self) -> None:
        value = judgment(c1="FULLY_RECOVERABLE", c2="NONE")
        result = aggregate_rubric_result(value, CONFIG)
        self.assertEqual(result["final_label"], "INCONSEQUENTIAL")

    def test_invalid_alignment_stops_scoring(self) -> None:
        result = aggregate_rubric_result(judgment(g1="INVALID"), CONFIG)
        self.assertEqual(result["final_label"], "UNSCORABLE")
        self.assertEqual(result["eval_status"], "ALIGNMENT_ERROR")

    def test_unresolved_semantic_uncertainty(self) -> None:
        value = judgment(s1="UNCERTAIN", s3="NO_CHANGE", c1="NOT_RECOVERABLE", c2="MATERIAL")
        result = aggregate_rubric_result(value, CONFIG)
        self.assertEqual(result["aggregation_rule"], "U_SEMANTIC_UNCERTAIN")

class EvaluationTests(unittest.TestCase):
    def test_end_to_end_metrics(self) -> None:
        positive = {
            **sample(),
            "inference_status": "SUCCESS",
            "aggregated_result": {"final_label": "CONSEQUENTIAL", "aggregation_rule": "R1"},
        }
        negative = {
            **sample(),
            "sample_id": "test:2:0",
            "human_annotation": {"is_semantic_error": False},
            "inference_status": "SUCCESS",
            "aggregated_result": {"final_label": "INCONSEQUENTIAL", "aggregation_rule": "R2"},
        }
        report = evaluate_records([positive, negative])
        self.assertEqual(report["scorable_metrics"]["accuracy"], 1.0)
        self.assertEqual(report["scorable_coverage"], 1.0)


@unittest.skip("legacy minimal rubric config is not included in this distribution")
class MinimalChineseRubricTests(unittest.TestCase):
    def test_minimal_judgment_is_valid(self) -> None:
        self.assertEqual(
            validate_rubric_judgment(minimal_judgment(), sample(), MINIMAL_CONFIG),
            [],
        )

    def test_reason_code_must_match_answer(self) -> None:
        value = minimal_judgment()
        value["items"]["Q1"]["reason_code"] = "拼写或空格差异"
        errors = validate_rubric_judgment(value, sample(), MINIMAL_CONFIG)
        self.assertTrue(any("is not valid for answer" in error for error in errors))

    def test_invalid_change_type_is_rejected(self) -> None:
        value = minimal_judgment()
        value["change_types"] = ["不存在的类型"]
        errors = validate_rubric_judgment(value, sample(), MINIMAL_CONFIG)
        self.assertTrue(any("invalid change_types" in error for error in errors))

    def test_minimal_consequential_aggregation(self) -> None:
        result = aggregate_rubric_result(minimal_judgment(), MINIMAL_CONFIG)
        self.assertEqual(result["final_label"], "CONSEQUENTIAL")
        self.assertEqual(result["aggregation_rule"], "R1_实质语义后果")

    def test_minimal_inconsequential_aggregation(self) -> None:
        value = minimal_judgment(q1="仅形式变化", q2="完全可恢复", q3="无影响")
        result = aggregate_rubric_result(value, MINIMAL_CONFIG)
        self.assertEqual(result["final_label"], "INCONSEQUENTIAL")

    def test_minimal_uncertain_is_unscorable(self) -> None:
        value = minimal_judgment(q1="不确定", q2="不确定", q3="不确定")
        result = aggregate_rubric_result(value, MINIMAL_CONFIG)
        self.assertEqual(result["final_label"], "UNSCORABLE")

    def test_recoverability_rejects_reference_evidence(self) -> None:
        value = minimal_judgment()
        value["items"]["Q2"]["evidence"] = [
            {"source": "reference_text", "quote": "not"}
        ]
        errors = validate_rubric_judgment(value, sample(), MINIMAL_CONFIG)
        self.assertTrue(any("not allowed for Q2" in error for error in errors))

    def test_prompt_is_chinese_and_hides_human_label(self) -> None:
        messages = build_messages(sample(), MINIMAL_CONFIG)
        prompt = "\n".join(message["content"] for message in messages)
        self.assertIn("上下文可恢复性", prompt)
        self.assertIn("不要输出 final_label", prompt)
        self.assertNotIn("human_annotation", prompt)


@unittest.skip("legacy theory rubric config is not included in this distribution")
class TheoryGroundedChineseRubricTests(unittest.TestCase):
    def test_theory_config_and_judgment_are_valid(self) -> None:
        self.assertEqual(THEORY_CONFIG["rubric_version"], "semantic-theory-v1.0.0")
        self.assertEqual(len(THEORY_CONFIG["questions"]), 8)
        self.assertEqual(
            validate_rubric_judgment(theory_judgment(), sample(), THEORY_CONFIG),
            [],
        )

    def test_semantic_change_requires_both_sides_of_evidence(self) -> None:
        value = theory_judgment()
        value["items"]["S4"]["evidence"] = [
            {"source": "reference_span", "quote": "did not improve"}
        ]
        errors = validate_rubric_judgment(value, sample(), THEORY_CONFIG)
        self.assertTrue(any("hypothesis_span" in error for error in errors))

    def test_empty_span_is_valid_evidence_for_deletion(self) -> None:
        deletion_sample = sample()
        deletion_sample["pred"] = "The model accuracy."
        deletion_sample["hypothesis_segment"] = ""
        value = theory_judgment()
        for question_id in ("S2", "S4"):
            value["items"][question_id]["evidence"] = [
                {"source": "reference_span", "quote": "did not improve"},
                {"source": "hypothesis_span", "quote": ""},
            ]
        value["items"]["C1"]["evidence"] = [
            {"source": "hypothesis_text", "quote": "model accuracy"}
        ]
        value["items"]["C2"]["evidence"] = [
            {"source": "hypothesis_text", "quote": "model accuracy"}
        ]
        self.assertEqual(
            validate_rubric_judgment(value, deletion_sample, THEORY_CONFIG),
            [],
        )

    def test_empty_quote_is_rejected_for_nonempty_span(self) -> None:
        value = theory_judgment()
        value["items"]["S4"]["evidence"][1]["quote"] = ""
        errors = validate_rubric_judgment(value, sample(), THEORY_CONFIG)
        self.assertTrue(any("may be empty only" in error for error in errors))

    def test_recoverability_rejects_reference_leakage(self) -> None:
        value = theory_judgment()
        value["items"]["C1"]["evidence"] = [
            {"source": "reference_text", "quote": "not"}
        ]
        errors = validate_rubric_judgment(value, sample(), THEORY_CONFIG)
        self.assertTrue(any("not allowed for C1" in error for error in errors))

    def test_theory_consequential_aggregation(self) -> None:
        result = aggregate_rubric_result(theory_judgment(), THEORY_CONFIG)
        self.assertEqual(result["final_label"], "CONSEQUENTIAL")
        self.assertEqual(result["aggregation_rule"], "R1_实质语义后果")
        self.assertEqual(result["semantic_change_items"], ["S2", "S4"])

    def test_fully_recovered_change_is_inconsequential(self) -> None:
        value = theory_judgment(c1="完全可恢复", c2="无影响")
        result = aggregate_rubric_result(value, THEORY_CONFIG)
        self.assertEqual(result["final_label"], "INCONSEQUENTIAL")
        self.assertEqual(result["aggregation_rule"], "R3_无实质理解偏差")

    def test_invalid_alignment_stops_theory_scoring(self) -> None:
        result = aggregate_rubric_result(theory_judgment(g1="无效"), THEORY_CONFIG)
        self.assertEqual(result["final_label"], "UNSCORABLE")
        self.assertEqual(result["eval_status"], "ALIGNMENT_ERROR")

    def test_theory_prompt_is_chinese_and_hides_human_label(self) -> None:
        messages = build_messages(sample(), THEORY_CONFIG)
        prompt = "\n".join(message["content"] for message in messages)
        self.assertIn("指称与词义", prompt)
        self.assertIn("分侧证据要求", prompt)
        self.assertIn("不要输出 final_label", prompt)
        self.assertNotIn("human_annotation", prompt)


@unittest.skip("legacy rubric variants are not included in this distribution")
class RubricViewerTests(unittest.TestCase):
    def test_viewer_contains_questions_rules_and_config_hash(self) -> None:
        viewer = build_html(THEORY_CONFIG)
        self.assertEqual(viewer.count('<details class="question"'), 8)
        self.assertIn("确定性聚合规则", viewer)
        self.assertIn("R1_实质语义后果", viewer)
        self.assertIn(THEORY_CONFIG["config_hash"], viewer)

    def test_generated_viewer_matches_current_config(self) -> None:
        generated = (ROOT / "viewer" / "index.html").read_text(encoding="utf-8")
        self.assertIn(THEORY_CONFIG["config_hash"], generated)


if __name__ == "__main__":
    unittest.main()
