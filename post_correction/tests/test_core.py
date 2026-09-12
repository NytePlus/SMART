import tempfile
import unittest
import json
import io
from contextlib import redirect_stdout
from pathlib import Path

from asr_post_correction.core import CorrectionEngine
from asr_post_correction.evaluation import validate_llm_only_output
from asr_post_correction.segmenter import split_segments
from asr_post_correction.io import add_auxiliary_context, build_contexts, systematic_indices
from asr_post_correction.cli import command_normalize
from asr_post_correction.audio_evidence import resolve_audio_path
from argparse import Namespace


PROMPTS = {"version": "test", "detector_system": "d", "detector_user": "{{PAYLOAD}}",
           "corrector_system": "c", "corrector_user": "{{PAYLOAD}}"}


def candidate(span):
    return {"candidates": [{"segment_id": 0, "span": span, "start_in_segment": 0,
                            "end_in_segment": len(span), "confidence": .99,
                            "rubric": {"G1": "有效", "G2": "充分", "S1": "可疑",
                                       "S2": "正常", "S3": "正常", "S4": "正常",
                                       "C1": "部分可理解", "C2": "实质"},
                            "reason": "test", "evidence": [{"source": "hypothesis", "quote": span}]}]}


def suggestion(original, replacement):
    return {"suggestions": [{"segment_id": 0, "original": original, "replacement": replacement,
                              "confidence": .99, "rationale": "test",
                              "evidence": [{"source": "hypothesis", "quote": original}]}]}


class SequenceClient:
    def __init__(self, values): self.values = iter(values)
    def complete_json(self, messages, *, purpose): return next(self.values), {"purpose": purpose}


class CoreTests(unittest.TestCase):
    def test_segmenter_preserves_input(self):
        text = "第一句，第二句。third sentence here!"
        segments = split_segments(text, max_chars=10)
        self.assertEqual("".join(s.text for s in segments), text)
        self.assertEqual([(s.start, s.end) for s in segments],
                         [(s.start, s.start + len(s.text)) for s in segments])

    def test_single_round_applies_exact_edit(self):
        client = SequenceClient([candidate("错词"), suggestion("错词", "正词")])
        result = CorrectionEngine(client, PROMPTS).single_round("错词在这里")
        self.assertEqual(result.corrected_hypothesis, "正词在这里")

    def test_duplicate_original_is_rejected(self):
        engine = CorrectionEngine(SequenceClient([]), PROMPTS)
        segments = split_segments("错错")
        from asr_post_correction.models import Suggestion
        output, applied, rejected = engine.apply("错错", segments, [Suggestion(0, "错", "正", .99, "", [])])
        self.assertEqual(output, "错错")
        self.assertFalse(applied)
        self.assertEqual(rejected[0]["reason"], "original_not_unique_in_segment")

    def test_duplicate_original_can_be_addressed_by_exact_offset(self):
        engine = CorrectionEngine(SequenceClient([]), PROMPTS)
        segments = split_segments("错错")
        from asr_post_correction.models import Suggestion
        edit = Suggestion(0, "错", "正", .99, "", [], 1, 2)
        output, applied, rejected = engine.apply("错错", segments, [edit])
        self.assertEqual(output, "错正")
        self.assertEqual(len(applied), 1)
        self.assertFalse(rejected)

    def test_multi_round_detects_a_b_a_cycle_without_accepting_reversal(self):
        client = SequenceClient([candidate("A"), suggestion("A", "B"),
                                 candidate("B"), suggestion("B", "A")])
        result = CorrectionEngine(client, PROMPTS).multi_round("A", max_rounds=5)
        self.assertEqual(result.corrected_hypothesis, "B")
        self.assertEqual(result.stop_reason, "two_cycle_detected")
        self.assertEqual(result.seen_hypotheses, ["A", "B"])

    def test_segment_candidate_can_authorize_exact_subspan_suggestion(self):
        prompts = {**PROMPTS, "allow_subspan_suggestions": True}
        detector = candidate("错词在这里")
        corrector = {"suggestions": [{"segment_id": 0, "original": "错词",
                                       "start_in_segment": 0, "end_in_segment": 2,
                                       "replacement": "正词", "confidence": .99,
                                       "rationale": "test",
                                       "evidence": [{"source": "hypothesis", "quote": "错词"}]}]}
        result = CorrectionEngine(SequenceClient([detector, corrector]), prompts).single_round("错词在这里")
        self.assertEqual(result.corrected_hypothesis, "正词在这里")

    def test_independent_verifier_can_reject_proposal(self):
        prompts = {**PROMPTS, "verifier_system": "v", "verifier_user": "{{PAYLOAD}}"}
        client = SequenceClient([candidate("错词"), suggestion("错词", "正词"),
                                 {"decisions": [{"suggestion_index": 0, "accept": False,
                                                  "confidence": .99, "evidence": []}]}])
        result = CorrectionEngine(client, prompts).single_round("错词在这里")
        self.assertEqual(len(result.rounds[0].suggestions), 1)
        self.assertEqual(result.rounds[0].verified_suggestions, [])
        self.assertFalse(result.rounds[0].verification_decisions[0]["accepted"])
        self.assertEqual(result.corrected_hypothesis, "错词在这里")

    def test_detector_passes_merge_duplicate_candidates(self):
        prompts = {**PROMPTS, "detector_passes": [
            {"name": "context", "system": "d2", "user": "{{PAYLOAD}}"}
        ]}
        first = candidate("错词")
        second = candidate("错词")
        second["candidates"][0]["confidence"] = .995
        second["candidates"][0]["reason"] = "second pass"
        client = SequenceClient([first, second, suggestion("错词", "正词")])
        result = CorrectionEngine(client, prompts).single_round("错词")
        self.assertEqual(len(result.rounds[0].candidates), 1)
        self.assertEqual(result.rounds[0].candidates[0].confidence, .995)
        self.assertEqual(result.rounds[0].candidates[0].reason, "test\nsecond pass")
        self.assertEqual(result.corrected_hypothesis, "正词")

    def test_corrector_passes_merge_duplicate_suggestions(self):
        prompts = {**PROMPTS, "corrector_passes": [
            {"name": "coverage", "system": "c2", "user": "{{PAYLOAD}}"}
        ]}
        detector = candidate("错词")
        first = suggestion("错词", "正词")
        second = suggestion("错词", "正词")
        second["suggestions"][0]["confidence"] = .995
        client = SequenceClient([detector, first, second])
        result = CorrectionEngine(client, prompts).single_round("错词")
        self.assertEqual(len(result.rounds[0].suggestions), 1)
        self.assertEqual(result.rounds[0].suggestions[0].confidence, .995)

    def test_split_ascii_letter_postprocessor_is_explicit_and_audited(self):
        prompts = {**PROMPTS, "deterministic_postprocessors": ["merge_split_ascii_letters"]}
        result = CorrectionEngine(SequenceClient([{"candidates": []}]), prompts).single_round("H O一班")
        self.assertEqual(result.corrected_hypothesis, "HO一班")
        self.assertEqual(result.stop_reason, "single_round_complete")
        self.assertEqual(result.rounds[0].applied_edits[0]["source"], "deterministic_postprocessor")

    def test_chinese_disfluency_postprocessor_is_explicit_and_audited(self):
        prompts = {**PROMPTS, "deterministic_postprocessors": ["drop_chinese_disfluency_e"]}
        result = CorrectionEngine(SequenceClient([{"candidates": []}]), prompts).single_round("在呃这里")
        self.assertEqual(result.corrected_hypothesis, "在这里")
        self.assertEqual(result.rounds[0].applied_edits[0]["postprocessor"], "drop_chinese_disfluency_e")

    def test_internal_apostrophe_postprocessor_splits_only_letter_boundaries(self):
        prompts = {**PROMPTS, "deterministic_postprocessors": ["split_internal_apostrophes"]}
        result = CorrectionEngine(SequenceClient([{"candidates": []}]), prompts).single_round(
            "I'd we're cats' rock'n'roll"
        )
        self.assertEqual(result.corrected_hypothesis, "I d we re cats' rock n roll")
        self.assertEqual(
            [edit["postprocessor"] for edit in result.rounds[0].applied_edits],
            ["split_internal_apostrophes"] * 4,
        )

    def test_llm_only_engine_rejects_normalization_only_suggestion(self):
        client = SequenceClient([candidate("I'd"), suggestion("I'd", "I d")])
        result = CorrectionEngine(
            client, PROMPTS, forbid_normalization_edits=True
        ).single_round("I'd")
        self.assertEqual(result.corrected_hypothesis, "I'd")
        self.assertEqual(result.rounds[0].rejected_edits[0]["reason"], "normalization_only_edit")

    def test_normalize_reports_changed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            output = root / "output.jsonl"
            prompt = root / "prompt.json"
            source.write_text(json.dumps({"id": "1", "pred": "H O"}, ensure_ascii=False) + "\n"
                              + json.dumps({"id": "2", "pred": "正常"}, ensure_ascii=False) + "\n",
                              encoding="utf-8")
            prompt.write_text(json.dumps({"version": "normalize-test", "detector_system": "",
                                          "detector_user": "", "corrector_system": "",
                                          "corrector_user": "", "deterministic_postprocessors":
                                          ["merge_split_ascii_letters"]}), encoding="utf-8")
            captured = io.StringIO()
            with redirect_stdout(captured):
                command_normalize(Namespace(input=str(source), output=str(output), prompt=str(prompt)))
            summary = json.loads(captured.getvalue())
            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["changed"], 1)

    def test_llm_only_audit_rejects_deterministic_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.jsonl"
            output.write_text(json.dumps({
                "id": "1", "status": "ok", "deterministic_edits": [{"original": "a"}],
                "rounds": [],
            }) + "\n", encoding="utf-8")
            output.with_suffix(".manifest.json").write_text(json.dumps({
                "llm_only": True, "normalization_only_edits_forbidden": True,
                "deterministic_postprocessors": [],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "deterministic edits"):
                validate_llm_only_output(output)

    def test_empty_candidates_stop(self):
        result = CorrectionEngine(SequenceClient([{"candidates": []}]), PROMPTS).multi_round("ok")
        self.assertEqual(result.stop_reason, "no_candidates")

    def test_systematic_sample_is_content_blind_and_spans_file(self):
        self.assertEqual(systematic_indices(10, 3), [0, 4, 9])

    def test_prefixed_prediction_id_receives_history_ocr_and_alternate(self):
        predictions = [
            {"id": "dataset:1", "source_id": "1", "pred": "first"},
            {"id": "dataset:2", "source_id": "2", "pred": "second"},
        ]
        messages = [
            {"id": "1", "session": "s"},
            {"id": "2", "session": "s"},
        ]
        contexts = build_contexts(predictions, messages, history_size=2)
        add_auxiliary_context(
            contexts,
            predictions,
            alternate_rows=[{"id": "2", "pred": "alternate"}],
            ocr_rows=[{"id": "2", "ocr": {"text": "slide term"}}],
        )
        self.assertEqual(contexts["dataset:2"]["history"], "first")
        self.assertEqual(contexts["dataset:2"]["metadata"]["session"], "s")
        self.assertEqual(contexts["dataset:2"]["alternate_hypothesis"], "alternate")
        self.assertEqual(contexts["dataset:2"]["slide_ocr"], "slide term")

    def test_slide_ocr_is_valid_extractive_evidence(self):
        prompts = {**PROMPTS, "allow_subspan_suggestions": True}
        detector = candidate("刀米")
        corrector = {"suggestions": [{
            "segment_id": 0,
            "original": "刀米",
            "start_in_segment": 0,
            "end_in_segment": 2,
            "replacement": "dominant",
            "confidence": .99,
            "rationale": "slide evidence",
            "evidence": [{"source": "slide_ocr", "quote": "dominant"}],
        }]}
        result = CorrectionEngine(SequenceClient([detector, corrector]), prompts).single_round(
            "刀米", {"slide_ocr": "pattern: dominant"}
        )
        self.assertEqual(result.corrected_hypothesis, "dominant")

    def test_multiple_alternate_hypotheses_remain_independent(self):
        predictions = [{"id": "1", "pred": "primary"}]
        contexts = build_contexts(predictions, [], history_size=1)
        add_auxiliary_context(
            contexts,
            predictions,
            alternate_rows=[
                {"id": "1", "pred": "alternative one"},
                {"id": "1", "pred": "alternative two"},
            ],
        )
        self.assertEqual(
            contexts["1"]["alternate_hypothesis"],
            "ASR1: alternative one\n\nASR2: alternative two",
        )

    def test_alternate_evidence_keeps_source_labels_and_content_only_disagreements(self):
        predictions = [{"id": "1", "pred": "the wrong term"}]
        contexts = build_contexts(predictions, [], history_size=1)
        add_auxiliary_context(
            contexts,
            predictions,
            alternate_rows=[
                {"id": "1", "model": "model-a", "pred": "the right term"},
            ],
        )
        context = contexts["1"]
        self.assertEqual(context["alternate_hypotheses"], [
            {"source": "model-a", "text": "the right term"},
        ])
        self.assertEqual(context["alternate_disagreements"][0]["source"], "model-a")
        difference = context["alternate_disagreements"][0]["differences"][0]
        self.assertEqual(difference["operation"], "replace")
        self.assertEqual(difference["hypothesis_tokens"], ["wrong"])
        self.assertEqual(difference["alternate_tokens"], ["right"])
        self.assertEqual(
            difference["hypothesis_context"], ["the", "wrong", "term"]
        )
        self.assertEqual(
            difference["alternate_context"], ["the", "right", "term"]
        )
        self.assertEqual(
            (difference["hypothesis_char_start"], difference["hypothesis_char_end"]),
            (4, 9),
        )
        self.assertEqual(
            (difference["alternate_char_start"], difference["alternate_char_end"]),
            (4, 9),
        )
        self.assertEqual(difference["hypothesis_context_text"], "the wrong term")
        self.assertEqual(difference["alternate_context_text"], "the right term")

    def test_identical_text_from_independent_alternates_is_not_collapsed(self):
        predictions = [{"id": "1", "pred": "primary"}]
        contexts = build_contexts(predictions, [], history_size=1)
        add_auxiliary_context(
            contexts,
            predictions,
            alternate_rows=[
                {"id": "1", "model": "model-a", "pred": "same text"},
                {"id": "1", "model": "model-b", "pred": "same text"},
                {"id": "1", "model": "model-a", "pred": "same text"},
            ],
        )
        self.assertEqual(
            contexts["1"]["alternate_hypotheses"],
            [
                {"source": "model-a", "text": "same text"},
                {"source": "model-b", "text": "same text"},
            ],
        )

    def test_alternate_disagreement_is_annotated_with_segment_ids(self):
        engine = CorrectionEngine(SequenceClient([]), PROMPTS)
        segments = split_segments("the wrong term")
        context = {
            "alternate_disagreements": [{
                "source": "model-a",
                "differences": [{
                    "operation": "replace",
                    "hypothesis_char_start": 4,
                    "hypothesis_char_end": 9,
                    "alternate_char_start": 4,
                    "alternate_char_end": 9,
                }],
            }],
        }
        annotated = engine._annotated_alternate_disagreements(context, segments)
        self.assertEqual(annotated[0]["differences"][0]["segment_ids"], [0])

    def test_audio_path_uses_dataset_local_source_id(self):
        row = {"id": "dataset:2", "source_id": "2"}
        messages = {"2": {"audio": "audios/0002.wav"}}
        self.assertEqual(
            resolve_audio_path(row, messages, "/data"),
            Path("/data/audios/0002.wav"),
        )


if __name__ == "__main__":
    unittest.main()
