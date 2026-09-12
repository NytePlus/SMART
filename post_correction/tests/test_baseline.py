import unittest

from asr_post_correction.baseline import (
    apply_baseline_edits,
    render_baseline_messages,
    render_split_baseline_messages,
    restrict_edits_to_candidates,
    validate_rubric_candidates,
)


class BaselineTest(unittest.TestCase):
    def test_payload_exposes_history_but_not_target(self):
        messages = render_baseline_messages(
            {"system": "system", "user": "{{PAYLOAD}}"}, "参考上文", "待修正句"
        )
        self.assertIn("参考上文", messages[1]["content"])
        self.assertIn("待修正句", messages[1]["content"])
        self.assertNotIn("target", messages[1]["content"])
        self.assertNotIn("context_ground_truth_suffix", messages[1]["content"])

    def test_v9_prompt_contains_six_rubric_dimensions(self):
        import json
        from pathlib import Path
        prompt = json.loads(
            (Path(__file__).resolve().parents[1] / "prompts" / "gt_context_baseline_v9.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(all(label in prompt["user"] for label in ("S1", "S2", "S3", "S4", "C1", "C2")))
        self.assertNotIn("ground_truth transcript", prompt["system"])
        self.assertNotIn("context_ground_truth_suffix", prompt["user"])

    def test_split_v1_prompt_preserves_rubric_and_payload_contract(self):
        import json
        from pathlib import Path
        prompt = json.loads(
            (Path(__file__).resolve().parents[1] / "prompts" / "v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(all(label in prompt["detector_user"] for label in ("S1", "S2", "S3", "S4", "C1", "C2")))
        for key in ("detector_system", "detector_user", "corrector_system", "corrector_user"):
            self.assertNotIn("context_ground_truth_suffix", prompt[key])
            self.assertNotIn("alternate", prompt[key])
            self.assertNotIn("hotword", prompt[key])

    def test_split_payload_never_contains_current_target(self):
        prompt = {
            "detector_system": "detector", "detector_user": "{{PAYLOAD}}",
            "corrector_system": "corrector", "corrector_user": "{{PAYLOAD}}",
        }
        detector = render_split_baseline_messages(prompt, "detector", "历史", "假设")
        corrector = render_split_baseline_messages(
            prompt, "corrector", "历史", "假设", [{"original_fragment": "假"}]
        )
        for messages in (detector, corrector):
            self.assertNotIn("target", messages[1]["content"])
            self.assertIn("context_ground_truth_history", messages[1]["content"])
            self.assertIn("假设", messages[1]["content"])

    def test_rubric_candidates_require_exact_offsets_and_six_dimensions(self):
        rubric = {key: "正常" for key in ("S1", "S2", "S3", "S4", "C1", "C2")}
        accepted, rejected = validate_rubric_candidates(
            "pokenizer很好",
            [{"segment_ref": "s1", "original_fragment": "pokenizer", "start": 0,
              "end": 9, "confidence": 0.8, "rubric": rubric}],
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, [])

    def test_corrector_cannot_edit_outside_detector_candidates(self):
        candidates = [{"segment_ref": "s1", "original_fragment": "pokenizer"}]
        accepted, rejected = restrict_edits_to_candidates(
            [
                {"segment_ref": "s1", "original_fragment": "pokenizer", "replacement": "tokenizer"},
                {"segment_ref": "s1", "original_fragment": "很好", "replacement": "挺好"},
            ],
            candidates,
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected[0]["reason"], "not_detector_authorized")

    def test_applies_non_overlapping_edits(self):
        corrected, accepted, rejected = apply_baseline_edits(
            "pokenizer很好",
            [{"segment_ref": "s1", "original_fragment": "pokenizer", "replacement": "tokenizer",
              "category": "term", "reason": "术语近音"}],
        )
        self.assertEqual(corrected, "tokenizer很好")
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, [])

    def test_rejects_duplicate_overlap_numbers_and_whole_delete(self):
        _, accepted, rejected = apply_baseline_edits(
            "2026年很好很好",
            [
                {"segment_ref": "s1", "original_fragment": "2026", "replacement": "2025", "category": "typo", "reason": "错字"},
                {"segment_ref": "s1", "original_fragment": "很好", "replacement": "好", "category": "disfluency", "reason": "重复"},
                {"segment_ref": "s1", "original_fragment": "2026年很好很好", "replacement": "", "category": "disfluency", "reason": "重复"},
            ],
        )
        self.assertEqual(accepted, [])
        self.assertEqual({item["reason"] for item in rejected}, {"number_change", "original_not_unique", "whole_segment_deletion"})

    def test_sensevoice_tags_are_not_part_of_hypothesis(self):
        from asr_post_correction.io import prediction_text
        self.assertEqual(prediction_text({"pred": "<|zh|><|NEUTRAL|>你好"}), "你好")


if __name__ == "__main__":
    unittest.main()
