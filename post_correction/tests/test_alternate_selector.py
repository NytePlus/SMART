import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from asr_post_correction.alternate_selector import (propose_consensus_edits,
                                                     propose_local_edits)
from asr_post_correction.cli import (command_alternate_consensus,
                                     command_alternate_local, command_export)


class AlternateSelectorTests(unittest.TestCase):
    def test_applies_content_difference_as_local_edit(self):
        corrected, edits = propose_local_edits("这是错词。", "这是正词。")
        self.assertEqual(corrected, "这是正词。")
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0]["source"], "alternate_local_selector")

    def test_rejects_format_only_difference(self):
        corrected, edits = propose_local_edits("H O", "HO")
        self.assertEqual(corrected, "H O")
        self.assertEqual(edits, [])

    def test_rejects_new_repeated_ngram(self):
        corrected, edits = propose_local_edits("这是很好。", "这是很好很好。")
        self.assertEqual(corrected, "这是很好。")
        self.assertEqual(edits, [])

    def test_respects_span_limit(self):
        corrected, edits = propose_local_edits("这是错误", "这是正确", max_span_chars=1)
        self.assertEqual(corrected, "这是错误")
        self.assertEqual(edits, [])

    def test_rejects_large_token_length_collapse(self):
        corrected, edits = propose_local_edits(
            "这是一个很长的错误片段", "晚", max_token_ratio=4
        )
        self.assertEqual(corrected, "这是一个很长的错误片段")
        self.assertEqual(edits, [])

    def test_consensus_requires_distinct_alternate_support(self):
        corrected, edits = propose_consensus_edits(
            "这是错词。", ["这是正词。", "这是正词。", "这是另词。"], min_agreement=2
        )
        self.assertEqual(corrected, "这是正词。")
        self.assertEqual(edits[0]["supporting_alternates"], 2)

    def test_consensus_rejects_single_source_disagreement(self):
        corrected, edits = propose_consensus_edits(
            "这是错词。", ["这是正词。", "这是另词。"], min_agreement=2
        )
        self.assertEqual(corrected, "这是错词。")
        self.assertEqual(edits, [])

    def test_consensus_cli_writes_llm_only_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary.jsonl"
            alternate_one = root / "alternate_one.jsonl"
            alternate_two = root / "alternate_two.jsonl"
            output = root / "result.jsonl"
            primary.write_text(json.dumps({"id": "1", "pred": "这是错词"}, ensure_ascii=False) + "\n", encoding="utf-8")
            for path in (alternate_one, alternate_two):
                path.write_text(json.dumps({"id": "1", "pred": "这是正词"}, ensure_ascii=False) + "\n", encoding="utf-8")
            command_alternate_consensus(Namespace(
                input=str(primary), alternate_input=[str(alternate_one), str(alternate_two)],
                output=str(output), min_agreement=2, max_ngram=4, max_span_chars=40,
            ))
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["corrected_hypothesis"], "这是正词")
            manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["llm_only"])
            self.assertEqual(manifest["deterministic_postprocessors"], [])

    def test_export_uses_dataset_local_id_for_converted_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corrected = root / "corrected.jsonl"
            output = root / "eval" / "pred.raw.jsonl"
            corrected.write_text(json.dumps({
                "id": "canvas-speech:0000001",
                "status": "ok",
                "target": "目标",
                "corrected_hypothesis": "预测",
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            command_export(Namespace(corrected=str(corrected), output=str(output)))
            exported = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exported["id"], "0000001")

    def test_cli_writes_llm_only_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary.jsonl"
            alternate = root / "alternate.jsonl"
            output = root / "result.jsonl"
            primary.write_text(json.dumps({"id": "1", "pred": "这是错词"}, ensure_ascii=False) + "\n", encoding="utf-8")
            alternate.write_text(json.dumps({"id": "1", "pred": "这是正词"}, ensure_ascii=False) + "\n", encoding="utf-8")
            command_alternate_local(Namespace(
                input=str(primary), alternate_input=str(alternate), output=str(output),
                max_ngram=4, max_span_chars=40,
            ))
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["corrected_hypothesis"], "这是正词")
            manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["llm_only"])
            self.assertEqual(manifest["deterministic_postprocessors"], [])

    def test_cli_prefers_composite_id_over_colliding_source_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary.jsonl"
            alternate = root / "alternate.jsonl"
            output = root / "result.jsonl"
            primary.write_text(
                json.dumps({
                    "id": "X0Y0Z0:0000001", "source_id": "0000001", "pred": "甲"
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            alternate.write_text(
                json.dumps({
                    "id": "X0Y0Z0:0000001", "source_id": "0000001", "pred": "乙"
                }, ensure_ascii=False) + "\n" +
                json.dumps({
                    "id": "X0Y0Z2:0000001", "source_id": "0000001", "pred": "丙"
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            command_alternate_local(Namespace(
                input=str(primary), alternate_input=str(alternate), output=str(output),
                max_ngram=4, max_span_chars=40, max_token_ratio=4.0,
            ))
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["corrected_hypothesis"], "乙")


if __name__ == "__main__":
    unittest.main()
