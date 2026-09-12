import tempfile
import unittest
from argparse import Namespace
import json
from pathlib import Path

from asr_post_correction.cli import command_prepare_headline_cseer
from asr_post_correction.evaluation import evaluate_files, evaluate_headline


DATA_PREPARE_ROOT = Path(__file__).resolve().parents[3] / "data-prepare"


@unittest.skipUnless(DATA_PREPARE_ROOT.is_dir(), "data-prepare checkout is required")
class DataPrepareEvaluationIntegrationTests(unittest.TestCase):
    def test_evaluate_files_normalizes_sensevoice_tags_for_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = root / "before.jsonl"
            corrected = root / "corrected.jsonl"
            before.write_text(
                '{"id":"1","target":"你好","pred":"<|zh|><|NEUTRAL|>你好"}\n',
                encoding="utf-8",
            )
            corrected.write_text(
                '{"id":"1","status":"ok","original_hypothesis":"<|zh|><|NEUTRAL|>你好","corrected_hypothesis":"你好"}\n',
                encoding="utf-8",
            )
            summary = evaluate_files(
                DATA_PREPARE_ROOT,
                before,
                corrected,
                root / "analysis",
                require_complete=True,
            )

        self.assertEqual(summary["raw_input_mismatches"], 0)

    def test_non_aishell_headline_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qwen = evaluate_headline(
                DATA_PREPARE_ROOT, "qwen3-asr", root / "qwen3"
            )
            sensevoice = evaluate_headline(
                DATA_PREPARE_ROOT, "sensevoice", root / "sensevoice"
            )

        self.assertEqual(qwen["utterances"], 467)
        self.assertEqual(qwen["baseline"]["reference_tokens"], 18_284)
        self.assertAlmostEqual(qwen["baseline"]["wer"], 12.97309122730256)
        self.assertEqual(sensevoice["utterances"], 467)
        self.assertEqual(sensevoice["baseline"]["reference_tokens"], 18_284)
        self.assertAlmostEqual(sensevoice["baseline"]["wer"], 11.868300153139357)

    def test_prepare_headline_cseer_uses_official_rule_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corrected = root / "corrected.jsonl"
            corrected.write_text(
                json.dumps({
                    "id": "X0Y0Z0:0000001",
                    "status": "ok",
                    "original_hypothesis": "嗯。",
                    "corrected_hypothesis": "嗯。",
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            model_dir = root / "model"
            command_prepare_headline_cseer(Namespace(
                data_prepare_root=str(DATA_PREPARE_ROOT),
                model_key="qwen3-asr",
                corrected=str(corrected),
                model_dir=str(model_dir),
                sample_count=1,
            ))
            mapping = json.loads(
                (model_dir / "segment_mapping.json").read_text(encoding="utf-8")
            )

        self.assertEqual(mapping[0]["mapping_pred"], "嗯")
        self.assertEqual(mapping[0]["normalization"]["pred_engine"], "rule")

    def test_prepare_headline_cseer_strips_sensevoice_control_tags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corrected = root / "corrected.jsonl"
            corrected.write_text(
                json.dumps({
                    "id": "X0Y0Z0:0000001",
                    "status": "ok",
                    "original_hypothesis": "<|zh|><|NEUTRAL|>嗯。",
                    "corrected_hypothesis": "<|zh|><|NEUTRAL|>嗯。",
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            model_dir = root / "model"
            command_prepare_headline_cseer(Namespace(
                data_prepare_root=str(DATA_PREPARE_ROOT),
                model_key="sensevoice",
                corrected=str(corrected),
                model_dir=str(model_dir),
                sample_count=1,
            ))
            pred = json.loads(
                (model_dir / "pred.raw.jsonl").read_text(encoding="utf-8")
            )

        self.assertEqual(pred["pred"], "嗯。")


if __name__ == "__main__":
    unittest.main()
