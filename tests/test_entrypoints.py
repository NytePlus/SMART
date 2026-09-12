from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_rubric_sier import rubric_inputs
from run_segment_mapping import map_record


class EntrypointTests(unittest.TestCase):
    def test_segment_mapping_schema(self) -> None:
        result = map_record({
            "id": "sample-1",
            "reference": "我想吃一个三明治",
            "hypothesis": "我想吃一个三民治",
        })
        self.assertEqual(result["id"], "sample-1")
        self.assertEqual(len(result["error_segments"]), 1)
        self.assertIn("mapped_reference", result["mapping"])

    def test_rubric_adapter_schema(self) -> None:
        mapped = map_record({
            "id": "sample-1",
            "dataset": "smoke",
            "reference": "hello world",
            "hypothesis": "hello word",
        })
        records = rubric_inputs([mapped])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["sample_id"], "sample-1:1")
        self.assertEqual(records[0]["dataset"], "smoke")

    def test_all_commands_expose_help(self) -> None:
        for script in (
            "run_segment_mapping.py",
            "run_rubric_sier.py",
            "run_post_correction.py",
            "run_evaluation.py",
        ):
            completed = subprocess.run(
                [sys.executable, str(ROOT / script), "--help"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
