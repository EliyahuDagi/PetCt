import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training.utils.metrics import MetricsWriter, read_metrics


class TestMetricsWriter(unittest.TestCase):
    def test_writes_jsonl_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = MetricsWriter(tmp, "ae2d")
            w.log("train", 0, {"loss": 0.5, "recon_l1": 0.4, "kl": 0.1}, epoch=0)
            rec = w.log("val", 1, {"loss": 0.3}, epoch=0)
            w.close()

            self.assertEqual(rec["task"], "ae2d")
            self.assertEqual(rec["phase"], "val")
            self.assertIn("wall_time", rec)

            rows = read_metrics(os.path.join(tmp, "metrics.jsonl"))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["phase"], "train")
            self.assertEqual(rows[0]["recon_l1"], 0.4)
            self.assertEqual(rows[1]["phase"], "val")
            self.assertEqual(rows[1]["step"], 1)

    def test_truncates_on_new_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            w1 = MetricsWriter(tmp, "diff2d")
            w1.log("train", 0, {"loss": 1.0})
            w1.close()
            # A second writer (new run) starts the file fresh.
            w2 = MetricsWriter(tmp, "diff2d")
            w2.log("train", 0, {"loss": 0.2})
            w2.close()
            rows = read_metrics(os.path.join(tmp, "metrics.jsonl"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["loss"], 0.2)

    def test_read_metrics_skips_partial_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metrics.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"task": "ae2d", "phase": "train", "step": 0, "loss": 0.1}) + "\n")
                f.write('{"task": "ae2d", "phase": "tra')  # partial, no newline
            rows = read_metrics(path)
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
