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

    def test_archives_each_run_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            w1 = MetricsWriter(tmp, "ae2d", run_id="runA")
            w1.log("train", 0, {"loss": 1.0})
            w1.close()
            w2 = MetricsWriter(tmp, "ae2d", run_id="runB")
            w2.log("train", 0, {"loss": 0.2})
            w2.close()
            # Each run kept its own archive + meta.
            a = read_metrics(os.path.join(tmp, "runs", "runA", "metrics.jsonl"))
            b = read_metrics(os.path.join(tmp, "runs", "runB", "metrics.jsonl"))
            self.assertEqual(a[0]["loss"], 1.0)
            self.assertEqual(b[0]["loss"], 0.2)
            self.assertTrue(os.path.exists(os.path.join(tmp, "runs", "runA", "meta.json")))
            # The root mirror reflects the most recent run.
            root = read_metrics(os.path.join(tmp, "metrics.jsonl"))
            self.assertEqual(len(root), 1)
            self.assertEqual(root[0]["loss"], 0.2)

    def test_resume_appends_to_latest_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            w1 = MetricsWriter(tmp, "ae2d", run_id="runA")
            w1.log("train", 0, {"loss": 1.0})
            w1.close()
            # Resume (append, no run_id) continues the latest run's archive.
            w2 = MetricsWriter(tmp, "ae2d", append=True)
            self.assertEqual(w2.run_id, "runA")
            w2.log("train", 1, {"loss": 0.9})
            w2.close()
            a = read_metrics(os.path.join(tmp, "runs", "runA", "metrics.jsonl"))
            self.assertEqual(len(a), 2)

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
