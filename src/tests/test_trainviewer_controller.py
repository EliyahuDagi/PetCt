import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.TrainViewer.model import TrainModel
# The controller must be importable WITHOUT gradio installed.
from src.TrainViewer.controller import TrainController


def _rows_with_metric(rows, key):
    """Mirror the UI's curve filter: keep rows that carry the metric key."""
    return [r for r in rows if key in r and r.get(key) is not None]


class TestController(unittest.TestCase):
    def _make(self, outputs_root):
        model = TrainModel(outputs_root=outputs_root)
        controller = TrainController(model)
        return model, controller

    def test_init_sets_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, controller = self._make(tmp)
            self.assertEqual(controller.task, "ae2d")

    def test_set_task_maps_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, controller = self._make(tmp)
            key = controller.set_task("NAC->AC 2D")
            self.assertEqual(key, "diff2d")
            self.assertEqual(controller.task, "diff2d")
            self.assertEqual(model.config.task, "diff2d")

    def test_tick_reads_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, controller = self._make(tmp)
            mdir = os.path.join(tmp, "ae2d")
            os.makedirs(mdir)
            with open(os.path.join(mdir, "metrics.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"task": "ae2d", "phase": "train", "step": 0, "loss": 0.5}) + "\n")
                f.write(json.dumps({"task": "ae2d", "phase": "val", "step": 1, "loss": 0.3}) + "\n")
            result = controller.tick()
            self.assertTrue(result.curves_dirty)
            rows = _rows_with_metric(result.metric_rows, "loss")
            self.assertEqual(len(rows), 2)

    def test_tick_drops_stale_rows_on_external_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, controller = self._make(tmp)
            mdir = os.path.join(tmp, "ae2d")
            os.makedirs(mdir)
            path = os.path.join(mdir, "metrics.jsonl")
            # First (old) run: a long curve ending at a low loss.
            with open(path, "w", encoding="utf-8") as f:
                for step in range(5):
                    f.write(json.dumps({"phase": "train", "step": step, "loss": 1.0 - step * 0.1}) + "\n")
            result = controller.tick()
            self.assertEqual(len(result.metric_rows), 5)
            # A new run starts OUTSIDE the GUI: the file is truncated and rewritten
            # from step 0. The plot must show only the new run, not old+new.
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"phase": "train", "step": 0, "loss": 0.9}) + "\n")
            result = controller.tick()
            self.assertEqual(len(result.metric_rows), 1)
            self.assertEqual(result.metric_rows[0]["step"], 0)
            self.assertEqual(len(controller.metric_rows), 1)

    def _write_run(self, tmp, run_id, loss, mtime, steps=1):
        d = os.path.join(tmp, "ae2d", "runs", run_id)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "metrics.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for s in range(steps):
                f.write(json.dumps({"phase": "train", "step": s, "loss": loss}) + "\n")
        os.utime(path, (mtime, mtime))

    def test_live_follow_switches_to_newest_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, controller = self._make(tmp)
            self._write_run(tmp, "runA", loss=0.5, mtime=1000)
            result = controller.tick()
            self.assertEqual(result.metric_rows[0]["loss"], 0.5)
            # A newer run appears -> live-follow jumps to it, plot resets.
            self._write_run(tmp, "runB", loss=0.2, mtime=2000)
            result = controller.tick()
            self.assertEqual(len(result.metric_rows), 1)
            self.assertEqual(result.metric_rows[0]["loss"], 0.2)

    def test_selecting_a_run_pins_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, controller = self._make(tmp)
            self._write_run(tmp, "runA", loss=0.5, mtime=1000)
            self._write_run(tmp, "runB", loss=0.2, mtime=2000)
            result = controller.tick()
            self.assertEqual(result.metric_rows[0]["loss"], 0.2)  # live -> newest (runB)
            # Pin runA explicitly.
            result = controller.on_run_changed("runA")
            self.assertEqual(result.metric_rows[0]["loss"], 0.5)
            # Even when a newer run appears, the pinned run stays.
            self._write_run(tmp, "runC", loss=0.1, mtime=3000)
            result = controller.tick()
            self.assertEqual(result.metric_rows[0]["loss"], 0.5)

    def test_data_dirs_change_updates_config_and_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, controller = self._make(tmp)
            dirs = [os.path.join(tmp, "rootA"), os.path.join(tmp, "rootB")]
            n = controller.on_data_dirs_changed(dirs)
            self.assertEqual(model.config.data_dirs, dirs)
            # size is either an int (module present) or None (degraded).
            self.assertTrue(n is None or isinstance(n, int))

    def test_init_refreshes_dataset_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, controller = self._make(tmp)
            # The controller exposes the same recompute the UI seeds with at build.
            n = controller.refresh_dataset_size(model.config.data_dirs)
            self.assertTrue(n is None or isinstance(n, int))

    def test_refresh_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, controller = self._make(tmp)
            self.assertEqual(controller.refresh_checkpoint(), "no checkpoint yet")
            ckdir = os.path.join(tmp, "ae2d")
            os.makedirs(ckdir)
            open(os.path.join(ckdir, "best.pt"), "wb").close()
            self.assertIn("best.pt present", controller.refresh_checkpoint())


if __name__ == "__main__":
    unittest.main()
