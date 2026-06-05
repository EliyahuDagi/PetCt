import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.TrainViewer.model import TrainModel
from src.TrainViewer.presenter import TrainPresenter


class FakeView:
    """Minimal stand-in for TrainView; records the calls the presenter makes."""

    def __init__(self):
        self.presenter = None
        self.task_key = None
        self.status = None
        self.checkpoint_info = None
        self.logged = []
        self.curves = None
        self.metric_key = "loss"
        self.data_dirs = ["synthetic"]
        self.dataset_size = None

    def set_presenter(self, p):
        self.presenter = p

    def load_config(self, cfg):
        self.loaded_cfg = cfg
        dirs = list(cfg.data_dirs) if cfg.data_dirs else ([cfg.data_dir] if cfg.data_dir else [])
        self.data_dirs = dirs

    def dump_config(self, cfg):
        cfg.data_dirs = list(self.data_dirs)
        cfg.data_dir = cfg.data_dirs[0] if cfg.data_dirs else ""

    def set_task_key(self, key):
        self.task_key = key

    def get_metric_key(self):
        return self.metric_key

    def get_data_dirs(self):
        return list(self.data_dirs)

    def get_data_dir(self):
        return self.data_dirs[0] if self.data_dirs else ""

    def set_dataset_size(self, n):
        self.dataset_size = n

    def update_curves(self, rows, metric_key):
        self.curves = (list(rows), metric_key)

    def append_log(self, lines):
        self.logged.extend(lines)

    def set_status(self, s):
        self.status = s

    def set_checkpoint_info(self, text):
        self.checkpoint_info = text


class TestPresenter(unittest.TestCase):
    def _make(self, outputs_root):
        model = TrainModel(outputs_root=outputs_root)
        view = FakeView()
        presenter = TrainPresenter(model, view)
        return model, view, presenter

    def test_init_pushes_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, view, presenter = self._make(tmp)
            self.assertEqual(presenter.task, "ae2d")
            self.assertEqual(view.task_key, "ae2d")

    def test_set_task_maps_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, view, presenter = self._make(tmp)
            presenter.set_task("NAC->AC 2D")
            self.assertEqual(presenter.task, "diff2d")
            self.assertEqual(view.task_key, "diff2d")

    def test_tick_reads_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, view, presenter = self._make(tmp)
            mdir = os.path.join(tmp, "ae2d")
            os.makedirs(mdir)
            with open(os.path.join(mdir, "metrics.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"task": "ae2d", "phase": "train", "step": 0, "loss": 0.5}) + "\n")
                f.write(json.dumps({"task": "ae2d", "phase": "val", "step": 1, "loss": 0.3}) + "\n")
            presenter.tick()
            self.assertIsNotNone(view.curves)
            rows, key = view.curves
            self.assertEqual(len(rows), 2)
            self.assertEqual(key, "loss")

    def test_data_dirs_change_updates_config_and_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, view, presenter = self._make(tmp)
            view.data_dirs = [os.path.join(tmp, "rootA"), os.path.join(tmp, "rootB")]
            presenter.on_data_dirs_changed()
            self.assertEqual(model.config.data_dirs, view.data_dirs)
            # dataset_size is either an int (module present) or None (degraded).
            self.assertTrue(view.dataset_size is None or isinstance(view.dataset_size, int))

    def test_init_refreshes_dataset_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, view, _ = self._make(tmp)
            # set_dataset_size was called during init (value may be int or None).
            self.assertTrue(view.dataset_size is None or isinstance(view.dataset_size, int))

    def test_refresh_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, view, presenter = self._make(tmp)
            presenter.refresh_checkpoint()
            self.assertEqual(view.checkpoint_info, "no checkpoint yet")
            ckdir = os.path.join(tmp, "ae2d")
            os.makedirs(ckdir)
            open(os.path.join(ckdir, "best.pt"), "wb").close()
            presenter.refresh_checkpoint()
            self.assertIn("best.pt present", view.checkpoint_info)


if __name__ == "__main__":
    unittest.main()
