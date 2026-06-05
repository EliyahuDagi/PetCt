"""Structured (JSONL) metrics logging for training runs.

Each training run appends one JSON object per line to ``<output_dir>/metrics.jsonl``.
The Train Viewer GUI tails this file to plot live train/val curves, so the schema
is intentionally flat and stable:

    {"task": "ae2d", "phase": "train"|"val", "step": 12, "epoch": 1,
     "loss": 0.31, "wall_time": 1700000000.0, ...task-specific keys...}

Task-specific keys:
    ae2d            -> recon_l1, kl
    diff2d / ft3d   -> loss (MSE); val rows may also carry l1 / psnr on the
                       decoded prediction.

This is complementary to ``setup_logging`` (which writes the human-readable
``train.log``); both are written during a run.
"""

import json
import time
from pathlib import Path


class MetricsWriter:
    """Append-only JSONL metrics writer.

    The file is opened in append mode and flushed after every record so that a
    GUI polling the file (possibly across the Windows/WSL boundary) always sees
    complete lines.
    """

    def __init__(self, output_dir, task, filename="metrics.jsonl", append=False):
        self.task = task
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.file_path = self.output_path / filename
        # Truncate any metrics from a previous run so the GUI plots a clean curve,
        # unless we are resuming -- then append so the existing curve is preserved.
        self._fh = open(self.file_path, "a" if append else "w", encoding="utf-8")

    def log(self, phase, step, metrics, epoch=None):
        """Write a single metrics record.

        Args:
            phase: "train" or "val".
            step: global step index (int).
            metrics: dict of scalar metric name -> value.
            epoch: optional epoch index (int).
        """
        record = {
            "task": self.task,
            "phase": phase,
            "step": int(step),
            "wall_time": time.time(),
        }
        if epoch is not None:
            record["epoch"] = int(epoch)
        for key, value in metrics.items():
            record[key] = float(value)
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        return record

    def close(self):
        if self._fh and not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def read_metrics(path):
    """Read all metrics records from a JSONL file.

    Tolerant of partially written trailing lines (skips any line that does not
    parse). Returns a list of dicts. Used by tests and one-shot readers; the GUI
    uses an incremental tailing reader instead.
    """
    records = []
    p = Path(path)
    if not p.exists():
        return records
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
