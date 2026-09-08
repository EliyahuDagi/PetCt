"""Structured (JSONL) metrics logging for training runs.

Each training run appends one JSON object per line to ``<output_dir>/metrics.jsonl``.
The Train Viewer GUI tails this file to plot live train/val curves, so the schema
is intentionally flat and stable:

    {"task": "ae2d", "phase": "train"|"val", "step": 12, "epoch": 1,
     "loss": 0.31, "wall_time": 1700000000.0, ...task-specific keys...}

Task-specific keys:
    ae2d / ae3d     -> recon_l1, kl. VAL rows additionally carry image-quality
                       metrics on the reconstruction: psnr, ssim, nrmse, mae,
                       rel_bias. ae3d TRAIN rows also carry lr (the learning
                       rate that step's update was taken at; warmup + cosine).
    diff2d / ft3d   -> loss (total). TRAIN rows also carry mse (the diffusion
                       noise MSE) and perceptual (the weighted perceptual term, 0
                       when disabled). VAL rows additionally carry l1 plus the
                       image-quality metrics on the decoded one-step prediction:
                       psnr, ssim, nrmse, mae, rel_bias.

Image-quality keys come from ``image_metrics.image_quality_metrics`` and compare
the (decoded) prediction to the reference in image space. ``rel_bias`` is a
normalized-intensity proxy for SUV mean bias (not calibrated SUV units).

This is complementary to ``setup_logging`` (which writes the human-readable
``train.log``); both are written during a run.
"""

import json
import time
from pathlib import Path


def _generate_run_id():
    """A human-readable, sortable run id from the local wall clock."""
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _latest_run_dir(runs_root):
    """Return the most-recently-modified run dir under runs_root, or None."""
    if not runs_root.is_dir():
        return None
    candidates = [d for d in runs_root.iterdir() if d.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


class MetricsWriter:
    """Append-only JSONL metrics writer with per-run archival.

    Every record is written to two places, flushed after each write so a GUI
    polling across the Windows/WSL boundary always sees complete lines:

    * ``<output_dir>/metrics.jsonl`` -- the "live" mirror of the most recent run
      (truncated at start unless resuming). Kept for backward compatibility:
      inference, smoke and older GUIs read this path.
    * ``<output_dir>/runs/<run_id>/metrics.jsonl`` -- a durable per-run archive so
      finished runs are never overwritten and can be revisited in the GUI.

    A small ``meta.json`` is written beside each archive with the run id and
    start time so the GUI can label runs.
    """

    def __init__(self, output_dir, task, filename="metrics.jsonl", append=False, run_id=None):
        self.task = task
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.file_path = self.output_path / filename

        # Resolve the per-run archive directory.
        runs_root = self.output_path / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        if run_id is None:
            if append:
                # Resuming: continue the most recent existing run if there is one.
                latest = _latest_run_dir(runs_root)
                run_id = latest.name if latest is not None else _generate_run_id()
            else:
                run_id = _generate_run_id()
                # Avoid clobbering a run started within the same second.
                base, n = run_id, 1
                while (runs_root / run_id).exists():
                    run_id = f"{base}-{n}"
                    n += 1
        self.run_id = run_id
        self.run_dir = runs_root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_file_path = self.run_dir / filename

        mode = "a" if append else "w"
        # Truncate any metrics from a previous run so the GUI plots a clean curve,
        # unless we are resuming -- then append so the existing curve is preserved.
        self._fh = open(self.file_path, mode, encoding="utf-8")
        self._run_fh = open(self.run_file_path, mode, encoding="utf-8")

        if not append:
            self._write_meta()

    def _write_meta(self):
        meta = {"task": self.task, "run_id": self.run_id, "start_time": time.time()}
        try:
            with open(self.run_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f)
        except OSError:
            pass

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
        line = json.dumps(record) + "\n"
        for fh in (self._fh, self._run_fh):
            fh.write(line)
            fh.flush()
        return record

    def close(self):
        for fh in (getattr(self, "_fh", None), getattr(self, "_run_fh", None)):
            if fh and not fh.closed:
                fh.close()

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
