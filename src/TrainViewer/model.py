import os
import sys
import json
import time
import queue
import shlex
import threading
import subprocess
from dataclasses import dataclass, field, asdict

import numpy as np

# Repo root: src/TrainViewer/model.py -> ../../ is the repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Ensure `import src.training.dataset_index` resolves from the GUI process too.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Display labels shown in the task combobox -> internal task keys.
TASK_DISPLAY_MAP = {
    "AE 2D (encode/decode)": "ae2d",
    "AE 3D (encode/decode)": "ae3d",
    "NAC->AC 2D": "diff2d",
    "NAC->AC 3D": "ft3d",
}
TASK_KEYS = ["ae2d", "ae3d", "diff2d", "ft3d"]

# Tasks that use --latent_size instead of --slice_size.
LATENT_TASKS = {"diff2d", "ft3d"}

# The training-script size flag each task expects.
SIZE_FLAG = {"ae2d": "--slice_size", "ae3d": "--crop_size", "diff2d": "--latent_size", "ft3d": "--latent_size"}

# Which frozen AE checkpoint inference loads per task (ae3d/ft3d use the 3D AE).
INFER_AE_CKPT = {
    "ae2d": "outputs/ae2d/best.pt",
    "ae3d": "outputs/ae3d/best.pt",
    "diff2d": "outputs/ae2d/best.pt",
    "ft3d": "outputs/ae3d/best.pt",
}
# Tasks whose inference uses a separate diffusion checkpoint.
DIFFUSION_TASKS = {"diff2d", "ft3d"}


def win_to_wsl_path(p):
    """Convert a Windows path to its /mnt WSL form.
    C:\\Users\\x -> /mnt/c/Users/x. Already-WSL ('/...') and '~' paths pass through.
    """
    if p is None:
        return p
    p = str(p).strip()
    if not p:
        return p
    # Already a WSL/posix path or a home-relative path: leave as-is.
    if p.startswith("/") or p.startswith("~"):
        return p
    # Drive-letter form: C:\... or C:/...
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/")
        if not rest.startswith("/"):
            rest = "/" + rest
        return "/mnt/" + drive + rest
    # UNC or relative path: best-effort backslash conversion.
    return p.replace("\\", "/")


def _data_dir_arg(data_dirs):
    """Build the multi-value '--data_dir' argument body.

    Accepts a str or a list of roots; emits each root as
    win_to_wsl_path(root) -> bash_quote(...), space-joined. Empty roots are
    skipped. Returns '' when there are no usable roots.
    """
    if data_dirs is None:
        return ""
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    quoted = [bash_quote(win_to_wsl_path(d)) for d in data_dirs if d and str(d).strip()]
    return " ".join(quoted)


def _safe_mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def bash_quote(p):
    """shlex.quote for a bash arg, but keep a leading ~ / ~user unquoted so
    bash still performs tilde expansion. Spaces elsewhere are still quoted.
    """
    p = str(p)
    if p.startswith("~"):
        head, sep, tail = p.partition("/")
        return head + sep + shlex.quote(tail) if sep else head
    return shlex.quote(p)


@dataclass
class TrainViewerConfig:
    # Form defaults; pre-fill the Train tab.
    # data_dir is kept for backward-compatible settings migration; data_dirs is
    # the multi-root source of truth used by all command builders.
    data_dir: str = os.path.join(REPO_ROOT, "data")
    data_dirs: list = field(default_factory=list)
    patient_index: int = 0
    epochs: int = 50
    val_every: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    size: int = 256          # slice_size (ae2d) or latent_size (diff2d/ft3d)
    val_fraction: float = 0.1
    device: str = "cuda"
    # WSL settings.
    distro: str = "Ubuntu-24.04"
    venv_path: str = "~/petct/.venv"
    project_path: str = ""   # auto-filled with WSL form of repo root if empty
    # Last selected task display label.
    task: str = "ae2d"
    # View-tab inference inputs.
    infer_patient_index: int = 0
    infer_slice: int = 0

    def __post_init__(self):
        if not self.project_path:
            self.project_path = win_to_wsl_path(REPO_ROOT)
        self._seed_data_dirs()

    def _seed_data_dirs(self):
        """Migrate a legacy single data_dir into the data_dirs list when empty."""
        if not self.data_dirs and self.data_dir:
            self.data_dirs = [self.data_dir]


class MetricsReader:
    """Incrementally tails a metrics.jsonl file, returning only new parsed rows."""

    def __init__(self, path):
        self.path = path
        self._offset = 0
        # Set when the file is truncated/recreated under us (a new run started
        # while we were tailing). Consumers read this to drop stale rows.
        self.restarted = False

    def reset(self):
        self._offset = 0
        self.restarted = False

    def read_new(self):
        """Return a list of newly-appended, fully-parsed JSON rows.
        Tolerates the file being absent, truncated, or recreated. When a restart
        is detected (file shrank or vanished after we had read content), the
        ``restarted`` flag is set so the GUI can discard the previous run's rows.
        """
        rows = []
        if not self.path or not os.path.exists(self.path):
            # File gone (e.g. cleared at run start): reset for the next run.
            if self._offset != 0:
                self.restarted = True
            self._offset = 0
            return rows
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return rows
        # File shrank/recreated -> start over.
        if size < self._offset:
            self.restarted = True
            self._offset = 0
        if size == self._offset:
            return rows
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                f.seek(self._offset)
                data = f.read()
                # Advance offset only over complete lines; keep partial tail.
                last_nl = data.rfind("\n")
                if last_nl == -1:
                    return rows
                complete = data[: last_nl + 1]
                self._offset += len(complete.encode("utf-8"))
                for line in complete.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except (ValueError, json.JSONDecodeError):
                        # Skip malformed line.
                        continue
        except OSError:
            return rows
        return rows


class CheckpointIndex:
    """Torch-free checkpoint presence/metadata lookup under outputs/<task>/."""

    def __init__(self, outputs_root):
        self.outputs_root = outputs_root

    def best_path(self, task):
        return os.path.join(self.outputs_root, task, "best.pt")

    def last_path(self, task):
        return os.path.join(self.outputs_root, task, "last.pt")

    def info(self, task, which="best"):
        """Return dict: {present, path, mtime, mtime_str}. No torch import."""
        path = self.best_path(task) if which == "best" else self.last_path(task)
        present = os.path.exists(path)
        mtime = None
        mtime_str = None
        if present:
            try:
                mtime = os.path.getmtime(path)
                mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
            except OSError:
                pass
        return {"present": present, "path": path, "mtime": mtime, "mtime_str": mtime_str}


class WslRunner:
    """Builds and spawns the training command in WSL; streams stdout to a queue."""

    def __init__(self):
        self.proc = None
        self.lines = queue.Queue()
        self._reader = None
        self.exit_code = None

    def build_inner(self, task, cfg, outputs_root):
        """Build the bash -lc inner command string for the launcher.

        All dataset roots come from cfg.data_dirs (multi-value --data_dir).
        """
        wsl_data = _data_dir_arg(cfg.data_dirs)
        size_arg = f"{SIZE_FLAG.get(task, '--slice_size')} {int(cfg.size)}"

        parts = [
            f"source {bash_quote(cfg.venv_path)}/bin/activate",
            f"cd {bash_quote(cfg.project_path)}",
            (
                "python -m src.training.train.launcher "
                f"{task} "
                f"--data_dir {wsl_data} "
                f"--patient_index {int(cfg.patient_index)} "
                f"--device {cfg.device} "
                f"--epochs {int(cfg.epochs)} "
                f"--val_every {int(cfg.val_every)} "
                f"--val_fraction {float(cfg.val_fraction)} "
                f"--batch_size {int(cfg.batch_size)} "
                f"--learning_rate {float(cfg.learning_rate)} "
                f"{size_arg}"
            ),
        ]
        inner = parts[0] + " && " + parts[1] + " && " + parts[2]

        # For ft3d optionally inflate from the 2D diffusion checkpoint if present.
        if task == "ft3d":
            diff_best = os.path.join(outputs_root, "diff2d", "best.pt")
            if os.path.exists(diff_best):
                inner += " --inflate_from outputs/diff2d/best.pt"
        return inner

    def build_cmd(self, task, cfg, outputs_root):
        inner = self.build_inner(task, cfg, outputs_root)
        return ["wsl", "-d", cfg.distro, "bash", "-lc", inner]

    def start(self, task, cfg, outputs_root):
        if self.is_running():
            return False
        self.exit_code = None
        # Drain any stale lines.
        self.lines = queue.Queue()
        cmd = self.build_cmd(task, cfg, outputs_root)
        self.lines.put("[runner] " + " ".join(cmd))
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as e:
            self.lines.put(f"[runner] failed to start: {e}")
            self.proc = None
            return False
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        return True

    def _read_stdout(self):
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for line in iter(proc.stdout.readline, ""):
            self.lines.put(line.rstrip("\n"))
        try:
            proc.stdout.close()
        except OSError:
            pass
        proc.wait()
        self.exit_code = proc.returncode
        self.lines.put(f"[runner] process exited with code {proc.returncode}")

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except OSError:
                pass

    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def drain_lines(self):
        """Non-blocking: return a list of all queued stdout lines."""
        out = []
        while True:
            try:
                out.append(self.lines.get_nowait())
            except queue.Empty:
                break
        return out


class InferenceRunner:
    """Spawns the infer CLI in WSL on a background thread; loads result arrays."""

    def __init__(self):
        self.proc = None
        self._thread = None
        self.lines = queue.Queue()

    def build_inner(self, task, cfg, data_dirs, patient_index, slice_idx, outputs_root):
        # Inference resolves a GLOBAL patient_index across all roots. Fall back
        # to the configured train data_dirs when no infer-specific roots given.
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs] if data_dirs.strip() else []
        roots = data_dirs if data_dirs else cfg.data_dirs
        wsl_data = _data_dir_arg(roots)
        out_dir = f"outputs/infer/{task}"
        ae_ckpt = INFER_AE_CKPT.get(task, "outputs/ae2d/best.pt")
        cmd = (
            "python -m src.training.infer "
            f"--task {task} "
            f"--data_dir {wsl_data} "
            f"--patient_index {int(patient_index)} "
            f"--slice {int(slice_idx)} "
            f"--ae_ckpt {ae_ckpt} "
            f"--device {cfg.device} "
        )
        # Only the diffusion tasks consume a separate diffusion checkpoint.
        if task in DIFFUSION_TASKS:
            cmd += f"--diff_ckpt outputs/{task}/best.pt "
        cmd += f"--out {out_dir}"
        inner = (
            f"source {bash_quote(cfg.venv_path)}/bin/activate"
            f" && cd {bash_quote(cfg.project_path)}"
            f" && {cmd}"
        )
        return inner

    def build_cmd(self, task, cfg, data_dirs, patient_index, slice_idx, outputs_root):
        inner = self.build_inner(task, cfg, data_dirs, patient_index, slice_idx, outputs_root)
        return ["wsl", "-d", cfg.distro, "bash", "-lc", inner]

    def drain_lines(self):
        out = []
        while True:
            try:
                out.append(self.lines.get_nowait())
            except queue.Empty:
                break
        return out

    def run_async(self, task, cfg, data_dirs, patient_index, slice_idx, outputs_root, on_done):
        """Run inference on a background thread; call on_done(result, error) when complete.
        result is (pred, gt, meta) on success; error is a string on failure.
        """
        if self._thread is not None and self._thread.is_alive():
            on_done(None, "inference already running")
            return

        def _work():
            cmd = self.build_cmd(task, cfg, data_dirs, patient_index, slice_idx, outputs_root)
            self.lines.put("[infer] " + " ".join(cmd))
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except OSError as e:
                on_done(None, f"failed to start: {e}")
                return
            if self.proc.stdout is not None:
                for line in iter(self.proc.stdout.readline, ""):
                    self.lines.put(line.rstrip("\n"))
                try:
                    self.proc.stdout.close()
                except OSError:
                    pass
            self.proc.wait()
            code = self.proc.returncode
            self.lines.put(f"[infer] process exited with code {code}")
            if code != 0:
                on_done(None, f"inference exited with code {code}")
                return
            try:
                result = self.load_results(task, outputs_root)
            except Exception as e:
                on_done(None, f"failed to load results: {e}")
                return
            on_done(result, None)

        self._thread = threading.Thread(target=_work, daemon=True)
        self._thread.start()

    def load_results(self, task, outputs_root):
        """Load pred.npy, gt.npy and meta.json from outputs/infer/<task>.
        Returns (pred_array, gt_array, meta_dict). Tolerates a missing meta.json.
        """
        base = os.path.join(outputs_root, "infer", task)
        pred = np.load(os.path.join(base, "pred.npy"))
        gt = np.load(os.path.join(base, "gt.npy"))
        meta_path = os.path.join(base, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except (ValueError, OSError):
                meta = {}
        # Fill display limits if absent.
        if "vmin" not in meta or meta.get("vmin") is None:
            meta["vmin"] = float(np.min(gt)) if gt.size else 0.0
        if "vmax" not in meta or meta.get("vmax") is None:
            meta["vmax"] = float(np.max(gt)) if gt.size else 1.0
        return pred, gt, meta


class SmokeRunner:
    """Spawns the pipeline smoke test in WSL on a background thread; loads the report.

    Runs `python -m src.training.smoke`, which exercises every stage for a few
    steps on the GPU and writes outputs/smoke_report.json (per-stage pass/fail,
    ms/step, peak GPU memory, and an extrapolated full-run ETA).
    """

    REPORT_REL = "smoke_report.json"

    def __init__(self):
        self.proc = None
        self._thread = None
        self.lines = queue.Queue()

    def build_inner(self, cfg):
        wsl_data = _data_dir_arg(cfg.data_dirs)
        cmd = (
            "python -m src.training.smoke "
            f"--data_dir {wsl_data} "
            f"--patient_index {int(cfg.patient_index)} "
            f"--device {cfg.device} "
            f"--size {int(cfg.size)} "
            f"--batch2d {int(cfg.batch_size)} "
            f"--plan_epochs {int(cfg.epochs)} "
            f"--report outputs/{self.REPORT_REL}"
        )
        return f"source {bash_quote(cfg.venv_path)}/bin/activate && cd {bash_quote(cfg.project_path)} && {cmd}"

    def build_cmd(self, cfg):
        return ["wsl", "-d", cfg.distro, "bash", "-lc", self.build_inner(cfg)]

    def drain_lines(self):
        out = []
        while True:
            try:
                out.append(self.lines.get_nowait())
            except queue.Empty:
                break
        return out

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def run_async(self, cfg, outputs_root, on_done):
        """Run the smoke test on a background thread; call on_done(report, error)."""
        if self.is_running():
            on_done(None, "smoke test already running")
            return

        def _work():
            cmd = self.build_cmd(cfg)
            self.lines.put("[smoke] " + " ".join(cmd))
            try:
                self.proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                )
            except OSError as e:
                on_done(None, f"failed to start: {e}")
                return
            if self.proc.stdout is not None:
                for line in iter(self.proc.stdout.readline, ""):
                    self.lines.put(line.rstrip("\n"))
                try:
                    self.proc.stdout.close()
                except OSError:
                    pass
            self.proc.wait()
            code = self.proc.returncode
            self.lines.put(f"[smoke] process exited with code {code}")
            # Load the report even on non-zero exit (a failed stage still writes it).
            report, err = self.load_report(outputs_root)
            if report is None:
                on_done(None, err or f"smoke exited with code {code} and wrote no report")
                return
            on_done(report, None)

        self._thread = threading.Thread(target=_work, daemon=True)
        self._thread.start()

    def load_report(self, outputs_root):
        path = os.path.join(outputs_root, self.REPORT_REL)
        if not os.path.exists(path):
            return None, "smoke_report.json not found"
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), None
        except (ValueError, OSError) as e:
            return None, f"could not read report: {e}"


class TrainModel:
    """Top-level model: owns config, runners, readers, checkpoint index, persistence."""

    SETTINGS_NAME = "trainviewer_settings.json"

    def __init__(self, outputs_root=None):
        self.outputs_root = outputs_root or os.path.join(REPO_ROOT, "outputs")
        self.settings_path = os.path.join(self.outputs_root, self.SETTINGS_NAME)
        self.config = TrainViewerConfig()
        self.runner = WslRunner()
        self.infer_runner = InferenceRunner()
        self.smoke_runner = SmokeRunner()
        self.checkpoints = CheckpointIndex(self.outputs_root)
        # One metrics reader per task; created lazily.
        self._readers = {}
        # Load persisted settings if present.
        self.load_settings()

    # --- runs ---
    def runs_root(self, task):
        return os.path.join(self.outputs_root, task, "runs")

    def live_metrics_path(self, task):
        """The root 'live mirror' file the trainer always writes to."""
        return os.path.join(self.outputs_root, task, "metrics.jsonl")

    def list_runs(self, task):
        """Enumerate archived runs for a task, newest-first.

        Each run is {"run_id", "path", "mtime"}. Reads the per-run archive under
        outputs/<task>/runs/<run_id>/metrics.jsonl. Falls back to the legacy root
        metrics.jsonl as a single run named "current" when no archive exists yet
        (e.g. data from before archival, or a run launched by old training code).
        """
        runs = []
        rroot = self.runs_root(task)
        if os.path.isdir(rroot):
            try:
                names = os.listdir(rroot)
            except OSError:
                names = []
            for name in names:
                p = os.path.join(rroot, name, "metrics.jsonl")
                if os.path.exists(p):
                    runs.append({"run_id": name, "path": p, "mtime": _safe_mtime(p)})
        if not runs:
            root = self.live_metrics_path(task)
            if os.path.exists(root):
                runs.append({"run_id": "current", "path": root, "mtime": _safe_mtime(root)})
        runs.sort(key=lambda r: r["mtime"], reverse=True)
        return runs

    def latest_run(self, task):
        runs = self.list_runs(task)
        return runs[0] if runs else None

    def run_path(self, task, run_id):
        """Resolve a run_id to its metrics path (None if it no longer exists)."""
        for r in self.list_runs(task):
            if r["run_id"] == run_id:
                return r["path"]
        return None

    # --- metrics (path-based: one tailing reader per distinct file) ---
    def _reader_for_path(self, path):
        if path not in self._readers:
            self._readers[path] = MetricsReader(path)
        return self._readers[path]

    def read_new_metrics_path(self, path):
        return self._reader_for_path(path).read_new()

    def consume_metrics_restart_path(self, path):
        """Return True once if the file at path was truncated/recreated since the
        last check (a new run started under us), clearing the flag."""
        reader = self._reader_for_path(path)
        if reader.restarted:
            reader.restarted = False
            return True
        return False

    def reset_metrics_path(self, path):
        self._reader_for_path(path).reset()

    # --- training ---
    def start_training(self, task):
        # Fresh tail of the live mirror; the presenter live-follows the new run
        # dir once the trainer creates it. Roots come from config.data_dirs.
        self.reset_metrics_path(self.live_metrics_path(task))
        return self.runner.start(task, self.config, self.outputs_root)

    def stop_training(self):
        self.runner.stop()

    def is_training(self):
        return self.runner.is_running()

    def drain_train_lines(self):
        return self.runner.drain_lines()

    def training_exit_code(self):
        return self.runner.exit_code

    # --- inference ---
    def run_inference(self, task, data_dirs, patient_index, slice_idx, on_done):
        self.infer_runner.run_async(
            task, self.config, data_dirs, patient_index, slice_idx, self.outputs_root, on_done
        )

    def drain_infer_lines(self):
        return self.infer_runner.drain_lines()

    # --- smoke test ---
    def run_smoke(self, on_done):
        self.smoke_runner.run_async(self.config, self.outputs_root, on_done)

    # --- dataset size ---
    @staticmethod
    def count_patients(roots):
        """Count patients across roots via the torch-free dataset_index module.

        Returns an int, or None if the (concurrently-developed) module is
        unavailable or errors. Uses missing_ok=True so a not-yet-existing root
        does not crash the GUI.
        """
        try:
            from src.training.dataset_index import enumerate_patients
            return len(enumerate_patients(list(roots), missing_ok=True))
        except Exception:
            return None

    def is_smoke_running(self):
        return self.smoke_runner.is_running()

    def drain_smoke_lines(self):
        return self.smoke_runner.drain_lines()

    # --- checkpoints ---
    def checkpoint_info(self, task, which="best"):
        return self.checkpoints.info(task, which)

    # --- settings persistence ---
    def save_settings(self):
        try:
            os.makedirs(self.outputs_root, exist_ok=True)
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self.config), f, indent=2)
        except OSError as e:
            print(f"Could not save settings: {e}")

    def load_settings(self):
        if not os.path.exists(self.settings_path):
            return
        try:
            with open(self.settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (ValueError, OSError) as e:
            print(f"Could not load settings: {e}")
            return
        # Apply only known fields onto the dataclass.
        for k, v in data.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        # Legacy settings only had data_dir (no data_dirs key): rebuild the list
        # from the loaded single root so old settings files migrate forward.
        if "data_dirs" not in data:
            self.config.data_dirs = []
        self.config._seed_data_dirs()
