"""Shared data loading + train/val splitting for the training scripts.

Loads a patient's normalized volumes via the existing ``DicomModel`` and provides
deterministic train/val sampling for the three tasks:

  * AE (ae2d)        -> pooled 2D PET slices (NAC + AC) so both share one latent.
  * NAC->AC 2D       -> paired (NAC, AC) 2D slices at matching normalized depth.
  * NAC->AC 3D       -> paired (NAC, AC) 3D crops.

The split is deterministic given ``seed`` so train/val never overlap across calls.
With more than one usable patient the stages use a by-patient holdout: the train
scripts use :func:`make_patient_split3` (train/val/TEST -- test patients held out
from both train and val) while the legacy two-way :func:`make_patient_split` is
kept for backward compatibility (evaluate.py / tests). They sample over each
patient's full depth; with a single patient they fall back to a within-patient split -- for 2D a grid of
normalized depth positions split into train/val pools, for 3D a contiguous
train/val depth band. Patients are enumerated via the torch-free
:mod:`src.training.dataset_index` and loaded lazily through
:class:`PatientVolumeCache` to bound memory when pooling across many patients.
"""

import hashlib
import json
import os
import queue
import threading
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from src.training.dataset_index import enumerate_patients


def normalize_volume(vol_np):
    """Percentile (1/99) normalize a volume to [0, 1] float32."""
    vol = np.asarray(vol_np).astype(np.float32)
    vmin = np.percentile(vol, 1.0)
    vmax = np.percentile(vol, 99.0)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return np.clip((vol - vmin) / (vmax - vmin), 0.0, 1.0)


def load_patient_by_path(patient_path, device=None, load_ct=True, run_segmentation=True,
                         cache_dir=None, write_through=False):
    """Load one patient's normalized CT / AC-PET / NAC-PET volumes by folder path.

    Returns a dict with torch tensors (or None) under keys ``ct``, ``pet_ac``,
    ``pet_nac`` plus ``spacing`` (dz,dy,dx) and ``origin`` (z,y,x). Each volume is
    shaped (Z, Y, X). ``DicomModel`` is imported lazily so the torch-free
    enumeration helpers stay importable on a host without the viewer deps.

    ``load_ct=False`` skips reading the (large) CT series and ``run_segmentation
    =False`` skips the segmentor -- the PET training pipelines pass both False,
    which cuts per-patient load time several-fold (CT I/O dominates the cost).

    SSD cache (opt-in, transparent): when ``cache_dir`` is set (or the
    ``PETCT_CACHE_DIR`` env var is) and a fresh PET-only entry exists for this
    patient, the volumes are read from the SSD in ~ms WITHOUT touching DICOM and
    returned in the same dict shape (note: a cache hit always has ``ct=None`` --
    the cache is PET-only). On a miss the DICOM path runs as before; with
    ``write_through=True`` a PET-only load also populates the cache. ``cache_dir
    =None`` (the default) is the pure-DICOM path -- byte-identical to before.
    """
    from src.training.precache import (
        is_fresh,
        load_entry,
        resolve_cache_dir,
        write_entry,
    )

    cache_dir = resolve_cache_dir(cache_dir)
    if cache_dir is not None and is_fresh(cache_dir, patient_path):
        cached = load_entry(cache_dir, patient_path, device=device, to_tensor=True)
        if cached is not None:
            return cached

    from src.DataViewer.model import DicomModel

    dm = DicomModel()
    # `patient_path` is already a single patient folder (enumerate_patients
    # resolved it), so load it directly instead of via load_dataset(). That
    # method's study-grouping heuristic only recognizes subdirs named *exactly*
    # CT/PET/PT/Segmentation/..., so a layout like ACRIN 6668 -- whose patient
    # folders hold sibling study dirs CT_*/PET_AC_*/PET_NAC_PET_WB_UNCORRECTED --
    # is split into separate "studies" and only the first (the CT) would load,
    # silently dropping the NAC/AC PET pair. load_patient_data() os.walks the
    # whole folder and collects every CT/PET series, so AC and NAC are both found.
    dm.patient_list = [patient_path]
    dm.current_patient_index = 0
    try:
        dm.load_patient_data(patient_path, load_ct=load_ct, run_segmentation=run_segmentation)
    except ValueError:
        # No usable volumes (e.g. CT skipped and this patient has no PET).
        # Return an all-None dict so multi-patient samplers skip it instead of
        # crashing; single-patient callers still fail downstream as before.
        return {"ct": None, "pet_ac": None, "pet_nac": None,
                "spacing": None, "origin": None, "patient_path": patient_path}

    def _to_tensor(vol):
        if vol is None:
            return None
        t = torch.from_numpy(normalize_volume(vol))
        return t.to(device) if device is not None else t

    result = {
        "ct": _to_tensor(dm.ct_volume),
        "pet_ac": _to_tensor(dm.pet_volume),
        "pet_nac": _to_tensor(dm.pet_nac_volume),
        "spacing": dm.get_voxel_spacing(),
        "origin": dm.get_origin(),
        "patient_path": dm.patient_list[0],
    }

    # Write-through: populate the SSD cache on a miss so the next run is fast. Only
    # meaningful for a PET-only load (the cache is PET-only); a CT-bearing dict
    # still writes just its PET arrays. Failures here must never break training.
    if cache_dir is not None and write_through:
        try:
            if result.get("pet_ac") is not None or result.get("pet_nac") is not None:
                write_entry(cache_dir, result)
        except OSError:
            pass

    return result


def load_patient_volumes(data_dir, patient_index, device=None, cache_dir=None,
                         load_ct=True, run_segmentation=True):
    """Load one patient's normalized volumes from a (single) dataset root.

    Backward-compatible single-root entry point: enumerates patients under
    ``data_dir`` and loads the one at ``patient_index``. Implemented on top of
    :func:`enumerate_patients` + :func:`load_patient_by_path`. ``cache_dir``
    (default None) threads the opt-in SSD cache through; None keeps prior behavior.

    ``data_dir`` may be either a *dataset root* (whose immediate subdirs are
    patient folders) or an already-resolved *single patient folder*. The latter
    happens when ``infer.main`` has resolved a global ``--patient_index`` against
    multiple roots and collapses to one patient path before calling the runners.
    For an ACRIN-style patient folder the children are DICOM *series* dirs
    (``CT_CT_WB_2_5`` / ``PET_AC_...`` / ``PET_NAC_...``) -- none of which match a
    patient marker -- so a naive ``enumerate_patients`` would descend one level too
    deep and return the series dirs (loading e.g. CT-only). Detect that case
    (a single string path that enumeration does *not* report as its own sole
    patient) and load the folder directly via :func:`load_patient_by_path`.
    """
    if isinstance(data_dir, str):
        patients = enumerate_patients(data_dir)
        # If ``data_dir`` is itself a patient folder, ``enumerate_patients`` returns
        # its (non-marker) series subdirs rather than ``[data_dir]``. Load it directly.
        norm = os.path.normcase(os.path.abspath(data_dir))
        is_self = len(patients) == 1 and os.path.normcase(os.path.abspath(patients[0])) == norm
        if not is_self:
            return load_patient_by_path(data_dir, device=device, cache_dir=cache_dir,
                                        load_ct=load_ct, run_segmentation=run_segmentation)
    else:
        patients = enumerate_patients(data_dir)
    if patient_index >= len(patients):
        raise ValueError("patient_index out of range (have %d patients)" % len(patients))
    return load_patient_by_path(patients[patient_index], device=device, cache_dir=cache_dir,
                                load_ct=load_ct, run_segmentation=run_segmentation)


class PatientVolumeCache:
    """Bounded LRU cache of loaded patient volume dicts.

    Loads patients lazily by index into ``patient_paths`` and keeps at most
    ``max_cached`` of them resident, evicting the least-recently-used. This
    bounds GPU/host memory when pooling across many patients.
    """

    def __init__(self, patient_paths, device=None, max_cached=4, load_ct=False,
                 run_segmentation=False, cache_dir=None, write_through=False):
        self.patient_paths = list(patient_paths)
        self.device = device
        # Training pools are PET-only: skip CT + segmentation per load (default).
        self.load_ct = bool(load_ct)
        self.run_segmentation = bool(run_segmentation)
        # Opt-in SSD cache (None = pure DICOM, unchanged behavior). When set, loads
        # come from the SSD on a hit and (with write_through) populate it on a miss.
        self.cache_dir = cache_dir
        self.write_through = bool(write_through)
        # The cache size is the dominant lever on multi-patient step time: a tiny
        # cache over a large pool means nearly every step re-reads DICOM from disk
        # (~8 s/patient). PET-only entries are small (~100 MB: AC+NAC), so for the
        # PET pipelines raise the floor to cover the pool -- up to 64 patients
        # (~6.4 GB), which fully caches the diffusion pool (ACRIN, ~20 patients)
        # and gives the AE pool a high hit rate. CT-bearing caches keep the small
        # default (each entry is ~5x larger).
        if not self.load_ct:
            max_cached = max(int(max_cached), min(len(self.patient_paths), 64))
        self.max_cached = max(1, int(max_cached))
        self._cache = OrderedDict()  # index -> volumes dict (LRU: newest at end)

    def __len__(self):
        return len(self.patient_paths)

    def _load(self, index, device):
        """Load patient ``index`` onto ``device`` (override point for subclasses)."""
        return load_patient_by_path(
            self.patient_paths[index], device=device,
            load_ct=self.load_ct, run_segmentation=self.run_segmentation,
            cache_dir=self.cache_dir, write_through=self.write_through,
        )

    def get(self, index):
        """Return the volumes dict for ``patient_paths[index]`` (loads lazily)."""
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        vols = self._load(index, self.device)
        self._cache[index] = vols
        self._cache.move_to_end(index)
        while len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)  # evict LRU
        return vols


_VOLUME_KEYS = ("ct", "pet_ac", "pet_nac")


def move_patient_to_device(vols, device, non_blocking=False):
    """Return a copy of a loaded patient dict with its volume tensors on ``device``.

    Only the volume entries (``ct`` / ``pet_ac`` / ``pet_nac``) are moved; scalar
    metadata (``spacing`` / ``origin`` / ``patient_path``) is copied by reference.
    ``device=None`` returns the dict unchanged (CPU). This is the cheap (~ms) H2D
    copy the main thread does after a background thread has done the expensive
    CPU-side DICOM read; worker threads must not touch CUDA themselves.
    """
    if device is None:
        return vols
    out = dict(vols)
    for k in _VOLUME_KEYS:
        v = out.get(k)
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=non_blocking)
    return out


class _EpochIndexIterator:
    """Deterministic shuffled-epoch iterator over a fixed list of patient indices.

    Yields every index once per epoch in a shuffled order (standard shuffled SGD),
    reshuffling at each epoch boundary. Determinism comes from the supplied numpy
    RandomState, so a given seed reproduces the same stream. Unlike random-with-
    replacement this lets a prefetcher look ahead predictively (the next K indices
    are known in advance).
    """

    def __init__(self, indices, rng):
        self._indices = [int(i) for i in indices]
        if not self._indices:
            raise ValueError("_EpochIndexIterator needs at least one index.")
        self._rng = rng
        self._order = []
        self._pos = 0

    def _reshuffle(self):
        order = list(self._indices)
        self._rng.shuffle(order)
        self._order = order
        self._pos = 0

    def next(self):
        if self._pos >= len(self._order):
            self._reshuffle()
        idx = self._order[self._pos]
        self._pos += 1
        return idx


class PrefetchingPatientCache(PatientVolumeCache):
    """:class:`PatientVolumeCache` with an opt-in background prefetch thread.

    When ``prefetch > 0`` a daemon worker thread walks a shuffled-epoch iterator of
    the *training* indices, loads each patient on **CPU** (the expensive ~5-8 s
    DICOM read), and pushes it into a bounded queue so the read overlaps with the
    GPU compute of the current step. The main thread calls :meth:`next_train` to
    pop the next ready patient and moves its volumes to ``device`` (a cheap ms H2D
    copy). The LRU ``get(index)`` of the base class is preserved for the validation
    path and for direct index access.

    Threading (not multiprocessing) is intentional: pydicom/numpy file I/O release
    the GIL so the worker overlaps with the main thread, and we avoid pickling /
    CUDA-tensor-across-process issues. Worker threads never call CUDA.

    With ``prefetch <= 0`` (the default) this class behaves identically to
    :class:`PatientVolumeCache` -- no thread is started and ``next_train`` falls
    back to the synchronous shuffled-epoch path.
    """

    def __init__(self, patient_paths, device=None, max_cached=4, load_ct=False,
                 run_segmentation=False, prefetch=0, train_indices=None, rng=None,
                 pin_memory=False, loader=None, cache_dir=None, write_through=False):
        super().__init__(patient_paths, device=device, max_cached=max_cached,
                         load_ct=load_ct, run_segmentation=run_segmentation,
                         cache_dir=cache_dir, write_through=write_through)
        # Injectable loader (defaults to the module fn) so tests can supply a fake
        # without touching real DICOM; the base get() path also uses it.
        self._loader = loader or load_patient_by_path
        self.prefetch = max(0, int(prefetch))
        self.pin_memory = bool(pin_memory)
        if train_indices is None:
            train_indices = list(range(len(self.patient_paths)))
        # The epoch iterator runs on the worker thread (when prefetch>0). numpy
        # RandomState is NOT thread-safe and the caller's ``rng`` is also used by the
        # main training loop, so derive an INDEPENDENT, deterministically-seeded
        # RandomState for index shuffling instead of sharing the caller's object.
        # Only draw from the caller's rng when prefetch is enabled, so the default
        # (prefetch==0) path leaves the caller's rng stream byte-identical.
        if self.prefetch > 0 and rng is not None:
            iter_seed = int(rng.randint(0, 2 ** 31 - 1))
        else:
            iter_seed = 0
        self._train_iter = _EpochIndexIterator(train_indices, np.random.RandomState(iter_seed))
        # Background-prefetch state (only used when self.prefetch > 0).
        self._q = None
        self._stop = None
        self._worker = None
        self._exc = None

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        """Start the background prefetch worker (no-op if prefetch disabled)."""
        if self.prefetch <= 0 or self._worker is not None:
            return self
        self._q = queue.Queue(maxsize=self.prefetch)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name="patient-prefetch", daemon=True)
        self._worker.start()
        return self

    def close(self):
        """Signal the worker to stop and join it; safe to call repeatedly."""
        if self._stop is not None:
            self._stop.set()
        if self._q is not None:
            # Drain so a worker blocked on put() can observe the stop flag and exit.
            try:
                while True:
                    self._q.get_nowait()
            except queue.Empty:
                pass
        if self._worker is not None:
            self._worker.join(timeout=10.0)
            self._worker = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
        return False

    # -- worker ------------------------------------------------------------
    def _load(self, index, device):
        # The injected fake loader in tests has the (path, device, load_ct,
        # run_segmentation) signature only; cache kwargs are forwarded just to the
        # real module loader so a fake stays a drop-in. When no cache dir is set
        # there is nothing to forward, preserving the prior call exactly.
        if self.cache_dir is None and not self.write_through:
            return self._loader(
                self.patient_paths[index], device=device,
                load_ct=self.load_ct, run_segmentation=self.run_segmentation,
            )
        return self._loader(
            self.patient_paths[index], device=device,
            load_ct=self.load_ct, run_segmentation=self.run_segmentation,
            cache_dir=self.cache_dir, write_through=self.write_through,
        )

    def _load_cpu(self, index):
        return self._load(index, None)

    def _run(self):
        while not self._stop.is_set():
            idx = self._train_iter.next()
            try:
                vols = self._load_cpu(idx)
                if self.pin_memory:
                    for k in _VOLUME_KEYS:
                        v = vols.get(k)
                        if isinstance(v, torch.Tensor):
                            vols[k] = v.pin_memory()
            except BaseException as exc:  # surface to the main thread
                self._exc = exc
                # Push a sentinel so a blocked consumer wakes and re-raises.
                try:
                    self._q.put((idx, None), timeout=1.0)
                except queue.Full:
                    pass
                return
            # Block (with timeout) until there's room or we're asked to stop.
            while not self._stop.is_set():
                try:
                    self._q.put((idx, vols), timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _raise_worker_exc(self):
        if self._exc is not None:
            exc, self._exc = self._exc, None
            raise exc

    # -- consumer ----------------------------------------------------------
    def next_train(self):
        """Return ``(index, volumes_on_device)`` for the next training patient.

        Uses the background prefetch queue when enabled, else loads synchronously
        from the shuffled-epoch iterator (identical statistics, just not overlapped).
        """
        if self.prefetch <= 0 or self._worker is None:
            idx = self._train_iter.next()
            return idx, self.get(idx)
        # Poll the queue so a dead worker (after an exception) can't deadlock us.
        while True:
            try:
                idx, vols = self._q.get(timeout=0.1)
                break
            except queue.Empty:
                if not self._worker.is_alive():
                    self._raise_worker_exc()
                    raise RuntimeError("Prefetch worker stopped unexpectedly.")
        if vols is None:  # worker error sentinel
            self._raise_worker_exc()
            raise RuntimeError("Prefetch worker stopped unexpectedly.")
        self._raise_worker_exc()
        # Worker loaded on CPU; the main thread does the cheap H2D copy. The base
        # LRU is left device-only (it backs the synchronous get()/val path), so we
        # don't store the CPU dict there.
        return idx, move_patient_to_device(vols, self.device, non_blocking=self.pin_memory)


def ae_pool_volumes(vols, modality):
    """Pick the volumes to pool for AE training from a loaded patient dict.

    For ``modality == "pet"`` prefer [AC, NAC] PET, falling back to [CT] when no
    PET is present. For ``modality == "ct"`` use [CT]. Returns a list of non-None
    tensors (possibly empty if the patient has nothing usable).
    """
    if modality == "pet":
        pool = [vols.get("pet_ac"), vols.get("pet_nac")]
        if all(v is None for v in pool):
            pool = [vols.get("ct")]
    else:
        pool = [vols.get("ct")]
    return [v for v in pool if v is not None]


def filter_paired_patients(patient_paths, device=None, log=None, cache_dir=None):
    """Return the subset of ``patient_paths`` that have BOTH NAC and AC PET.

    Used by the diffusion stages, which require paired data. Each patient is
    loaded once (on CPU by default to keep the check cheap; pass ``device`` to
    load onto the target device). ``log`` is an optional callable (e.g.
    ``logger.info``) used to report skips. Returns the list of kept paths.

    When ``cache_dir`` is set and a patient has a fresh cache sidecar, pairing is
    read from the sidecar's ``has_ac``/``has_nac`` flags WITHOUT loading any array
    (near-instant); otherwise that patient falls back to a DICOM load as before.
    """
    from src.training.precache import is_fresh, resolve_cache_dir, sidecar_has_pair

    cache_dir = resolve_cache_dir(cache_dir)
    paired = []
    skipped = 0
    for path in patient_paths:
        if cache_dir is not None and is_fresh(cache_dir, path):
            has_pair = sidecar_has_pair(cache_dir, path)
        else:
            # Only PET presence matters for pairing; skip CT + segmentation so this
            # startup scan over every patient stays cheap.
            vols = load_patient_by_path(path, device=device, load_ct=False,
                                        run_segmentation=False, cache_dir=cache_dir)
            has_pair = vols.get("pet_nac") is not None and vols.get("pet_ac") is not None
        if has_pair:
            paired.append(path)
        else:
            skipped += 1
            if log is not None:
                log("Skipping %s for diffusion: missing a NAC/AC pair." % (path,))
    if log is not None:
        log("Paired patients: %d kept, %d skipped (missing a pair)." % (len(paired), skipped))
    return paired


def _paired_manifest_key(patient_paths):
    """A stable hash of the sorted patient-path set + each path's mtime (if any).

    Keying on mtimes means adding/removing/touching a patient folder invalidates
    the cached pairing result (it gets rebuilt), while a pure re-run reuses it.
    """
    items = []
    for p in sorted(set(patient_paths)):
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            mtime = -1.0
        items.append((os.path.normcase(os.path.abspath(p)), round(float(mtime), 3)))
    blob = json.dumps(items, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def default_paired_cache_path(save_dir=None):
    """Default on-disk location for the paired-patient manifest under ``outputs/``."""
    base = save_dir or os.path.join("outputs", "paired_cache")
    return os.path.join(base, "paired_patients.json")


def load_paired_cache(cache_path, patient_paths):
    """Return the cached paired-patient list if the manifest matches, else None.

    The manifest is considered fresh only when its stored key equals the current
    ``_paired_manifest_key`` (same roots + mtimes); any mismatch / missing file /
    parse error returns None so the caller rescans.
    """
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if data.get("key") != _paired_manifest_key(patient_paths):
        return None
    paired = data.get("paired")
    if not isinstance(paired, list):
        return None
    return [str(p) for p in paired]


def write_paired_cache(cache_path, patient_paths, paired):
    """Write the paired-patient manifest (key + list) to ``cache_path``."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    payload = {"key": _paired_manifest_key(patient_paths), "paired": list(paired)}
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, cache_path)


def filter_paired_patients_cached(patient_paths, cache_path=None, rescan=False,
                                  device=None, log=None, cache_dir=None):
    """:func:`filter_paired_patients` with a disk-cached manifest.

    On a cache hit (manifest key matches the current roots/mtimes and ``rescan``
    is False) the ~345-patient scan is skipped and the stored list is returned.
    Otherwise it scans (loading each patient once), writes the manifest, and
    returns the result. The pairing logic itself is unchanged. Stored paths are
    intersected with the current ``patient_paths`` so a stale-but-key-matching
    manifest never returns a path no longer present.
    """
    if cache_path is None:
        cache_path = default_paired_cache_path()
    if not rescan:
        cached = load_paired_cache(cache_path, patient_paths)
        if cached is not None:
            present = set(patient_paths)
            kept = [p for p in cached if p in present]
            if log is not None:
                log("Paired patients: reused cached manifest (%d paired) from %s."
                    % (len(kept), cache_path))
            return kept
    # Only forward cache_dir when set so a monkeypatched filter_paired_patients with
    # the legacy (patient_paths, device, log) signature stays a drop-in (tests).
    if cache_dir is None:
        paired = filter_paired_patients(patient_paths, device=device, log=log)
    else:
        paired = filter_paired_patients(patient_paths, device=device, log=log, cache_dir=cache_dir)
    try:
        write_paired_cache(cache_path, patient_paths, paired)
        if log is not None:
            log("Paired patients: wrote manifest to %s." % (cache_path,))
    except OSError as exc:  # don't fail training if outputs/ isn't writable
        if log is not None:
            log("Paired patients: could not write manifest (%s); continuing." % (exc,))
    return paired


def make_patient_split(num_patients, val_fraction, seed):
    """Deterministic by-patient holdout split of ``num_patients`` patients.

    Returns ``(train_indices, val_indices)`` as plain Python lists. Guarantees at
    least one train patient. When ``num_patients > 1`` it also guarantees at least
    one val patient. When ``num_patients == 1`` returns ``([0], [0])`` -- a signal
    to callers to fall back to a within-patient (depth) split.
    """
    n = int(num_patients)
    if n <= 0:
        raise ValueError("num_patients must be >= 1")
    if n == 1:
        return [0], [0]
    indices = np.arange(n)
    rng = np.random.RandomState(int(seed))
    rng.shuffle(indices)
    n_val = max(1, int(round(n * float(val_fraction))))
    n_val = min(n_val, n - 1)  # keep >= 1 train patient
    val = sorted(int(i) for i in indices[:n_val])
    train = sorted(int(i) for i in indices[n_val:])
    return train, val


def make_patient_split3(num_patients, val_fraction, test_fraction, seed):
    """Deterministic three-way by-patient split (train / val / test).

    Returns ``(train_indices, val_indices, test_indices)`` as plain Python lists.
    The TEST set is a hard holdout: test patients appear in neither train nor val,
    so they never leak into training. The shuffle reuses the same RNG approach as
    :func:`make_patient_split` (``np.random.RandomState(seed)`` then ``shuffle``),
    so for a given ``(num_patients, seed)`` the shuffled order is identical -- this
    makes the 3-way split nested-consistent with the 2-way split's *ordering*
    (test patients are carved from the same shuffled tail the 2-way split assigned
    to val), though the index sets themselves differ by construction.

    Rounding / degradation rules (correctness of disjoint by-patient sets is the
    priority; documented here so callers can reason about tiny cohorts):

      * ``num_patients == 1`` -> returns ``([0], [0], [])`` -- a signal to callers
        to fall back to a within-patient (depth) split, mirroring
        :func:`make_patient_split`. No test set is possible.
      * ``num_patients == 2`` -> 1 train, 1 val, 0 test. There are not enough
        patients to also hold out a disjoint test patient, so the test set is empty
        and a warning-worthy situation is left to the caller to log.
      * ``num_patients >= 3`` -> at least 1 train, 1 val, and 1 test patient are
        guaranteed regardless of the requested fractions. Counts are computed as
        ``round(n * fraction)`` then clamped: test is taken first (>=1, but never so
        large it would starve val+train of one patient each), then val (>=1, never
        starving train), leaving the remainder as train (always >=1).

    The split is deterministic given ``(num_patients, val_fraction, test_fraction,
    seed)``.
    """
    n = int(num_patients)
    if n <= 0:
        raise ValueError("num_patients must be >= 1")
    if n == 1:
        return [0], [0], []
    indices = np.arange(n)
    rng = np.random.RandomState(int(seed))
    rng.shuffle(indices)
    if n == 2:
        # Not enough patients for a disjoint test holdout: 1 train / 1 val / 0 test.
        val = sorted(int(i) for i in indices[:1])
        train = sorted(int(i) for i in indices[1:])
        return train, val, []
    # n >= 3: guarantee at least one patient in each of train/val/test.
    n_test = int(round(n * float(test_fraction)))
    # Leave at least one patient each for train and val (so test <= n - 2), >= 1.
    n_test = max(1, min(n_test, n - 2))
    n_val = int(round(n * float(val_fraction)))
    # Val gets >= 1 and must leave >= 1 train after both holdouts: val <= n - n_test - 1.
    n_val = max(1, min(n_val, n - n_test - 1))
    test = sorted(int(i) for i in indices[:n_test])
    val = sorted(int(i) for i in indices[n_test:n_test + n_val])
    train = sorted(int(i) for i in indices[n_test + n_val:])
    return train, val, test


def write_split_json(save_dir, task, patient_paths, train_idx, val_idx, test_idx,
                     val_fraction, test_fraction, seed):
    """Persist the by-patient split to ``<save_dir>/split.json`` for reproducibility.

    Records the patient PATHS (not just indices) in each of train/val/test plus the
    ``val_fraction`` / ``test_fraction`` / ``seed`` used, so evaluation can reload the
    exact held-out set rather than only recomputing it from fractions+seed (which is
    fragile if the dataset changed since training). Returns the written path. Best
    effort: an :class:`OSError` is swallowed so training is never blocked by a
    non-writable output dir.

    Schema (``split.json``)::

        {
          "task": "diff2d",
          "val_fraction": 0.2,
          "test_fraction": 0.1,
          "seed": 42,
          "num_patients": 17,
          "train": ["/path/to/patientA", ...],   # patient folder paths
          "val":   ["/path/to/patientB", ...],
          "test":  ["/path/to/patientC", ...]
        }
    """
    def _paths(idx):
        return [str(patient_paths[i]) for i in idx]

    payload = {
        "task": task,
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
        "seed": int(seed),
        "num_patients": len(patient_paths),
        "train": _paths(train_idx),
        "val": _paths(val_idx),
        "test": _paths(test_idx),
    }
    path = os.path.join(save_dir, "split.json")
    try:
        os.makedirs(save_dir, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except OSError:
        return None
    return path


def make_depth_split(seed, val_fraction, num_positions=128):
    """Split a grid of normalized depth positions into (train, val) pools.

    Returns two 1-D numpy arrays of floats in [0, 1). The split is a deterministic
    shuffle so train and val never overlap.
    """
    positions = np.linspace(0.0, 1.0, num=num_positions, endpoint=False)
    rng = np.random.RandomState(int(seed))
    rng.shuffle(positions)
    n_val = max(1, int(round(len(positions) * float(val_fraction))))
    n_val = min(n_val, len(positions) - 1)
    return positions[n_val:], positions[:n_val]


def _slice_at(volume, t, size, axis=0):
    """Extract a 2D slice at normalized position ``t`` in [0,1) along ``axis``,
    resized to size x size. ``volume`` is a (Z, Y, X) tensor.

    ``axis`` selects the orthogonal plane through the volume:
      0 -> XY (axial,    slice index along Z) -> (Y, X)
      1 -> XZ (coronal,  slice index along Y) -> (Z, X)
      2 -> YZ (sagittal, slice index along X) -> (Z, Y)
    Returns a (1, size, size) tensor.
    """
    n = volume.shape[axis]
    idx = int(min(n - 1, max(0, round(t * (n - 1)))))
    if axis == 0:
        sl = volume[idx, :, :]
    elif axis == 1:
        sl = volume[:, idx, :]
    else:
        sl = volume[:, :, idx]
    sl = sl.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    sl = F.interpolate(sl, size=(size, size), mode="bilinear", align_corners=False)
    return sl.squeeze(0)  # (1,size,size)


def _apply_aug_single(augment, tensor, key="img"):
    """Apply a built geometric augment callable to one channel-first tensor.

    ``augment`` is a callable returned by :func:`src.training.utils.augment.build_aug_*`
    (or None for a no-op). It operates on a dict of tensors; here we wrap the single
    volume under ``key`` and unwrap the result. ``None`` returns ``tensor`` unchanged.
    """
    if augment is None:
        return tensor
    return augment({key: tensor})[key]


def _apply_aug_pair(augment, a, b, keys=("nac", "ac")):
    """Apply ``augment`` to a paired (a, b) sample with SYNCHRONIZED geometry.

    Both halves go into one dict so the dictionary transform draws a single set of
    random params and applies it identically to each key -- preserving NAC/AC
    spatial correspondence. ``None`` returns the inputs unchanged.
    """
    if augment is None:
        return a, b
    out = augment({keys[0]: a, keys[1]: b})
    return out[keys[0]], out[keys[1]]


def sample_slices(volumes, pool, batch_size, size, rng, augment=None):
    """Sample a batch of single-channel 2D slices pooled across ``volumes``.

    ``volumes`` is a list of (Z,Y,X) tensors (Nones skipped). ``pool`` is the
    train/val normalized-position array. Each slice is taken from a randomly
    chosen orthogonal plane (XY / XZ / YZ) so the 2D model sees all three
    viewing aspects of the volume. Returns (batch_size, 1, size, size).

    ``augment`` is an optional built geometric transform (see
    :mod:`src.training.utils.augment`); ``None`` (default) is current behavior.
    """
    vols = [v for v in volumes if v is not None]
    if not vols:
        raise ValueError("No volumes available to sample from.")
    out = []
    for _ in range(batch_size):
        v = vols[rng.randint(0, len(vols))]
        axis = int(rng.randint(0, 3))  # 0=XY axial, 1=XZ coronal, 2=YZ sagittal
        t = float(pool[rng.randint(0, len(pool))])
        out.append(_apply_aug_single(augment, _slice_at(v, t, size, axis)))
    return torch.stack(out, dim=0)


def sample_pairs(nac, ac, pool, batch_size, size, rng, augment=None):
    """Sample paired (NAC, AC) 2D slices at matching plane + normalized position.

    Each sample picks one orthogonal plane (XY / XZ / YZ) and one normalized
    position, then slices BOTH volumes the same way so the NAC/AC pair stays
    spatially corresponding. Returns two tensors each (batch_size, 1, size, size).

    ``augment`` (optional built geometric transform) is applied with a SINGLE
    random draw shared across the NAC and AC halves so the pair stays
    spatially corresponding; ``None`` (default) is current behavior.
    """
    if nac is None or ac is None:
        raise ValueError("Both NAC and AC volumes are required for NAC->AC training.")
    nac_b, ac_b = [], []
    for _ in range(batch_size):
        axis = int(rng.randint(0, 3))  # same plane for both halves of the pair
        t = float(pool[rng.randint(0, len(pool))])
        n_sl = _slice_at(nac, t, size, axis)
        a_sl = _slice_at(ac, t, size, axis)
        n_sl, a_sl = _apply_aug_pair(augment, n_sl, a_sl)
        nac_b.append(n_sl)
        ac_b.append(a_sl)
    return torch.stack(nac_b, dim=0), torch.stack(ac_b, dim=0)


def _resize_volume(volume, size):
    """Resize a (Z,Y,X) tensor to a (size,size,size) cube -> (1,size,size,size).

    Follow-up for anisotropic crops: accept ``size`` as a (dz,dy,dx) tuple and
    pass it straight to ``F.interpolate(size=...)`` (samplers would thread the
    tuple through). Cube-only for now per the B2 config.
    """
    vol = volume.unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)
    vol = F.interpolate(vol, size=(size, size, size), mode="trilinear", align_corners=False)
    return vol.squeeze(0)  # (1,size,size,size)


def _rand_crop_window(rng, min_frac=0.7):
    """A random [start, end) sub-window covering a fraction in [min_frac, 1] of an
    axis, expressed in normalized [0, 1) coordinates so it can be applied to two
    volumes of differing shape (NAC/AC) identically."""
    frac = float(min_frac) + (1.0 - float(min_frac)) * float(rng.rand())
    start = (1.0 - frac) * float(rng.rand())
    return start, start + frac


def _crop_normalized(volume, windows):
    """Crop a (Z,Y,X) tensor to per-axis normalized [start, end) ``windows``."""
    out = volume
    for axis, (s, e) in enumerate(windows):
        n = out.shape[axis]
        a = int(round(s * n))
        b = max(a + 1, int(round(e * n)))
        b = min(b, n)
        a = min(a, b - 1)
        idx = [slice(None)] * out.ndim
        idx[axis] = slice(a, b)
        out = out[tuple(idx)]
    return out


def draw_volume_aug(rng, min_frac=0.7):
    """Draw one set of 3D augmentation parameters (shared across a NAC/AC pair so
    the pair stays spatially corresponding): a normalized crop window per axis,
    a flip flag per spatial axis, and a 0/90/180/270-degree axial rotation."""
    return {
        "windows": [_rand_crop_window(rng, min_frac) for _ in range(3)],
        "flips": [bool(rng.rand() < 0.5) for _ in range(3)],  # Z, Y, X
        "rot_k": int(rng.randint(0, 4)),                      # 90 deg steps in the Y-X (axial) plane
    }


def apply_volume_aug(volume, size, params):
    """Apply ``params`` (from :func:`draw_volume_aug`) to a (Z,Y,X) volume:
    normalized crop -> resize to size^3 -> per-axis flips -> axial rot90.
    Returns (1, size, size, size). Output shape is invariant to ``params``."""
    v = _crop_normalized(volume, params["windows"])
    v = _resize_volume(v, size)  # (1, S, S, S)
    flip_dims = [d + 1 for d, f in enumerate(params["flips"]) if f]  # spatial dims are 1,2,3
    if flip_dims:
        v = torch.flip(v, dims=flip_dims)
    if params["rot_k"]:
        v = torch.rot90(v, params["rot_k"], dims=[2, 3])  # rotate within the Y-X plane (equal size)
    return v


def _band_crop(volume, size, phase, val_fraction):
    """Resize the train/val depth band of a (Z,Y,X) volume to a size^3 cube.

    For the common single-patient case we reserve the top ``val_fraction`` of the
    depth axis for validation and the remainder for training, so the two bands
    never overlap. Returns (1, size, size, size).
    """
    z_dim = volume.shape[0]
    split = int(round(z_dim * (1.0 - float(val_fraction))))
    split = min(max(split, 1), z_dim - 1)
    if phase == "val":
        z0, z1 = split, z_dim
    else:
        z0, z1 = 0, split
    if z1 - z0 < 1:
        z0, z1 = 0, z_dim
    return _resize_volume(volume[z0:z1, :, :], size)


def sample_pair_volumes(nac, ac, batch_size, size, phase, val_fraction, rng, geo_aug=None):
    """Sample paired (NAC, AC) 3D crops, split along depth into train/val bands.

    Returns two tensors each (batch_size, 1, size, size, size).

    ``geo_aug`` is an optional built 3D geometric transform (see
    :mod:`src.training.utils.augment`) applied per-sample with a SINGLE shared
    random draw across the NAC/AC pair (preserving correspondence); ``None``
    (default) is current behavior. The single-patient depth-band split is kept;
    geometric augmentation is layered on top of the resized band crop.
    """
    if nac is None or ac is None:
        raise ValueError("Both NAC and AC volumes are required for NAC->AC 3D training.")
    nac_list, ac_list = [], []
    for _ in range(batch_size):
        n = _band_crop(nac, size, phase, val_fraction)
        a = _band_crop(ac, size, phase, val_fraction)
        n, a = _apply_aug_pair(geo_aug, n, a)
        nac_list.append(n)
        ac_list.append(a)
    return torch.stack(nac_list, dim=0), torch.stack(ac_list, dim=0)


def sample_pair_volumes_full(nac, ac, batch_size, size, rng, augment=False, geo_aug=None):
    """Sample paired (NAC, AC) 3D crops over the FULL depth (no band split).

    Used in the multi-patient diffusion case where train/val separation comes
    from a by-patient holdout. Returns two (batch_size, 1, size, size, size).

    With ``augment=True`` each sample gets independent random crop/flip/rotation
    (the SAME transform for its NAC and AC halves), so a small pool of patients
    yields diverse 3D examples instead of one fixed resized cube per patient.
    Validation should pass ``augment=False`` for a stable, comparable metric.

    ``geo_aug`` is an optional built MONAI geometric transform layered ON TOP of
    the legacy crop/flip/rot90 (shared random draw across NAC/AC); ``None`` keeps
    the prior behavior.
    """
    if nac is None or ac is None:
        raise ValueError("Both NAC and AC volumes are required for NAC->AC 3D training.")
    nac_list, ac_list = [], []
    for _ in range(batch_size):
        if augment:
            p = draw_volume_aug(rng)
            n = apply_volume_aug(nac, size, p)
            a = apply_volume_aug(ac, size, p)
        else:
            n = _resize_volume(nac, size)
            a = _resize_volume(ac, size)
        n, a = _apply_aug_pair(geo_aug, n, a)
        nac_list.append(n)
        ac_list.append(a)
    return torch.stack(nac_list, dim=0), torch.stack(ac_list, dim=0)


def sample_volumes_full(volumes, batch_size, size, rng, augment=False, geo_aug=None):
    """Sample single-channel 3D crops over the FULL depth of ``volumes``.

    Used in the multi-patient case where train/val separation comes from a
    by-patient holdout, so no within-volume depth band split is needed.
    Returns (batch_size, 1, size, size, size).

    With ``augment=True`` each sample gets a random crop/flip/rotation, turning a
    small patient pool into diverse 3D examples. Validation should pass
    ``augment=False``.

    ``geo_aug`` is an optional built MONAI geometric transform layered ON TOP of
    the legacy crop/flip/rot90; ``None`` keeps the prior behavior.
    """
    vols = [v for v in volumes if v is not None]
    if not vols:
        raise ValueError("No volumes available to sample from.")
    out = []
    for _ in range(batch_size):
        v = vols[rng.randint(0, len(vols))]
        if augment:
            cube = apply_volume_aug(v, size, draw_volume_aug(rng))
        else:
            cube = _resize_volume(v, size)
        out.append(_apply_aug_single(geo_aug, cube))
    return torch.stack(out, dim=0)


def sample_volumes(volumes, batch_size, size, phase, val_fraction, rng, geo_aug=None):
    """Sample single-channel 3D crops pooled across ``volumes`` (e.g. NAC + AC).

    Mirrors :func:`sample_slices` but in 3D, using the same depth-band train/val
    split as :func:`sample_pair_volumes`. Returns (batch_size, 1, size, size, size).

    ``geo_aug`` is an optional built 3D geometric transform applied per-sample on
    top of the resized depth-band crop; ``None`` (default) is current behavior.
    """
    vols = [v for v in volumes if v is not None]
    if not vols:
        raise ValueError("No volumes available to sample from.")
    out = [
        _apply_aug_single(geo_aug, _band_crop(vols[rng.randint(0, len(vols))], size, phase, val_fraction))
        for _ in range(batch_size)
    ]
    return torch.stack(out, dim=0)
