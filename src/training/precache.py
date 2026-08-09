"""One-time SSD pre-cache of normalized PET volumes for the training pipeline.

The dataset lives on a spinning HDD as many small DICOM files; loading one patient
is ~5-8 s of random I/O, which starves the GPU during training. This module
decodes + percentile-normalizes each patient's PET ONCE (via the same
``load_patient_by_path`` the training loop uses) and writes a compact per-patient
cache entry to the NVMe SSD, so a later training run reads it back in ~ms.

Layout per patient under ``cache_dir``::

    <hash>.npz    # uncompressed: pet_ac / pet_nac arrays in the chosen dtype
    <hash>.json   # sidecar: spacing, origin, patient_path, has_ac/has_nac, dtype,
                  #          source signature (dir mtime) for staleness, version

The cache stores **native-resolution** normalized volumes (no resize/crop) -- the
samplers resize/crop at sample time, so caching pre-resized data would lock in one
crop size. float16 (default) halves size/IO and is cast back to float32 on load.

CLI::

    python -m src.training.precache --data_dir <roots...> \
        --cache_dir /mnt/c/DeepTrainingData/PetCT

The loader side (``load_patient_by_path(cache_dir=...)`` in :mod:`src.training.data`)
is transparent: when ``cache_dir`` is unset behavior is byte-identical to before.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

from src.training.dataset_index import enumerate_patients

# Default cache location (fixed by the user). On the WSL training host
# ``C:\DeepTrainingData\PetCT`` is mounted at ``/mnt/c/DeepTrainingData/PetCT``.
DEFAULT_CACHE_DIR = "/mnt/c/DeepTrainingData/PetCT"

# Environment variable consulted as a convenience by both the precache CLI and the
# transparent loader path so a run can opt in without threading a flag everywhere.
CACHE_ENV_VAR = "PETCT_CACHE_DIR"

# Bump when the on-disk format changes OR when the upstream loader semantics change
# so stale entries are treated as a miss. v2: NAC series classifier now reads the
# DICOM CorrectedImage (0028,0051) tag first (was SeriesDescription/ImageType tokens
# only), so v1 entries built with the buggy classifier dropped NAC for many
# ACRIN/TCIA patients and MUST be rebuilt even though the source DICOM is unchanged.
CACHE_VERSION = 2

_VOLUME_KEYS = ("pet_ac", "pet_nac")


def resolve_cache_dir(cache_dir=None):
    """Resolve the effective cache dir from an explicit arg or the env var.

    Returns ``None`` (cache disabled) when neither is set or the value is empty,
    so the default pure-DICOM path is preserved.
    """
    if cache_dir is None:
        cache_dir = os.environ.get(CACHE_ENV_VAR)
    if not cache_dir:
        return None
    return str(cache_dir)


def cache_key(patient_path):
    """Stable hash of a patient's absolute, case-normalized path.

    Matches the de-dup key used by :func:`enumerate_patients` so the same patient
    always maps to the same cache entry regardless of how the path was spelled.
    """
    norm = os.path.normcase(os.path.abspath(patient_path))
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


def entry_paths(cache_dir, patient_path):
    """Return ``(npz_path, sidecar_json_path)`` for a patient under ``cache_dir``."""
    key = cache_key(patient_path)
    return (
        os.path.join(cache_dir, key + ".npz"),
        os.path.join(cache_dir, key + ".json"),
    )


def source_signature(patient_path):
    """A cheap source-freshness signature: the patient folder's mtime.

    Touching/rebuilding the patient folder changes its mtime, which lets the loader
    treat the cached entry as stale. Returns -1.0 when the path is gone.
    """
    try:
        return round(float(os.path.getmtime(patient_path)), 3)
    except OSError:
        return -1.0


def read_sidecar(cache_dir, patient_path):
    """Return the parsed sidecar dict for a patient, or ``None`` if absent/bad."""
    _, side = entry_paths(cache_dir, patient_path)
    if not os.path.exists(side):
        return None
    try:
        with open(side, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def is_fresh(cache_dir, patient_path, check_source=True):
    """True when a usable, version-matching cache entry exists for this patient.

    When ``check_source`` is True the entry must also match the current source
    signature (patient-dir mtime); pass False to accept any present entry (e.g. if
    the source HDD is offline at training time but the cache is trusted).
    """
    side = read_sidecar(cache_dir, patient_path)
    if side is None:
        return False
    if int(side.get("version", -1)) != CACHE_VERSION:
        return False
    npz, _ = entry_paths(cache_dir, patient_path)
    if not os.path.exists(npz):
        return False
    if check_source and side.get("source_sig") != source_signature(patient_path):
        return False
    return True


def write_entry(cache_dir, vols, dtype="float16"):
    """Atomically write one patient's cache entry from a loaded ``vols`` dict.

    ``vols`` is the dict returned by ``load_patient_by_path`` (tensors or numpy or
    None). Volumes are stored in ``dtype`` (float16 default). Returns the number of
    bytes written (npz + sidecar). Writes go to ``*.tmp`` then ``os.replace`` so a
    crash never leaves a partial entry a later run would treat as valid.
    """
    os.makedirs(cache_dir, exist_ok=True)
    patient_path = vols.get("patient_path")
    if patient_path is None:
        raise ValueError("Cannot cache a patient dict without 'patient_path'.")
    npz_path, side_path = entry_paths(cache_dir, patient_path)

    np_dtype = np.float16 if str(dtype) == "float16" else np.float32
    arrays = {}
    flags = {}
    for k in _VOLUME_KEYS:
        v = vols.get(k)
        if v is None:
            flags["has_" + k.split("_")[-1]] = False
            continue
        arr = _to_numpy(v).astype(np_dtype, copy=False)
        arrays[k] = arr
        flags["has_" + k.split("_")[-1]] = True

    # Atomic npz write (tmp + replace). np.savez wants a file object/path; use a
    # tmp path in the SAME dir so os.replace is atomic (same filesystem). The npz is
    # written BEFORE the sidecar so a present sidecar (which gates is_fresh) always
    # has a present npz -- a crash between the two leaves only a stale npz, never a
    # sidecar pointing at a missing array. Tmp files are removed on any failure so a
    # partial write never lingers.
    npz_tmp = npz_path + ".tmp"
    _atomic_write(npz_tmp, npz_path, lambda f: np.savez(f, **arrays), binary=True)

    sidecar = {
        "version": CACHE_VERSION,
        "patient_path": patient_path,
        "spacing": _jsonable(vols.get("spacing")),
        "origin": _jsonable(vols.get("origin")),
        "dtype": str(dtype),
        "source_sig": source_signature(patient_path),
        **flags,
    }
    side_tmp = side_path + ".tmp"
    _atomic_write(side_tmp, side_path, lambda f: json.dump(sidecar, f), binary=False)

    total = os.path.getsize(npz_path) + os.path.getsize(side_path)
    return int(total)


def _atomic_write(tmp_path, final_path, writer, binary):
    """Write via ``writer(file)`` to ``tmp_path`` then ``os.replace`` to ``final_path``.

    Removes ``tmp_path`` if the write fails so a crash never leaves a partial file.
    """
    mode = "wb" if binary else "w"
    kwargs = {} if binary else {"encoding": "utf-8"}
    try:
        with open(tmp_path, mode, **kwargs) as f:
            writer(f)
        os.replace(tmp_path, final_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def load_entry(cache_dir, patient_path, device=None, to_tensor=True):
    """Load a cached patient entry back into a ``load_patient_by_path``-shaped dict.

    Arrays are cast to float32 (the training dtype) and, when ``to_tensor`` is
    True (the default), wrapped as torch tensors and moved to ``device``. Returns
    ``None`` when no usable entry exists (caller falls back to DICOM).
    """
    side = read_sidecar(cache_dir, patient_path)
    if side is None:
        return None
    npz_path, _ = entry_paths(cache_dir, patient_path)
    if not os.path.exists(npz_path):
        return None

    with np.load(npz_path) as data:
        ac = data["pet_ac"].astype(np.float32) if "pet_ac" in data.files else None
        nac = data["pet_nac"].astype(np.float32) if "pet_nac" in data.files else None

    out = {
        "ct": None,
        "pet_ac": ac,
        "pet_nac": nac,
        "spacing": _tupleize(side.get("spacing")),
        "origin": _tupleize(side.get("origin")),
        "patient_path": side.get("patient_path", patient_path),
    }
    if to_tensor:
        import torch

        for k in _VOLUME_KEYS:
            if out[k] is not None:
                t = torch.from_numpy(out[k])
                out[k] = t.to(device) if device is not None else t
    return out


def sidecar_has_pair(cache_dir, patient_path):
    """True when the cached sidecar reports BOTH NAC and AC present (no array read)."""
    side = read_sidecar(cache_dir, patient_path)
    if side is None:
        return False
    return bool(side.get("has_ac")) and bool(side.get("has_nac"))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_numpy(v):
    """Coerce a torch tensor / numpy array / array-like to a numpy array."""
    if hasattr(v, "detach"):  # torch.Tensor
        return v.detach().cpu().numpy()
    return np.asarray(v)


def _jsonable(v):
    if v is None:
        return None
    return [float(x) for x in v]


def _tupleize(v):
    if v is None:
        return None
    return tuple(float(x) for x in v)


# ---------------------------------------------------------------------------
# CLI: populate the cache
# ---------------------------------------------------------------------------
def _human_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return "%.1f %s" % (n, unit)
        n /= 1024.0
    return "%.1f PB" % n


def _precache_one(patient_path, cache_dir, dtype, force, loader):
    """Build (or skip) one patient's cache entry. Returns (status, bytes).

    ``status`` is one of "cached" / "skipped" / "empty" / "failed".
    """
    if not force and is_fresh(cache_dir, patient_path):
        return "skipped", 0
    try:
        vols = loader(patient_path, device=None, load_ct=False, run_segmentation=False)
    except Exception:  # surface as a per-patient failure, keep going
        return "failed", 0
    if vols.get("pet_ac") is None and vols.get("pet_nac") is None:
        return "empty", 0
    nbytes = write_entry(cache_dir, vols, dtype=dtype)
    return "cached", nbytes


def precache(data_dir, cache_dir=None, dtype="float16", workers=2, force=False,
             log=print, loader=None):
    """Populate the SSD cache for every enumerated patient under ``data_dir``.

    ``loader`` defaults to ``load_patient_by_path`` (imported lazily so the
    enumeration helpers stay torch-free for callers that only resolve the cache
    dir). Returns a summary dict.
    """
    cache_dir = resolve_cache_dir(cache_dir) or DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    if loader is None:
        from src.training.data import load_patient_by_path as loader  # noqa: PLW0127

    patients = enumerate_patients(data_dir, missing_ok=True)
    summary = {"cached": 0, "skipped": 0, "empty": 0, "failed": 0, "bytes": 0}
    n = len(patients)
    log("Pre-caching %d patient(s) -> %s (dtype=%s, workers=%d, force=%s)"
        % (n, cache_dir, dtype, workers, force))

    def _do(idx_path):
        idx, path = idx_path
        status, nbytes = _precache_one(path, cache_dir, dtype, force, loader)
        return idx, path, status, nbytes

    items = list(enumerate(patients))
    if int(workers) > 1 and n > 1:
        results = _run_pool(_do, items, int(workers))
    else:
        results = (_do(it) for it in items)

    for idx, path, status, nbytes in results:
        summary[status] = summary.get(status, 0) + 1
        summary["bytes"] += nbytes
        log("[%d/%d] %s  %s  (%s, running total %s)"
            % (idx + 1, n, status.upper(), path,
               _human_bytes(nbytes), _human_bytes(summary["bytes"])))

    log("Done: %d cached / %d skipped / %d empty / %d failed; total %s in %s"
        % (summary["cached"], summary["skipped"], summary["empty"],
           summary["failed"], _human_bytes(summary["bytes"]), cache_dir))
    return summary


def _run_pool(fn, items, workers):
    """Run ``fn`` over ``items`` on a small thread pool, yielding in submit order.

    Threading (not multiprocessing) mirrors the prefetch cache: pydicom/numpy I/O
    releases the GIL so reads overlap, and we avoid pickling. Kept small by default
    because the source is a single HDD (over-parallelizing thrashes the seek head).
    """
    from concurrent.futures import ThreadPoolExecutor

    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(fn, items):
            out.append(res)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pre-cache normalized PET volumes onto the SSD.")
    parser.add_argument("--data_dir", required=True, nargs="+",
                        help="One or more dataset roots / patient folders (HDD source).")
    parser.add_argument("--cache_dir", default=None,
                        help="SSD cache dir (default: $%s or %s)." % (CACHE_ENV_VAR, DEFAULT_CACHE_DIR))
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16",
                        help="Stored array dtype (float16 halves size/IO; cast to float32 on load).")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel loader threads (small: the source is one HDD).")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild entries even if a fresh one already exists.")
    args = parser.parse_args(argv)

    summary = precache(
        args.data_dir, cache_dir=args.cache_dir, dtype=args.dtype,
        workers=args.workers, force=args.force,
    )
    # Non-zero exit if every patient failed (lets a wrapper script detect a bad run).
    if summary["failed"] and not (summary["cached"] or summary["skipped"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
