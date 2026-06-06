"""Shared data loading + train/val splitting for the training scripts.

Loads a patient's normalized volumes via the existing ``DicomModel`` and provides
deterministic train/val sampling for the three tasks:

  * AE (ae2d)        -> pooled 2D PET slices (NAC + AC) so both share one latent.
  * NAC->AC 2D       -> paired (NAC, AC) 2D slices at matching normalized depth.
  * NAC->AC 3D       -> paired (NAC, AC) 3D crops.

The split is deterministic given ``seed`` so train/val never overlap across calls.
With more than one usable patient the stages use a by-patient holdout
(:func:`make_patient_split`) and sample over each patient's full depth; with a
single patient they fall back to a within-patient split -- for 2D a grid of
normalized depth positions split into train/val pools, for 3D a contiguous
train/val depth band. Patients are enumerated via the torch-free
:mod:`src.training.dataset_index` and loaded lazily through
:class:`PatientVolumeCache` to bound memory when pooling across many patients.
"""

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


def load_patient_by_path(patient_path, device=None, load_ct=True, run_segmentation=True):
    """Load one patient's normalized CT / AC-PET / NAC-PET volumes by folder path.

    Returns a dict with torch tensors (or None) under keys ``ct``, ``pet_ac``,
    ``pet_nac`` plus ``spacing`` (dz,dy,dx) and ``origin`` (z,y,x). Each volume is
    shaped (Z, Y, X). ``DicomModel`` is imported lazily so the torch-free
    enumeration helpers stay importable on a host without the viewer deps.

    ``load_ct=False`` skips reading the (large) CT series and ``run_segmentation
    =False`` skips the segmentor -- the PET training pipelines pass both False,
    which cuts per-patient load time several-fold (CT I/O dominates the cost).
    """
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

    return {
        "ct": _to_tensor(dm.ct_volume),
        "pet_ac": _to_tensor(dm.pet_volume),
        "pet_nac": _to_tensor(dm.pet_nac_volume),
        "spacing": dm.get_voxel_spacing(),
        "origin": dm.get_origin(),
        "patient_path": dm.patient_list[0],
    }


def load_patient_volumes(data_dir, patient_index, device=None):
    """Load one patient's normalized volumes from a (single) dataset root.

    Backward-compatible single-root entry point: enumerates patients under
    ``data_dir`` and loads the one at ``patient_index``. Implemented on top of
    :func:`enumerate_patients` + :func:`load_patient_by_path`.
    """
    patients = enumerate_patients(data_dir)
    if patient_index >= len(patients):
        raise ValueError("patient_index out of range (have %d patients)" % len(patients))
    return load_patient_by_path(patients[patient_index], device=device)


class PatientVolumeCache:
    """Bounded LRU cache of loaded patient volume dicts.

    Loads patients lazily by index into ``patient_paths`` and keeps at most
    ``max_cached`` of them resident, evicting the least-recently-used. This
    bounds GPU/host memory when pooling across many patients.
    """

    def __init__(self, patient_paths, device=None, max_cached=4, load_ct=False, run_segmentation=False):
        self.patient_paths = list(patient_paths)
        self.device = device
        # Training pools are PET-only: skip CT + segmentation per load (default).
        self.load_ct = bool(load_ct)
        self.run_segmentation = bool(run_segmentation)
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

    def get(self, index):
        """Return the volumes dict for ``patient_paths[index]`` (loads lazily)."""
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        vols = load_patient_by_path(
            self.patient_paths[index], device=self.device,
            load_ct=self.load_ct, run_segmentation=self.run_segmentation,
        )
        self._cache[index] = vols
        self._cache.move_to_end(index)
        while len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)  # evict LRU
        return vols


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


def filter_paired_patients(patient_paths, device=None, log=None):
    """Return the subset of ``patient_paths`` that have BOTH NAC and AC PET.

    Used by the diffusion stages, which require paired data. Each patient is
    loaded once (on CPU by default to keep the check cheap; pass ``device`` to
    load onto the target device). ``log`` is an optional callable (e.g.
    ``logger.info``) used to report skips. Returns the list of kept paths.
    """
    paired = []
    skipped = 0
    for path in patient_paths:
        # Only PET presence matters for pairing; skip CT + segmentation so this
        # startup scan over every patient stays cheap.
        vols = load_patient_by_path(path, device=device, load_ct=False, run_segmentation=False)
        if vols.get("pet_nac") is not None and vols.get("pet_ac") is not None:
            paired.append(path)
        else:
            skipped += 1
            if log is not None:
                log("Skipping %s for diffusion: missing a NAC/AC pair." % (path,))
    if log is not None:
        log("Paired patients: %d kept, %d skipped (missing a pair)." % (len(paired), skipped))
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


def sample_slices(volumes, pool, batch_size, size, rng):
    """Sample a batch of single-channel 2D slices pooled across ``volumes``.

    ``volumes`` is a list of (Z,Y,X) tensors (Nones skipped). ``pool`` is the
    train/val normalized-position array. Each slice is taken from a randomly
    chosen orthogonal plane (XY / XZ / YZ) so the 2D model sees all three
    viewing aspects of the volume. Returns (batch_size, 1, size, size).
    """
    vols = [v for v in volumes if v is not None]
    if not vols:
        raise ValueError("No volumes available to sample from.")
    out = []
    for _ in range(batch_size):
        v = vols[rng.randint(0, len(vols))]
        axis = int(rng.randint(0, 3))  # 0=XY axial, 1=XZ coronal, 2=YZ sagittal
        t = float(pool[rng.randint(0, len(pool))])
        out.append(_slice_at(v, t, size, axis))
    return torch.stack(out, dim=0)


def sample_pairs(nac, ac, pool, batch_size, size, rng):
    """Sample paired (NAC, AC) 2D slices at matching plane + normalized position.

    Each sample picks one orthogonal plane (XY / XZ / YZ) and one normalized
    position, then slices BOTH volumes the same way so the NAC/AC pair stays
    spatially corresponding. Returns two tensors each (batch_size, 1, size, size).
    """
    if nac is None or ac is None:
        raise ValueError("Both NAC and AC volumes are required for NAC->AC training.")
    nac_b, ac_b = [], []
    for _ in range(batch_size):
        axis = int(rng.randint(0, 3))  # same plane for both halves of the pair
        t = float(pool[rng.randint(0, len(pool))])
        nac_b.append(_slice_at(nac, t, size, axis))
        ac_b.append(_slice_at(ac, t, size, axis))
    return torch.stack(nac_b, dim=0), torch.stack(ac_b, dim=0)


def _resize_volume(volume, size):
    """Resize a (Z,Y,X) tensor to a (size,size,size) cube -> (1,size,size,size)."""
    vol = volume.unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)
    vol = F.interpolate(vol, size=(size, size, size), mode="trilinear", align_corners=False)
    return vol.squeeze(0)  # (1,size,size,size)


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


def sample_pair_volumes(nac, ac, batch_size, size, phase, val_fraction, rng):
    """Sample paired (NAC, AC) 3D crops, split along depth into train/val bands.

    Returns two tensors each (batch_size, 1, size, size, size).
    """
    if nac is None or ac is None:
        raise ValueError("Both NAC and AC volumes are required for NAC->AC 3D training.")
    nac_b = torch.stack([_band_crop(nac, size, phase, val_fraction) for _ in range(batch_size)], dim=0)
    ac_b = torch.stack([_band_crop(ac, size, phase, val_fraction) for _ in range(batch_size)], dim=0)
    return nac_b, ac_b


def sample_pair_volumes_full(nac, ac, batch_size, size, rng):
    """Sample paired (NAC, AC) 3D crops over the FULL depth (no band split).

    Used in the multi-patient diffusion case where train/val separation comes
    from a by-patient holdout. Returns two (batch_size, 1, size, size, size).
    """
    if nac is None or ac is None:
        raise ValueError("Both NAC and AC volumes are required for NAC->AC 3D training.")
    nac_b = torch.stack([_resize_volume(nac, size) for _ in range(batch_size)], dim=0)
    ac_b = torch.stack([_resize_volume(ac, size) for _ in range(batch_size)], dim=0)
    return nac_b, ac_b


def sample_volumes_full(volumes, batch_size, size, rng):
    """Sample single-channel 3D crops over the FULL depth of ``volumes``.

    Used in the multi-patient case where train/val separation comes from a
    by-patient holdout, so no within-volume depth band split is needed.
    Returns (batch_size, 1, size, size, size).
    """
    vols = [v for v in volumes if v is not None]
    if not vols:
        raise ValueError("No volumes available to sample from.")
    out = [_resize_volume(vols[rng.randint(0, len(vols))], size) for _ in range(batch_size)]
    return torch.stack(out, dim=0)


def sample_volumes(volumes, batch_size, size, phase, val_fraction, rng):
    """Sample single-channel 3D crops pooled across ``volumes`` (e.g. NAC + AC).

    Mirrors :func:`sample_slices` but in 3D, using the same depth-band train/val
    split as :func:`sample_pair_volumes`. Returns (batch_size, 1, size, size, size).
    """
    vols = [v for v in volumes if v is not None]
    if not vols:
        raise ValueError("No volumes available to sample from.")
    out = [_band_crop(vols[rng.randint(0, len(vols))], size, phase, val_fraction) for _ in range(batch_size)]
    return torch.stack(out, dim=0)
