"""Pull one whole-body coronal slice of CT, non-corrected PET and corrected PET.

Runs in WSL (needs pydicom and torch); writes a small .npz that the Windows-side
scripts/make_scheme_figures.py turns into docs/figures/fig_problem.png. Split in two
because the WSL environment has no matplotlib and the Windows one has no DICOM stack.

The CT sits on its own grid (here 480 x 512 x 512 at 2.5 mm) and the PET on another
(369 x 256 x 256), so the CT is resampled onto the PET grid through physical millimetres
-- nearest voxel, which is plenty for a picture -- and all three panels then show the
same anatomy at the same scale.

Usage (inside WSL, from the repository root):
  ~/petct/.venv/bin/python scripts/_extract_problem_slices.py
"""
import os
import sys

import numpy as np

PATIENT = os.environ.get(
    "PATIENT", "/mnt/d/DeepTrainingData/Project/PetCt/Bladder 13.11.25/31128984")
OUT = os.environ.get("OUT", "outputs/eval/fig_problem_slices.npz")


def spacing_origin(meta):
    """(dz, dy, dx), (oz, oy, ox) in millimetres from a list of DICOM slices."""
    ds = meta[0]
    dy, dx = (float(v) for v in ds.PixelSpacing)
    if len(meta) > 1:
        dz = abs(float(meta[1].ImagePositionPatient[2])
                 - float(meta[0].ImagePositionPatient[2]))
    else:
        dz = float(getattr(ds, "SliceThickness", 1.0))
    p = ds.ImagePositionPatient
    return (dz, dy, dx), (float(p[2]), float(p[1]), float(p[0]))


def main():
    from src.DataViewer.model import DicomModel
    from src.training.data import normalize_volume

    dm = DicomModel()
    dm.patient_list = [PATIENT]
    dm.current_patient_index = 0
    dm.load_patient_data(PATIENT, load_ct=True, run_segmentation=False)

    ct = np.asarray(dm.ct_volume, dtype=np.float32)          # raw Hounsfield units
    ac = normalize_volume(np.asarray(dm.pet_volume, dtype=np.float32))
    nac = normalize_volume(np.asarray(dm.pet_nac_volume, dtype=np.float32))
    if ct is None or ac is None or nac is None:
        sys.exit("this patient is missing one of CT / AC / NAC")

    ct_sp, ct_or = spacing_origin(dm.ct_metadata)
    pet_sp, pet_or = spacing_origin(dm.pet_metadata)
    print("CT ", ct.shape, "spacing", ct_sp, "origin", ct_or)
    print("PET", ac.shape, "spacing", pet_sp, "origin", pet_or)

    # The coronal slice to show: the one carrying the most corrected uptake, which lands
    # mid-body rather than on an arm or on air.
    y = int(np.argmax(ac.sum(axis=(0, 2))))
    print("coronal row", y, "of", ac.shape[1])

    # Physical millimetres of every PET voxel on that coronal plane.
    zc = pet_or[0] + np.arange(ac.shape[0]) * pet_sp[0]
    xc = pet_or[2] + np.arange(ac.shape[2]) * pet_sp[2]
    yc = pet_or[1] + y * pet_sp[1]

    # ... and the nearest CT voxel for each of them.
    def to_idx(mm, origin, step, n):
        return np.clip(np.rint((mm - origin) / step).astype(np.int64), 0, n - 1)

    # Signed, because the CT slice index may run down the body while the PET runs up it.
    z_step = ct_sp[0]
    if len(dm.ct_metadata) > 1:
        z_step = (float(dm.ct_metadata[1].ImagePositionPatient[2])
                  - float(dm.ct_metadata[0].ImagePositionPatient[2]))
    kz = to_idx(zc, ct_or[0], z_step, ct.shape[0])
    ky = to_idx(np.array([yc]), ct_or[1], ct_sp[1], ct.shape[1])[0]
    kx = to_idx(xc, ct_or[2], ct_sp[2], ct.shape[2])
    ct_slice = ct[np.ix_(kz, [ky], kx)][:, 0, :]

    # The PET is 700 mm wide and the CT only 500 mm, so the outer PET columns have no CT
    # under them and would smear the CT's edge voxel sideways. Crop all three panels to
    # the columns the CT actually covers, so the three pictures stay comparable.
    inside = np.where((xc >= ct_or[2]) & (xc <= ct_or[2] + (ct.shape[2] - 1) * ct_sp[2]))[0]
    lo, hi = int(inside[0]), int(inside[-1]) + 1
    print("columns kept", lo, "..", hi, "of", ac.shape[2])
    ct_slice = ct_slice[:, lo:hi]

    np.savez_compressed(
        OUT,
        ct=ct_slice.astype(np.float32),
        nac=nac[:, y, lo:hi].astype(np.float32),
        ac=ac[:, y, lo:hi].astype(np.float32),
        patient=os.path.basename(PATIENT),
        row=y,
        pet_spacing=np.array(pet_sp, dtype=np.float32),
        pet_origin=np.array(pet_or, dtype=np.float32),
    )
    print("wrote", OUT, "panels", ct_slice.shape)


if __name__ == "__main__":
    main()
