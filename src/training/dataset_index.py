"""Torch-free patient enumeration across one or more dataset roots.

This module intentionally imports only the standard library so it can run on a
Windows host that has no torch installed (the Train Viewer GUI calls it directly
to populate patient lists). It must NOT import ``DicomModel`` or any torch code.

The single-vs-many heuristic mirrors ``DicomModel.load_dataset`` in
``src/DataViewer/model.py`` (kept in sync by hand to preserve the torch-free
guarantee): a root whose immediate subdirs include a "patient marker" folder
(DICOM/CT/PT/PET/Segmentation/SECTRA) is itself treated as a single patient;
otherwise each immediate subdir is a patient.
"""

import os

PATIENT_MARKERS = {"dicom", "ct", "pt", "pet", "segmentation", "sectra"}


def _patients_in_root(root):
    """Return the list of patient folder paths for a single existing root.

    Replicates the DicomModel heuristic: if ``root`` directly contains a subdir
    whose lowercased basename is a patient marker, ``root`` itself is one
    patient; otherwise each immediate subdir is a patient (sorted).
    """
    subdirs = [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    subdir_names = {os.path.basename(d).lower() for d in subdirs}
    if subdir_names & PATIENT_MARKERS:
        # The selected folder is itself a patient root (e.g. it holds DICOM/).
        return [root]
    return sorted(subdirs)


def enumerate_patients(data_dirs, missing_ok=False):
    """Enumerate patient folder paths across one or more dataset roots.

    ``data_dirs`` may be a single ``str`` root or an iterable of ``str`` roots.

    Roots are processed in the given order; patients within a root are sorted.
    The returned list is flat and de-duplicated (a patient path that appears
    under two roots is kept only once, at its first occurrence).

    A non-existent root raises ``ValueError`` when ``missing_ok`` is False, or is
    silently skipped when ``missing_ok`` is True (the GUI calls with
    ``missing_ok=True``).
    """
    if isinstance(data_dirs, str):
        roots = [data_dirs]
    else:
        roots = list(data_dirs)

    out = []
    seen = set()
    for root in roots:
        if not os.path.exists(root):
            if missing_ok:
                continue
            raise ValueError("Dataset root not found: %r" % (root,))
        for patient in _patients_in_root(root):
            key = os.path.normcase(os.path.abspath(patient))
            if key in seen:
                continue
            seen.add(key)
            out.append(patient)
    return out
