"""Move patient segmentation files into a destination subdirectory.

This script walks patient folders under the source root and, when a
segmentation directory is found, moves `mask.npy` and `labels.json` into a
named destination subdirectory beneath that segmentation directory (e.g.,
`Segmentation/MONAI` or `Segmentation/TotalSegmentor`).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Root directory containing patient subfolders.
SOURCE_ROOT = Path(
    r"C:\Users\eli.dagi\Projects\PetCt\data\gdrive_downloads\1T4LsR5QOGwwFyGnoQFCXtGBJq4qf8HO2"
)

# Name of the segmentation directory under each patient folder.
SEGMENTATION_DIR_NAME = "Segmentation"

# Default destination directory name inside the segmentation folder.
DESTINATION_SUBDIR_NAME = "MONAI"

# Files to move into each MONAI subdirectory.
FILES_TO_MOVE = ["mask.npy", "labels.json"]


def move_files_to_monai(
    patient_dir: Path, segmentation_dir_name: str, destination_subdir_name: str
) -> None:
    """Move target files into the requested destination subdirectory.

    Skips gracefully if directories or files are missing. Existing destination
    files are left untouched to avoid accidental overwrite.
    """

    seg_dir = patient_dir / segmentation_dir_name
    if not seg_dir.is_dir():
        print(f"[skip] No {segmentation_dir_name} folder in {patient_dir.name}")
        return

    destination_dir = seg_dir / destination_subdir_name
    destination_dir.mkdir(exist_ok=True)

    for filename in FILES_TO_MOVE:
        src = seg_dir / filename
        dst = destination_dir / filename

        if not src.exists():
            print(f"[skip] Missing {filename} in {patient_dir.name}")
            continue

        if dst.exists():
            print(f"[skip] Destination exists for {filename} in {patient_dir.name}")
            continue

        shutil.move(str(src), str(dst))
        print(f"[moved] {filename} for {patient_dir.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Move segmentation artifacts (mask.npy, labels.json) into a "
            "destination subdirectory under each patient's segmentation folder."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=SOURCE_ROOT,
        help="Root directory containing patient subfolders (default: %(default)s)",
    )
    parser.add_argument(
        "--segmentation-dir",
        default=SEGMENTATION_DIR_NAME,
        help="Name of the segmentation directory under each patient (default: %(default)s)",
    )
    parser.add_argument(
        "--destination-subdir",
        default=DESTINATION_SUBDIR_NAME,
        help=(
            "Name of the destination subdirectory inside the segmentation "
            "directory (e.g., MONAI, TotalSegmentor). Default: %(default)s"
        ),
    )

    args = parser.parse_args()

    source_root: Path = args.source_root
    segmentation_dir_name: str = args.segmentation_dir
    destination_subdir_name: str = args.destination_subdir

    if not source_root.is_dir():
        raise SystemExit(f"Source root not found: {source_root}")

    patient_dirs = [p for p in source_root.iterdir() if p.is_dir()]
    if not patient_dirs:
        print("No patient directories found.")
        return

    print(
        "Processing "
        f"{len(patient_dirs)} patient folders under {source_root} "
        f"-> {segmentation_dir_name}/{destination_subdir_name}"
    )
    for patient_dir in sorted(patient_dirs):
        move_files_to_monai(
            patient_dir,
            segmentation_dir_name=segmentation_dir_name,
            destination_subdir_name=destination_subdir_name,
        )


if __name__ == "__main__":
    main()
