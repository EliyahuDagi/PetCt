"""Attach segmentation results into a dataset tree.

Given a dataset root, a source segmentation root, and a segmentation name,
this script iterates patient folders under the source root and moves all
contents of each patient's `Segmentation` directory into
`<dataset_root>/<patient>/Segmentation/<segmentation_name>/`.

Existing destination files are left untouched; missing dataset patients are
skipped to avoid creating unexpected entries.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def move_patient_segmentation(
    dataset_root: Path, source_root: Path, segmentation_name: str
) -> None:
    for patient_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        src_seg_dir = patient_dir / "Segmentation"
        if not src_seg_dir.is_dir():
            print(f"[skip] No Segmentation folder in {patient_dir.name}")
            continue

        dst_patient_dir = dataset_root / patient_dir.name
        if not dst_patient_dir.is_dir():
            print(f"[skip] Patient {patient_dir.name} not found in dataset root")
            continue

        dst_seg_dir = dst_patient_dir / "Segmentation" / segmentation_name
        dst_seg_dir.mkdir(parents=True, exist_ok=True)

        for item in sorted(src_seg_dir.iterdir()):
            dst_item = dst_seg_dir / item.name
            if dst_item.exists():
                print(
                    f"[skip] Destination exists for {item.name} in {patient_dir.name}/{segmentation_name}"
                )
                continue

            shutil.move(str(item), str(dst_item))
            print(
                f"[moved] {item.name} -> {patient_dir.name}/Segmentation/{segmentation_name}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move segmentations into dataset")
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Target dataset root containing patient folders",
    )
    parser.add_argument(
        "--segmentation-dir",
        required=True,
        type=Path,
        help="Source root containing patient segmentation folders",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Segmentation name to use under each patient",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset_dir
    source_root: Path = args.segmentation_dir
    segmentation_name: str = args.name

    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root not found: {dataset_root}")
    if not source_root.is_dir():
        raise SystemExit(f"Segmentation root not found: {source_root}")

    print(
        f"Moving segmentations from {source_root} to {dataset_root} under name '{segmentation_name}'"
    )
    move_patient_segmentation(dataset_root, source_root, segmentation_name)


if __name__ == "__main__":
    main()
