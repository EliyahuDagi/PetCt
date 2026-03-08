
import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path before local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.config import Config
from src.utils.segmentation_utils import SegmentationPredictor, TotalSegmentatorRunner
from src.DataViewer.model import DicomModel


def run_batch_segmentation(
    data_dir,
    backend="totalseg",
    bundle_name=None,
    task="total",
    device=None,
    fast=True,
    output_root=None,
    temp_dir=None,
    cache_dir=None,
    skip_existing=True,
):
    data_path = Path(data_dir)
    patient_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} patient directories in {data_dir}")

    if backend == "totalseg":
        runner = TotalSegmentatorRunner(
            task=task,
            device=device,
            fast=fast,
            cache_dir=cache_dir,
            temp_dir=temp_dir,
            output_root=output_root,
        )
        label_map = runner.get_label_map()

        for patient_dir in tqdm(patient_dirs, desc="TotalSegmentator"):
            try:
                changed = runner.run_patient(patient_dir, skip_existing=skip_existing)
                if changed:
                    seg_parent = (runner.output_root / patient_dir.name) if runner.output_root else patient_dir
                    seg_dir = seg_parent / Config.SEGMENTATION_DIR_NAME
                    seg_dir.mkdir(parents=True, exist_ok=True)
                    labels_path = seg_dir / "labels.json"
                    if not labels_path.exists():
                        with open(labels_path, "w") as f:
                            json.dump({int(k): v for k, v in label_map.items()}, f, indent=2)
            except Exception as e:
                print(f"Error processing {patient_dir.name}: {e}")
                import traceback

                traceback.print_exc()
    else:
        bundle = bundle_name or Config.SEGMENTATION_BUNDLE_NAME
        output_root = Path(output_root) if output_root else None
        output_dir_name = Config.SEGMENTATION_DIR_NAME

        print(f"Initializing MONAI bundle predictor: {bundle}...")
        predictor = SegmentationPredictor(bundle_name=bundle)

        for patient_dir in tqdm(patient_dirs, desc="MONAI Bundle"):
            try:
                seg_parent = output_root / patient_dir.name if output_root else patient_dir
                seg_dir = seg_parent / output_dir_name
                seg_dir.mkdir(parents=True, exist_ok=True)
                seg_file = seg_dir / "mask.npy"
                if skip_existing and seg_file.exists():
                    continue

                model = DicomModel()
                model.load_patient_data(str(patient_dir))

                if model.ct_volume is None:
                    print(f"No CT volume found for {patient_dir.name}")
                    continue

                spacing = model.get_voxel_spacing()
                mask = predictor.predict(model.ct_volume, spacing)
                label_map = predictor.get_label_map()

                np.save(seg_file, mask)
                if label_map:
                    labels_path = seg_dir / "labels.json"
                    with open(labels_path, "w") as f:
                        json.dump({int(k): v for k, v in label_map.items()}, f, indent=2)

            except Exception as e:
                print(f"Error processing {patient_dir.name}: {e}")
                import traceback

                traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root (patients as subfolders)")
    parser.add_argument("--backend", choices=["totalseg", "monai"],
                        default="totalseg", help="Which segmentation backend to use")
    parser.add_argument("--bundle", type=str, default=None, help="MONAI bundle name (backend=monai)")
    parser.add_argument("--task", type=str, default="total", help="TotalSegmentator task (e.g., total, total_mr, body)")
    parser.add_argument("--device", type=str, default=None, help="Device for TotalSegmentator (gpu, cpu, mps, gpu:0)")
    parser.add_argument("--fast", action="store_true", help="Use fast TotalSegmentator model (3mm)")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Optional root to write segmentations; defaults to patient folder")
    parser.add_argument("--temp_dir", type=str, default=None, help="Optional temp directory for intermediate files")
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional TotalSegmentator cache dir (weights)")
    parser.add_argument("--no_skip_existing", action="store_true", help="Process even if mask.npy exists")
    args = parser.parse_args()

    run_batch_segmentation(
        data_dir=args.data_dir,
        backend=args.backend,
        bundle_name=args.bundle,
        task=args.task,
        device=args.device,
        fast=args.fast,
        output_root=Path(args.output_root) if args.output_root else None,
        temp_dir=args.temp_dir,
        cache_dir=args.cache_dir,
        skip_existing=not args.no_skip_existing,
    )
