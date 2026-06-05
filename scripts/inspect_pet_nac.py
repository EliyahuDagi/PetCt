import argparse
import os
from collections import defaultdict

import pydicom


def _tokens(val):
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        raw = [str(v) for v in val]
    else:
        raw = str(val).replace("\\", " ").replace("/", " ").split()
    return [t.strip().upper() for t in raw if t]


def classify_pet_series(ds):
    corrected = ds.get((0x0028, 0x0051))
    if corrected is not None and corrected.value:
        attrs = corrected.value
        if isinstance(attrs, str):
            attrs = [attrs]
        attrs = [str(a).upper() for a in attrs]
        if "ATTN" in attrs:
            return "ac", "CorrectedImage"
        return "nac", "CorrectedImage"

    desc = str(getattr(ds, "SeriesDescription", ""))
    img_type = getattr(ds, "ImageType", None)
    tokens = set(_tokens(desc) + _tokens(img_type))
    if "NAC" in tokens or "NON-AC" in tokens or "NONAC" in tokens or "UNCORRECTED" in tokens:
        return "nac", "SeriesDescription/ImageType"
    if "AC" in tokens or "ATTENUATION" in tokens or "CORRECTED" in tokens:
        return "ac", "SeriesDescription/ImageType"
    return "unknown", "None"


def iter_series(dataset_root, max_files=0, max_series=0, keep_first_ds=False):
    series_info = {}
    series_counts = defaultdict(int)
    files_scanned = 0
    first_ds = None

    for root, _, files in os.walk(dataset_root):
        if not files:
            continue
        for name in files:
            path = os.path.join(root, name)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
            except Exception:
                files_scanned += 1
                if max_files and files_scanned >= max_files:
                    return series_info, series_counts, first_ds
                continue

            files_scanned += 1
            if getattr(ds, "Modality", "") != "PT":
                if max_files and files_scanned >= max_files:
                    return series_info, series_counts, first_ds
                continue

            series_uid = getattr(ds, "SeriesInstanceUID", None)
            if not series_uid:
                if max_files and files_scanned >= max_files:
                    return series_info, series_counts, first_ds
                continue

            if series_uid in series_info:
                if max_files and files_scanned >= max_files:
                    return series_info, series_counts, first_ds
                continue

            ac_class, reason = classify_pet_series(ds)
            series_counts[ac_class] += 1
            series_info[series_uid] = {
                "ac_class": ac_class,
                "reason": reason,
                "desc": str(getattr(ds, "SeriesDescription", "")),
                "image_type": getattr(ds, "ImageType", None),
                "corrected": ds.get((0x0028, 0x0051)).value if ds.get((0x0028, 0x0051)) else None,
                "path": root,
            }

            if keep_first_ds and first_ds is None:
                first_ds = ds

            if max_series and len(series_info) >= max_series:
                return series_info, series_counts, first_ds

            if max_files and files_scanned >= max_files:
                return series_info, series_counts, first_ds

            break

    return series_info, series_counts, first_ds


def _format_value(value, max_len=200):
    try:
        text = str(value)
    except Exception:
        return "<unprintable>"
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def print_all_tags(ds):
    print("\nDICOM tags for first PET series:")
    for elem in ds.iterall():
        tag = f"{elem.tag.group:04X},{elem.tag.element:04X}"
        name = elem.keyword or elem.name
        vr = elem.VR
        if vr == "SQ":
            try:
                count = len(elem.value)
            except Exception:
                count = "?"
            value = f"<Sequence, {count} item(s)>"
        else:
            value = _format_value(elem.value)
        print(f"  ({tag}) {name} [{vr}] = {value}")


def main():
    parser = argparse.ArgumentParser(description="Inspect PET AC/NAC series using DICOM tags.")
    parser.add_argument("paths", nargs="+", help="Dataset root paths to scan")
    parser.add_argument("--samples", type=int, default=5, help="How many NAC examples to print")
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Stop after scanning this many files (0 = no limit)",
    )
    parser.add_argument(
        "--first-only",
        action="store_true",
        help="Stop after the first PET series is found",
    )
    parser.add_argument(
        "--print-tags",
        action="store_true",
        help="Print all DICOM tags for the first PET series found",
    )
    args = parser.parse_args()

    for dataset_root in args.paths:
        if not os.path.exists(dataset_root):
            print(f"Missing path: {dataset_root}")
            continue

        max_series = 1 if args.first_only else 0
        series_info, series_counts, first_ds = iter_series(
            dataset_root,
            max_files=args.max_files,
            max_series=max_series,
            keep_first_ds=args.print_tags,
        )
        total = sum(series_counts.values())

        print(f"\nDataset: {dataset_root}")
        print(f"  PET series total: {total}")
        print(f"  AC: {series_counts['ac']}  NAC: {series_counts['nac']}  Unknown: {series_counts['unknown']}")

        nac_samples = [
            (uid, info) for uid, info in series_info.items() if info["ac_class"] == "nac"
        ]
        print(f"  NAC samples (up to {args.samples}):")
        for uid, info in nac_samples[: args.samples]:
            print(f"    - UID: {uid}")
            print(f"      Desc: {info['desc']}")
            print(f"      CorrectedImage: {info['corrected']}")
            print(f"      ImageType: {info['image_type']}")
            print(f"      Reason: {info['reason']}")
            print(f"      Path: {info['path']}")

        if args.print_tags:
            if first_ds is None:
                print("\nNo PET series found to print tags.")
            else:
                print_all_tags(first_ds)


if __name__ == "__main__":
    main()
