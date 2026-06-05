import argparse
import os
import re
import zipfile
from collections import defaultdict

import requests


def is_nac(description):
    if not description:
        return False
    text = description.lower()
    return any(token in text for token in ["uncorrected", "non-ac", "nonac", "nac"])


def is_ct_series(series):
    return series.get("Modality") == "CT"


def is_pt_series(series):
    return series.get("Modality") == "PT"


def safe_name(name):
    if not name:
        return "series"
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")


def download_series(series_uid, description, output_dir, label, unzip):
    series_suffix = safe_name(description) or series_uid
    zip_path = os.path.join(output_dir, f"{label}_{series_suffix}.zip")
    extract_dir = os.path.splitext(zip_path)[0]

    if os.path.isdir(extract_dir) and os.listdir(extract_dir):
        print(f"{label} already extracted at {extract_dir}, skipping.")
        return None

    if os.path.isfile(zip_path):
        print(f"{label} ZIP already exists at {zip_path}.")
        if unzip:
            extracted = extract_zip(zip_path)
            if extracted is not None:
                return zip_path
        else:
            return zip_path

    print(f"Downloading {label} (Series UID: {series_uid})...")

    download_url = (
        "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
        f"?SeriesInstanceUID={series_uid}"
    )
    with requests.get(download_url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(zip_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    print(f"Saved to {zip_path}")
    if unzip:
        extract_zip(zip_path)
    return zip_path


def extract_zip(zip_path):
    extract_dir = os.path.splitext(zip_path)[0]
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Extracting to {extract_dir}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as handle:
            handle.extractall(extract_dir)
    except zipfile.BadZipFile as exc:
        print(f"Failed to extract {zip_path}: {exc}")
        try:
            os.remove(zip_path)
            print(f"Removed corrupt ZIP {zip_path}")
        except OSError as remove_exc:
            print(f"Failed to remove corrupt ZIP {zip_path}: {remove_exc}")
        return None
    try:
        os.remove(zip_path)
        print(f"Extracted and removed {zip_path}")
    except OSError as exc:
        print(f"Extracted {zip_path}, but failed to remove it: {exc}")
    return extract_dir


def select_series(series_list, include_ct):
    pet_ac = None
    pet_nac = None
    ct_series = None

    for series in series_list:
        if is_pt_series(series):
            description = series.get("SeriesDescription", "")
            if is_nac(description):
                if pet_nac is None:
                    pet_nac = series
            else:
                if pet_ac is None:
                    pet_ac = series
        elif include_ct and is_ct_series(series) and ct_series is None:
            ct_series = series

        if pet_ac and pet_nac and (ct_series or not include_ct):
            break

    return pet_ac, pet_nac, ct_series


def main():
    parser = argparse.ArgumentParser(
        description="Download PET AC/NAC and optional CT series from a TCIA collection."
    )
    parser.add_argument("collection", help="TCIA collection name")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save downloaded ZIP files",
    )
    parser.add_argument(
        "--max-studies",
        type=int,
        default=20,
        help="Maximum number of studies to download",
    )
    parser.add_argument(
        "--include-ct",
        action="store_true",
        help="Download one CT series per study when available",
    )
    parser.add_argument(
        "--unzip",
        dest="unzip",
        action="store_true",
        help="Extract downloaded ZIP files",
    )
    parser.add_argument(
        "--no-unzip",
        dest="unzip",
        action="store_false",
        help="Do not extract downloaded ZIP files",
    )
    parser.set_defaults(unzip=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    api_url = (
        "https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries"
        f"?Collection={args.collection}"
    )
    response = requests.get(api_url, timeout=120)
    response.raise_for_status()
    series_list = response.json()

    if not series_list:
        print("No series found for this collection.")
        return

    series_by_study = defaultdict(list)
    for series in series_list:
        study_uid = series.get("StudyInstanceUID")
        if study_uid:
            series_by_study[study_uid].append(series)

    study_uids = list(series_by_study.keys())
    if not study_uids:
        print("No studies found for this collection.")
        return

    downloaded = 0
    for study_uid in study_uids:
        if downloaded >= args.max_studies:
            break

        pet_ac, pet_nac, ct_series = select_series(
            series_by_study[study_uid], args.include_ct
        )
        if not pet_ac or not pet_nac:
            print(f"Skipping {study_uid}: missing PET AC or PET NAC series.")
            continue

        study_dir = os.path.join(args.output_dir, safe_name(study_uid))
        os.makedirs(study_dir, exist_ok=True)

        print(f"\nStudy {study_uid}:")
        download_series(
            pet_ac["SeriesInstanceUID"],
            pet_ac.get("SeriesDescription", ""),
            study_dir,
            "PET_AC",
            args.unzip,
        )

        download_series(
            pet_nac["SeriesInstanceUID"],
            pet_nac.get("SeriesDescription", ""),
            study_dir,
            "PET_NAC",
            args.unzip,
        )

        if args.include_ct:
            if ct_series:
                download_series(
                    ct_series["SeriesInstanceUID"],
                    ct_series.get("SeriesDescription", ""),
                    study_dir,
                    "CT",
                    args.unzip,
                )
            else:
                print("  CT: not found")

        downloaded += 1

    print(
        f"\nDownloaded {downloaded} study folder(s) to {os.path.abspath(args.output_dir)}"
    )


if __name__ == "__main__":
    main()
