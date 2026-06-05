import argparse
import os
import re

import requests


def is_nac(description):
    if not description:
        return False
    text = description.lower()
    return any(token in text for token in ["uncorrected", "non-ac", "nonac", "nac"])


def safe_name(name):
    if not name:
        return "series"
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")


def download_series(series_uid, description, output_dir):
    label = "PET_NAC" if is_nac(description) else "PET_AC"
    series_suffix = safe_name(description) or series_uid
    zip_path = os.path.join(output_dir, f"{label}_{series_suffix}.zip")

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
    return label, zip_path


def main():
    parser = argparse.ArgumentParser(
        description="Download PET AC/NAC series from TCIA by StudyInstanceUID."
    )
    parser.add_argument("study_uid", help="StudyInstanceUID from DICOM header")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save downloaded ZIP files",
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Download all PET series (default filters by modality=PT)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    api_url = (
        "https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries"
        f"?StudyInstanceUID={args.study_uid}"
    )
    response = requests.get(api_url, timeout=60)
    response.raise_for_status()
    series_list = response.json()

    if not series_list:
        print("No series found for this StudyInstanceUID.")
        return

    downloaded = []
    for series in series_list:
        modality = series.get("Modality", "")
        if modality != "PT" and not args.keep_all:
            continue

        series_uid = series.get("SeriesInstanceUID")
        description = series.get("SeriesDescription", "")
        if not series_uid:
            continue

        downloaded.append(download_series(series_uid, description, args.output_dir))

    if not downloaded:
        print("No PET series found to download.")
        return

    ac_count = sum(1 for label, _ in downloaded if label == "PET_AC")
    nac_count = sum(1 for label, _ in downloaded if label == "PET_NAC")
    print(f"\nDownloaded PET series: AC={ac_count}, NAC={nac_count}")


if __name__ == "__main__":
    main()
