"""Copy extracted Bladder 13.11.25 folders into the consolidated target directory.

The script traverses the numbered download folders (001-026) under
``data/gdrive_downloads`` and copies their internal ``Bladder 13.11.25``
contents into ``data/gdrive_downloads/Bladder 13.11.25/Bladder 13.11.25``.
Existing files are overwritten so the run can be repeated safely.
"""

from pathlib import Path
import shutil

def copy_bladder_splits() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    downloads_root = repo_root / "data" / "gdrive_downloads"
    target_dir = downloads_root / "Bladder 13.11.25" / "Bladder 13.11.25"
    target_dir.mkdir(parents=True, exist_ok=True)

    suffixes = [f"{i:03d}" for i in range(1, 27)]
    copied_sources: list[Path] = []
    missing_sources: list[Path] = []

    for suffix in suffixes:
        source_dir = (
            downloads_root / f"Bladder 13.11.25-20260212T080837Z-1-{suffix}" / "Bladder 13.11.25"
        )
        if not source_dir.exists():
            missing_sources.append(source_dir)
            continue

        for item in source_dir.iterdir():
            dest_path = target_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest_path, dirs_exist_ok=True)
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest_path)

        copied_sources.append(source_dir)
        print(f"Copied contents from {source_dir} -> {target_dir}")

    if missing_sources:
        print("Missing source folders detected:")
        for missing in missing_sources:
            print(f" - {missing}")

    print(f"Completed. Processed {len(copied_sources)} source folders.")

if __name__ == "__main__":
    copy_bladder_splits()
