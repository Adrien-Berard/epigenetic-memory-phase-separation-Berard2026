"""
collect_and_zip.py
------------------
Package LAMMPS input and trajectory files from a parameter-scan folder tree
into a single zip archive with an embedded README.

Usage:
    python collect_and_zip.py --root /path/to/simulations --out Figi.zip
"""

import argparse
from collections import defaultdict
from pathlib import Path
import zipfile

ALLOWED_EXTENSIONS = {
    ".template",
    ".lammpstrj",
    ".dat",
    ".map",
    ".txt",
    ".equil",
}

SPECIAL_FILES = {"input.lammps"}

EXCLUDE_FILES = {"log.lammps", "output.log"}

README_NAME = "README.md"
PAPER_LINE = (
    "Data generated with a modified LAMMPS build (stable April 2024) for "
    "Berard et al.: Epigenetic memory achieved through chromatin-induced "
    "phase separation.\n"
)


def collect_files(root_dir: Path):
    selected_files = []
    folder_set = set()
    file_type_counter = defaultdict(int)

    for file_path in root_dir.rglob("*"):
        if not file_path.is_file():
            continue
        name = file_path.name.lower()
        ext = file_path.suffix.lower()
        if name in EXCLUDE_FILES:
            continue
        if ext in ALLOWED_EXTENSIONS or file_path.name in SPECIAL_FILES:
            selected_files.append(file_path)
            folder_set.add(file_path.parent)
            file_type_counter[ext if ext else file_path.name] += 1

    return selected_files, folder_set, file_type_counter


def build_readme(root_dir: Path, folder_set, selected_files, file_type_counter):
    sorted_folders = sorted(str(p.relative_to(root_dir)) for p in folder_set)
    lines = [
        "# LAMMPS Data Package\n",
        PAPER_LINE,
        "## Summary\n",
        f"- Total folders: {len(folder_set)}",
        f"- Total files included: {len(selected_files)}\n",
        "## Folder structure\n",
    ]
    lines.extend(f"- {f}" for f in sorted_folders)
    lines.append("\n## File types included\n")
    for k, v in sorted(file_type_counter.items()):
        lines.append(f"- {k}: {v} files")
    lines.extend([
        "\n## Notes\n",
        "- Excluded large logs: log.lammps, output.log",
        "- Only LAMMPS-related input and data files were packaged",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Zip LAMMPS simulation folders.")
    parser.add_argument("--root", type=Path, required=True,
                        help="Root directory to scan (subfolders or flat layout).")
    parser.add_argument("--out", type=Path, default="lammps_data.zip",
                        help="Output zip filename.")
    args = parser.parse_args()

    root_dir = args.root.resolve()
    if not root_dir.is_dir():
        raise SystemExit(f"Not a directory: {root_dir}")

    selected_files, folder_set, file_type_counter = collect_files(root_dir)
    readme_content = build_readme(root_dir, folder_set, selected_files, file_type_counter)

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in selected_files:
            arcname = file_path.relative_to(root_dir)
            zipf.write(file_path, arcname)
        zipf.writestr(README_NAME, readme_content)

    print(f"Created: {args.out}")
    print(f"Included {len(selected_files)} files from {len(folder_set)} folders")


if __name__ == "__main__":
    main()
