"""Prepare and verify the shared DeFungi 64x64 split.

The committed ``split_manifest.csv`` is the authority for membership. A clean
clone first restores ``dataset_64x64.zip`` and validates every path, class,
split, size and colour mode. Raw-data regeneration uses that same manifest;
it never silently creates a different split.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import requests
from PIL import Image


REPO_DIR = Path(__file__).resolve().parent
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/773/defungi.zip"
RAW_ZIP_PATH = REPO_DIR / "defungi.zip"
PACKAGED_ZIP_PATH = REPO_DIR / "dataset_64x64.zip"
EXTRACT_DIR = REPO_DIR / "defungi_raw"
OUTPUT_DIR = REPO_DIR / "dataset"
MANIFEST_PATH = REPO_DIR / "split_manifest.csv"
IMG_SIZE = (64, 64)
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract a ZIP after rejecting members outside ``destination``."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def read_manifest(manifest_path: Path = MANIFEST_PATH) -> list[dict[str, str]]:
    """Read and validate the committed split manifest."""
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    required = {"filename", "class", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid manifest columns in {manifest_path}")

    identities = [(row["class"], row["filename"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate class/filename rows in split_manifest.csv")
    if {row["split"] for row in rows} != set(SPLIT_RATIOS):
        raise ValueError("Manifest must contain train, val and test splits")
    return rows


def verify_prepared_dataset(
    dataset_dir: Path = OUTPUT_DIR,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, int]:
    """Require the prepared tree to match the manifest exactly."""
    rows = read_manifest(manifest_path)
    expected = {
        Path(row["split"]) / row["class"] / row["filename"] for row in rows
    }
    actual = {
        path.relative_to(dataset_dir)
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    missing, extra = expected - actual, actual - expected
    if missing or extra:
        raise ValueError(
            f"Dataset/manifest mismatch: {len(missing)} missing, {len(extra)} extra"
        )

    for relative_path in sorted(expected):
        with Image.open(dataset_dir / relative_path) as image:
            if image.size != IMG_SIZE or image.mode != "RGB":
                raise ValueError(
                    f"Expected RGB {IMG_SIZE}, got {image.mode} {image.size}: "
                    f"{relative_path}"
                )

    counts = Counter(row["split"] for row in rows)
    return {split: counts[split] for split in SPLIT_RATIOS}


def download_dataset() -> Path:
    """Download the raw UCI archive with retries and an atomic final rename."""
    if RAW_ZIP_PATH.exists() and RAW_ZIP_PATH.stat().st_size > 1_000_000:
        return RAW_ZIP_PATH

    partial = RAW_ZIP_PATH.with_suffix(".zip.part")
    for attempt in range(1, 4):
        try:
            with requests.get(DOWNLOAD_URL, stream=True, timeout=180) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(128 * 1024):
                        if chunk:
                            stream.write(chunk)
            partial.replace(RAW_ZIP_PATH)
            return RAW_ZIP_PATH
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(5 * attempt)
    raise RuntimeError("Unreachable download state")


def discover_images(extract_dir: Path = EXTRACT_DIR) -> dict[str, list[Path]]:
    """Return deterministic, sorted source lists by class folder."""
    class_images: dict[str, list[Path]] = {}
    for path in sorted(extract_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            class_images.setdefault(path.parent.name, []).append(path)
    for class_name in class_images:
        class_images[class_name] = sorted(
            class_images[class_name], key=lambda item: item.as_posix()
        )
    return dict(sorted(class_images.items()))


def _source_index(class_images: dict[str, list[Path]]) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for class_name, images in class_images.items():
        for image_path in images:
            key = (class_name, image_path.name)
            if key in index:
                raise ValueError(f"Ambiguous raw image name for {key}")
            index[key] = image_path
    return index


def _write_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS).save(
            destination
        )


def split_and_save(
    class_images: dict[str, list[Path]],
    output_dir: Path = OUTPUT_DIR,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, int]:
    """Materialize raw images using the existing manifest.

    A deterministic fallback can create a new manifest only when no manifest
    exists. Image lists are sorted before the seeded shuffle.
    """
    source_by_key = _source_index(class_images)

    if manifest_path.exists():
        rows = read_manifest(manifest_path)
    else:
        rng = random.Random(RANDOM_SEED)
        rows = []
        for class_name, discovered in sorted(class_images.items()):
            images = sorted(discovered, key=lambda item: item.as_posix())
            rng.shuffle(images)
            n_train = int(len(images) * SPLIT_RATIOS["train"])
            n_val = int(len(images) * SPLIT_RATIOS["val"])
            memberships = {
                "train": images[:n_train],
                "val": images[n_train : n_train + n_val],
                "test": images[n_train + n_val :],
            }
            for split, members in memberships.items():
                rows.extend(
                    {"filename": item.name, "class": class_name, "split": split}
                    for item in members
                )
        with manifest_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["filename", "class", "split"])
            writer.writeheader()
            writer.writerows(rows)

    for row in rows:
        key = (row["class"], row["filename"])
        if key not in source_by_key:
            raise FileNotFoundError(f"Manifest image missing from raw data: {key}")
        destination = output_dir / row["split"] / row["class"] / row["filename"]
        _write_image(source_by_key[key], destination)
    return verify_prepared_dataset(output_dir, manifest_path)


def _restore_packaged_split() -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix="defungi-stage-", dir=REPO_DIR) as temp:
        staging = Path(temp)
        _safe_extract(PACKAGED_ZIP_PATH, staging)
        staged_dataset = staging / "dataset"
        counts = verify_prepared_dataset(staged_dataset)
        if OUTPUT_DIR.exists():
            raise FileExistsError(
                "Existing dataset does not match the manifest; move it aside explicitly."
            )
        shutil.move(str(staged_dataset), OUTPUT_DIR)
    return counts


def prepare_dataset() -> dict[str, int]:
    if OUTPUT_DIR.exists():
        return verify_prepared_dataset()
    if PACKAGED_ZIP_PATH.exists():
        return _restore_packaged_split()

    archive = download_dataset()
    if not EXTRACT_DIR.exists():
        EXTRACT_DIR.mkdir(parents=True)
        _safe_extract(archive, EXTRACT_DIR)
    class_images = discover_images()
    if not class_images:
        raise RuntimeError(f"No images found under {EXTRACT_DIR}")
    return split_and_save(class_images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only", action="store_true", help="Validate an existing dataset tree"
    )
    args = parser.parse_args()
    counts = verify_prepared_dataset() if args.verify_only else prepare_dataset()
    total = sum(counts.values())
    print(f"Verified DeFungi dataset at {OUTPUT_DIR}")
    print(
        f"Train: {counts['train']} | Validation: {counts['val']} | "
        f"Test: {counts['test']} | Total: {total}"
    )


if __name__ == "__main__":
    main()
