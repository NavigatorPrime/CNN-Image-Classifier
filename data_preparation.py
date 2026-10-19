"""Prepare, rebuild, and verify the shared DeFungi 64x64 split.

The committed ``split_manifest.csv`` is authoritative for membership. The
validator checks paths, filenames, classes, dimensions, colour mode, and both
raw-byte and decoded-pixel content hashes. Identical content may occur within
one split, but content shared by train, validation, and test is rejected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import random
import shutil
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

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
SPLIT_ORDER = tuple(SPLIT_RATIOS)
RANDOM_SEED = 42
EXPECTED_CLASSES = {"H1", "H2", "H3", "H5", "H6"}
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

    filenames = [row["filename"] for row in rows]
    duplicate_filenames = sorted(
        name for name, count in Counter(filenames).items() if count > 1
    )
    if duplicate_filenames:
        raise ValueError(
            "Duplicate filenames in split_manifest.csv: "
            f"{duplicate_filenames[:5]}"
        )

    identities = [(row["class"], row["filename"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate class/filename rows in split_manifest.csv")
    if {row["split"] for row in rows} != set(SPLIT_RATIOS):
        raise ValueError("Manifest must contain train, val and test splits")
    if {row["class"] for row in rows} != EXPECTED_CLASSES:
        raise ValueError(f"Manifest classes must be {sorted(EXPECTED_CLASSES)}")
    return rows


def _content_hashes(path: Path) -> tuple[str, str]:
    """Return SHA-256 hashes of file bytes and decoded RGB pixels."""
    raw = path.read_bytes()
    raw_hash = hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as image:
        if image.size != IMG_SIZE or image.mode != "RGB":
            raise ValueError(
                f"Expected RGB {IMG_SIZE}, got {image.mode} {image.size}: {path}"
            )
        pixel_payload = (
            f"{image.mode}:{image.width}x{image.height}:".encode("ascii")
            + image.tobytes()
        )
    return raw_hash, hashlib.sha256(pixel_payload).hexdigest()


def audit_prepared_dataset(
    dataset_dir: Path = OUTPUT_DIR,
    manifest_path: Path = MANIFEST_PATH,
) -> dict:
    """Return a full dataset audit, failing on any invalid condition."""
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

    actual_names = [path.name for path in actual]
    duplicate_names = sorted(
        name for name, count in Counter(actual_names).items() if count > 1
    )
    if duplicate_names:
        raise ValueError(f"Duplicate filenames in dataset: {duplicate_names[:5]}")

    row_by_path = {
        Path(row["split"]) / row["class"] / row["filename"]: row for row in rows
    }
    raw_groups: dict[str, list[Path]] = defaultdict(list)
    pixel_groups: dict[str, list[Path]] = defaultdict(list)
    for relative_path in sorted(expected):
        raw_hash, pixel_hash = _content_hashes(dataset_dir / relative_path)
        raw_groups[raw_hash].append(relative_path)
        pixel_groups[pixel_hash].append(relative_path)

    def duplicate_summary(groups: dict[str, list[Path]]) -> tuple[int, list[list[Path]]]:
        duplicates = [members for members in groups.values() if len(members) > 1]
        cross_split = [
            members
            for members in duplicates
            if len({row_by_path[path]["split"] for path in members}) > 1
        ]
        return len(duplicates), cross_split

    raw_duplicate_groups, raw_cross_split = duplicate_summary(raw_groups)
    pixel_duplicate_groups, pixel_cross_split = duplicate_summary(pixel_groups)
    if raw_cross_split or pixel_cross_split:
        examples = pixel_cross_split[:3] or raw_cross_split[:3]
        formatted = [" | ".join(path.as_posix() for path in group) for group in examples]
        raise ValueError(
            "Identical content crosses dataset splits: "
            f"{len(raw_cross_split)} raw-hash groups, "
            f"{len(pixel_cross_split)} pixel-hash groups. Examples: {formatted}"
        )

    split_counts = Counter(row["split"] for row in rows)
    class_split_counts = Counter((row["class"], row["split"]) for row in rows)
    return {
        "total": len(rows),
        "split_counts": {split: split_counts[split] for split in SPLIT_ORDER},
        "class_split_counts": {
            class_name: {
                split: class_split_counts[(class_name, split)]
                for split in SPLIT_ORDER
            }
            for class_name in sorted(EXPECTED_CLASSES)
        },
        "raw_duplicate_groups": raw_duplicate_groups,
        "pixel_duplicate_groups": pixel_duplicate_groups,
        "raw_cross_split_groups": 0,
        "pixel_cross_split_groups": 0,
    }


def verify_prepared_dataset(
    dataset_dir: Path = OUTPUT_DIR,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, int]:
    """Validate the prepared tree and return split counts."""
    audit = audit_prepared_dataset(dataset_dir, manifest_path)
    print(
        "Content audit: "
        f"{audit['pixel_duplicate_groups']} duplicate-content groups; "
        "0 cross-split groups"
    )
    return audit["split_counts"]


def verify_packaged_dataset(
    package_path: Path = PACKAGED_ZIP_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> dict:
    """Extract the committed package temporarily and return its audit."""
    with tempfile.TemporaryDirectory(prefix="defungi-verify-", dir=REPO_DIR) as temp:
        staging = Path(temp)
        _safe_extract(package_path, staging)
        return audit_prepared_dataset(staging / "dataset", manifest_path)


def download_dataset() -> Path:
    """Download the raw UCI archive with retries and an atomic final rename."""
    import requests

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


def _normalised_pixel_hash(path: Path) -> str:
    with Image.open(path) as image:
        normalised = image.convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS)
        payload = b"RGB:64x64:" + normalised.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _target_counts(total: int) -> dict[str, int]:
    train = round(total * SPLIT_RATIOS["train"])
    val = round(total * SPLIT_RATIOS["val"])
    return {"train": train, "val": val, "test": total - train - val}


def _assign_content_groups(
    class_name: str,
    groups: list[list[dict]],
    seed: int = RANDOM_SEED,
) -> dict[str, list[dict]]:
    """Assign intact duplicate groups to deterministic class-balanced splits."""
    target = _target_counts(sum(len(group) for group in groups))
    rng_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{class_name}".encode("utf-8")).digest()[:8],
        "big",
    )
    rng = random.Random(rng_seed)
    duplicate_groups = [group for group in groups if len(group) > 1]
    singleton_groups = [group for group in groups if len(group) == 1]
    rng.shuffle(duplicate_groups)
    rng.shuffle(singleton_groups)

    assigned = {split: [] for split in SPLIT_ORDER}
    counts = Counter()
    for group in duplicate_groups:
        eligible = [
            split
            for split in SPLIT_ORDER
            if counts[split] + len(group) <= target[split]
        ]
        if not eligible:
            raise ValueError(
                f"Cannot assign duplicate group of {len(group)} images in {class_name}"
            )
        split = max(
            eligible,
            key=lambda name: (
                (target[name] - counts[name]) / max(target[name], 1),
                -SPLIT_ORDER.index(name),
            ),
        )
        assigned[split].extend(group)
        counts[split] += len(group)

    singletons = [group[0] for group in singleton_groups]
    cursor = 0
    for split in SPLIT_ORDER:
        needed = target[split] - counts[split]
        assigned[split].extend(singletons[cursor : cursor + needed])
        cursor += needed
    if cursor != len(singletons):
        raise AssertionError("Singleton allocation did not consume every image")
    return assigned


def _clean_rows_from_records(records: list[dict]) -> list[dict[str, str]]:
    by_class_hash: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    hash_classes: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_class_hash[record["class"]][record["pixel_hash"]].append(record)
        hash_classes[record["pixel_hash"]].add(record["class"])
    conflicting = [digest for digest, classes in hash_classes.items() if len(classes) > 1]
    if conflicting:
        raise ValueError(
            f"Identical pixels have conflicting class labels in {len(conflicting)} groups"
        )

    rows = []
    for class_name in sorted(by_class_hash):
        # Canonicalize group and member order before seeded shuffling so the
        # result is independent of the current ZIP/member/split order.
        groups = [
            sorted(
                by_class_hash[class_name][digest],
                key=lambda record: record["filename"],
            )
            for digest in sorted(by_class_hash[class_name])
        ]
        assigned = _assign_content_groups(class_name, groups)
        for split in SPLIT_ORDER:
            for record in assigned[split]:
                rows.append(
                    {
                        "filename": record["filename"],
                        "class": class_name,
                        "split": split,
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            SPLIT_ORDER.index(row["split"]),
            row["class"],
            row["filename"],
        ),
    )


def split_and_save(
    class_images: dict[str, list[Path]],
    output_dir: Path = OUTPUT_DIR,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, int]:
    """Materialize raw images using the manifest or a clean grouped split."""
    source_by_key = _source_index(class_images)

    if manifest_path.exists():
        rows = read_manifest(manifest_path)
    else:
        records = [
            {
                "filename": path.name,
                "class": class_name,
                "pixel_hash": _normalised_pixel_hash(path),
            }
            for class_name, paths in sorted(class_images.items())
            for path in sorted(paths)
        ]
        rows = _clean_rows_from_records(records)
        with manifest_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["filename", "class", "split"],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    for row in rows:
        key = (row["class"], row["filename"])
        if key not in source_by_key:
            raise FileNotFoundError(f"Manifest image missing from raw data: {key}")
        destination = output_dir / row["split"] / row["class"] / row["filename"]
        _write_image(source_by_key[key], destination)
    return verify_prepared_dataset(output_dir, manifest_path)


def rebuild_clean_package(
    package_path: Path = PACKAGED_ZIP_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> dict:
    """Reassign the packaged images while keeping pixel-identical groups intact."""
    records = []
    payloads: dict[tuple[str, str], bytes] = {}
    with zipfile.ZipFile(package_path) as source:
        members = [name for name in source.namelist() if not name.endswith("/")]
        if len(members) != len(set(members)):
            raise ValueError("Duplicate member paths in dataset package")
        for member in members:
            parts = Path(member).parts
            if len(parts) != 4 or parts[0] != "dataset":
                raise ValueError(f"Unexpected package member: {member}")
            _, _old_split, class_name, filename = parts
            raw = source.read(member)
            with Image.open(io.BytesIO(raw)) as image:
                if image.size != IMG_SIZE or image.mode != "RGB":
                    raise ValueError(
                        f"Expected RGB {IMG_SIZE}, got {image.mode} {image.size}: {member}"
                    )
                payload = b"RGB:64x64:" + image.tobytes()
            records.append(
                {
                    "filename": filename,
                    "class": class_name,
                    "pixel_hash": hashlib.sha256(payload).hexdigest(),
                }
            )
            payloads[(class_name, filename)] = raw

    filenames = [record["filename"] for record in records]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Duplicate filenames in dataset package")
    rows = _clean_rows_from_records(records)

    package_temp = package_path.with_suffix(".zip.tmp")
    manifest_temp = manifest_path.with_suffix(".csv.tmp")
    with manifest_temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["filename", "class", "split"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    with zipfile.ZipFile(
        package_temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as destination:
        for row in rows:
            archive_name = f"dataset/{row['split']}/{row['class']}/{row['filename']}"
            info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            destination.writestr(
                info,
                payloads[(row["class"], row["filename"])],
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )

    audit = verify_packaged_dataset(package_temp, manifest_temp)
    package_temp.replace(package_path)
    manifest_temp.replace(manifest_path)
    # Some synchronized filesystems implement replacement via copy rather than
    # a native rename; remove any source names they leave behind.
    package_temp.unlink(missing_ok=True)
    manifest_temp.unlink(missing_ok=True)
    return audit


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


def _print_audit(audit: dict) -> None:
    counts = audit["split_counts"]
    print(
        f"Train: {counts['train']} | Validation: {counts['val']} | "
        f"Test: {counts['test']} | Total: {audit['total']}"
    )
    print("Per-class split counts:")
    for class_name, class_counts in audit["class_split_counts"].items():
        print(
            f"  {class_name}: train={class_counts['train']}, "
            f"val={class_counts['val']}, test={class_counts['test']}"
        )
    print(
        f"Duplicate content groups: raw={audit['raw_duplicate_groups']}, "
        f"pixels={audit['pixel_duplicate_groups']}"
    )
    print("CROSS-SPLIT DUPLICATES = 0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--verify-only", action="store_true", help="Validate an extracted dataset tree"
    )
    action.add_argument(
        "--verify-package", action="store_true", help="Validate dataset_64x64.zip"
    )
    action.add_argument(
        "--rebuild-clean-split",
        action="store_true",
        help="Rebuild the manifest/package with duplicate groups kept in one split",
    )
    parser.add_argument(
        "--yes-rebuild",
        action="store_true",
        help="Required acknowledgement when replacing the committed package and manifest",
    )
    args = parser.parse_args()

    if args.rebuild_clean_split:
        if not args.yes_rebuild:
            parser.error("--rebuild-clean-split requires --yes-rebuild")
        audit = rebuild_clean_package()
        print("Rebuilt deterministic leakage-free DeFungi package and manifest.")
        _print_audit(audit)
    elif args.verify_package:
        audit = verify_packaged_dataset()
        print(f"Verified DeFungi package at {PACKAGED_ZIP_PATH}")
        _print_audit(audit)
    else:
        counts = verify_prepared_dataset() if args.verify_only else prepare_dataset()
        total = sum(counts.values())
        print(f"Verified DeFungi dataset at {OUTPUT_DIR}")
        print(
            f"Train: {counts['train']} | Validation: {counts['val']} | "
            f"Test: {counts['test']} | Total: {total}"
        )


if __name__ == "__main__":
    main()
