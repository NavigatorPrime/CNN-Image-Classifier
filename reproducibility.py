"""Shared deterministic seeding helpers for future training runs."""

from __future__ import annotations

import random
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


DEFAULT_SEED = 42


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_provenance(repo_dir: Path) -> dict[str, object]:
    """Return immutable identifiers for the common corrected dataset split."""
    return {
        "package": "dataset_64x64.zip",
        "package_sha256": _sha256_file(repo_dir / "dataset_64x64.zip"),
        "manifest": "split_manifest.csv",
        "manifest_sha256": _sha256_file(repo_dir / "split_manifest.csv"),
        "split_counts": {"train": 6380, "validation": 1368, "test": 1366},
        "cross_split_duplicate_content": 0,
    }


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy and PyTorch and request deterministic CUDA kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker from PyTorch's assigned worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed: int = DEFAULT_SEED) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def prepare_run_directory(
    repo_dir: Path,
    run_name: str,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Create a run directory without silently replacing existing artifacts."""
    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = repo_dir / "runs" / run_name / timestamp
    else:
        destination = Path(output_dir)
        if not destination.is_absolute():
            destination = repo_dir / destination

    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. "
            "Choose a new --output-dir or pass --overwrite explicitly."
        )
    destination.mkdir(parents=True, exist_ok=True)
    return destination
