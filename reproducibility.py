"""Shared deterministic seeding helpers for future training runs."""

from __future__ import annotations

import random

import numpy as np
import torch


DEFAULT_SEED = 42


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
