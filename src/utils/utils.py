import shutil
from pathlib import Path
from typing import Any

import torch
import numpy as np
from torch.utils.data import Dataset, Subset

from src.utils.logger import get_logger

logger = get_logger(__name__)

def flatten(list_of_lists: list[list[Any]]) -> list[Any]:
    return [x for sublist in list_of_lists for x in sublist]


def remove_nones(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def sanitize_model_name(model_path: str) -> str:
    return model_path.replace("/", "_")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "auto"


def is_mps() -> bool:
    return torch.backends.mps.is_available()


def is_cuda() -> bool:
    return torch.cuda.is_available()


def extract_children(zip_path: str, out_path: str):
    tmp = Path(out_path) / "tmp"
    shutil.unpack_archive(zip_path, tmp)
    for parent in tmp.iterdir():
        for child in parent.iterdir():
            shutil.move(child, out_path)
    shutil.rmtree(tmp)

def create_subset_for_testing(ds: Dataset, test_set_size: int) -> Dataset:
    logger.info(f"Creating subset with size {test_set_size}")
    num_samples = min(test_set_size, len(ds))
    subset = Subset(ds, np.arange(num_samples))
    logger.info(f"Created subset with size {len(subset)}")
    return subset
