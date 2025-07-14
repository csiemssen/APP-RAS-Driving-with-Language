import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, Subset

from src.constants import GRID_IMG_SIZE, IMAGE_SIZE
from src.data.query_item import QueryItem
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


def create_subset(
    ds: Dataset, sample_size: int, by_tag=False, equal_distribution=False
) -> Dataset:
    logger.info(f"Creating subset with size {sample_size}")

    def get_keys(item: QueryItem):
        if by_tag:
            if item.tags is None:
                logger.warning(
                    f"Item {item.qa_id} has no tag, returning empty list for 'tag' by parameter."
                )
            return item.tags
        else:
            return [item.qa_type]

    total = len(ds)
    if total == 0:
        logger.warning("Dataset is empty, returning empty subset.")
        return Subset(ds, [])

    indices_by_key: Dict[str, List[int]] = {}
    for idx in range(len(ds)):
        item: QueryItem = ds[idx]
        for key in get_keys(item):
            indices_by_key.setdefault(key, []).append(idx)

    keys = list(indices_by_key.keys())
    n_keys = len(keys)

    sampled_indices = set()
    if equal_distribution:
        n_per_key = max(1, sample_size // n_keys)
        for key in keys:
            indices = indices_by_key[key]
            sampled = sorted(indices)[:n_per_key]
            sampled_indices.update(sampled)
    else:
        for key, indices in indices_by_key.items():
            n = max(1, int(len(indices) * sample_size / total))
            sampled = sorted(indices)[:n]
            sampled_indices.update(sampled)

    sampled_indices = list(sampled_indices)[:sample_size]

    sampled_count_by_key = {
        k: len([i for i in v if i in sampled_indices])
        for k, v in indices_by_key.items()
    }
    count_by_type = {
        qa_type: len(indices) for qa_type, indices in indices_by_key.items()
    }
    percent_by_qa_type = {
        qa_type: count / total for qa_type, count in count_by_type.items()
    }
    sampled_pct_by_type = {
        qa_type: count / sample_size for qa_type, count in sampled_count_by_key.items()
    }

    logger.debug(f"Original distribution (count): {count_by_type}")
    logger.debug(f"Original distribution (percent): {percent_by_qa_type}")
    logger.debug(f"Sampled distribution (count): {sampled_count_by_key}")
    logger.debug(f"Sampled distribution (percent): {sampled_pct_by_type}")

    subset = Subset(ds, sampled_indices)
    logger.info(f"Created subset with size {len(subset)}")
    return subset


def parse_key_objects(question: str) -> List[str]:
    pattern = r"<[^>]+>"
    return re.findall(pattern, question)


def get_resize_image_size(resize_factor: float, grid=False) -> Tuple[int, int]:
    if grid:
        height = int(GRID_IMG_SIZE[0] * resize_factor)
        width = int(GRID_IMG_SIZE[1] * resize_factor)
    else:
        height = int(IMAGE_SIZE[0] * resize_factor)
        width = int(IMAGE_SIZE[1] * resize_factor)

    return (height, width)
