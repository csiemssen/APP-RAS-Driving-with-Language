import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset, Subset

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


def create_subset_for_testing(ds: Dataset, test_set_size: int) -> Dataset:
    logger.info(f"Creating subset with size {test_set_size}")

    indices_by_type: Dict[str, List[int]] = {}
    for idx in range(len(ds)):
        item: QueryItem = ds[idx]
        qa_type = item.qa_type
        indices_by_type.setdefault(qa_type, []).append(idx)

    total = len(ds)

    sample_count_by_type = {
        qa_type: max(1, int(len(indices) / total * test_set_size))
        for qa_type, indices in indices_by_type.items()
    }

    count_sum = sum(sample_count_by_type.values())
    while count_sum < test_set_size:
        largest = max(sample_count_by_type, key=lambda k: len(indices_by_type[k]))
        sample_count_by_type[largest] += 1
        count_sum += 1
    while count_sum > test_set_size:
        largest = max(sample_count_by_type, key=lambda k: count_by_type[k])
        if sample_count_by_type[largest] > 1:
            sample_count_by_type[largest] -= 1
            count_sum -= 1
        else:
            break

    sampled_indices = []
    sampled_count_by_type = {}
    for qa_type, count in sample_count_by_type.items():
        indices = indices_by_type[qa_type]
        sampled = random.sample(indices, min(count, len(indices)))
        sampled_indices.extend(sampled)
        sampled_count_by_type[qa_type] = len(sampled)

    sampled_pct_by_type = {
        qa_type: count / test_set_size
        for qa_type, count in sampled_count_by_type.items()
    }
    count_by_type = {
        qa_type: len(indices) for qa_type, indices in indices_by_type.items()
    }
    percent_by_qa_type = {
        qa_type: count / total for qa_type, count in count_by_type.items()
    }

    logger.debug(f"Original distribution (count): {count_by_type}")
    logger.debug(f"Original distribution (percent): {percent_by_qa_type}")
    logger.debug(f"Sampled distribution (count): {sampled_count_by_type}")
    logger.debug(f"Sampled distribution (percent): {sampled_pct_by_type}")

    subset = Subset(ds, sampled_indices)
    logger.info(f"Created subset with size {len(subset)}")
    return subset


def parse_key_objects(question: str) -> List[str]:
    pattern = r"<[^>]+>"
    return re.findall(pattern, question)
