import os
from json import load

import gdown

from src.constants import (
    drivelm_dir,
    drivelm_train_json,
    drivelm_val_json,
    nuscenes_dir,
)
from src.data.create_image_grid_dataset import create_image_grid_dataset
from src.data.generate_descriptor_qas import (
    generate_descriptor_qas,
)
from src.utils.logger import get_logger
from src.utils.utils import extract_children

logger = get_logger(__name__)


def get_ds(split: str) -> None:
    logger.info("Downloading dataset")
    if split == "train":
        out_name = os.path.join(nuscenes_dir, "drivelm_nus_imgs_train.zip")
        gdown.download(
            id="1DeosPGYeM2gXSChjMODGsQChZyYDmaUz",
            output=out_name,
        )
        extract_children(out_name, nuscenes_dir)
        gdown.download(
            id="1CvTPwChKvfnvrZ1Wr0ZNVqtibkkNeGgt",
            output=os.path.join(drivelm_dir, "v1_1_train_nus.json"),
        )
    else:
        out_name = os.path.join(nuscenes_dir, "drivelm_nus_imgs_val.zip")
        gdown.download(
            id="18f8ygNxGZWat-crUjroYuQbd39Sk9xCo",
            output=out_name,
        )
        extract_children(out_name, nuscenes_dir)
        gdown.download(
            id="1fsVP7jOpvChcpoXVdypaZ4HREX1gA7As",
            output=os.path.join(drivelm_dir, "v1_1_val_nus_q_only.json"),
        )


def load_dataset(split: str, add_augmented: bool = False, use_grid: bool = False):
    dataset_paths = {
        "train": drivelm_train_json,
        "val": drivelm_val_json,
    }

    if split not in dataset_paths:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    base_path = dataset_paths[split]
    if not base_path.is_file():
        get_ds(split)

    with open(base_path) as f:
        data = load(f)

    if split == "train" and add_augmented:
        data = generate_descriptor_qas(data)

    if use_grid:
        data = create_image_grid_dataset(data)

    return data
