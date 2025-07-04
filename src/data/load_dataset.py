import os
from json import load

import gdown

from src.constants import (
    drivelm_dir,
    drivelm_test_json,
    drivelm_train_json,
    drivelm_val_json,
    nuscenes_dir,
)
from src.data.create_image_grid_dataset import create_image_grid_dataset
from src.data.extract_test_dataset import extract_data
from src.data.generate_descriptor_qas import (
    generate_descriptor_qas,
)
from src.utils.logger import get_logger
from src.utils.utils import extract_children

logger = get_logger(__name__)


def get_ds(split: str) -> None:
    logger.info("Downloading dataset")
    if split == "train" or split == "test":
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
        extract_children(out_name, nuscenes_dir / "samples")
        gdown.download(
            id="1fsVP7jOpvChcpoXVdypaZ4HREX1gA7As",
            output=os.path.join(drivelm_dir, "v1_1_val_nus_q_only.json"),
        )


def load_dataset(
    split: str,
    resize_factor: float = 0.5,
    add_augmented: bool = False,
    use_grid: bool = False,
):
    dataset_paths = {
        "train": drivelm_train_json,
        "val": drivelm_val_json,
        "test": drivelm_train_json,
    }

    if split not in dataset_paths:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val' or 'test'.")

    base_path = dataset_paths[split]
    if not base_path.is_file():
        get_ds(split)

    if split == "test" and not drivelm_test_json.is_file():
        logger.debug("Extracting test dataset from train dataset")
        extract_data(drivelm_train_json, drivelm_test_json)
        base_path = drivelm_test_json

    with open(base_path) as f:
        data = load(f)

    if split == "train" and add_augmented:
        data = generate_descriptor_qas(data)

    if use_grid:
        data = create_image_grid_dataset(data, resize_factor)

    return data
