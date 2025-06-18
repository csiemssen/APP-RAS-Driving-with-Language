import os
from json import load

import gdown

from src.constants import (
    drivelm_dir,
    drivelm_train_augmented_json,
    drivelm_train_json,
    drivelm_val_json,
    nuscenes_dir,
)
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
        extract_children(out_name, nuscenes_dir / "samples")
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


def load_or_download_dataset(split: str, use_augmented: bool):
    dataset_paths = {
        "train": (drivelm_train_json, drivelm_train_augmented_json),
        "val": (drivelm_val_json, None),
    }

    if split not in dataset_paths:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    base_path, augmented_path = dataset_paths[split]
    if not base_path.is_file():
        get_ds(split)

    if split == "train" and use_augmented:
        if not augmented_path.is_file():
            generate_descriptor_qas(
                input_dir=base_path,
                output_dir=augmented_path,
            )
        data_path = augmented_path
    else:
        data_path = base_path

    with open(data_path) as f:
        data = load(f)

    return data
