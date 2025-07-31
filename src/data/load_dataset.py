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
from src.data.extract_test_dataset import extract_data
from src.utils.logger import get_logger
from src.utils.utils import extract_children

logger = get_logger(__name__)


def get_ds(split: str) -> None:
    logger.info("Downloading dataset")
    if split == "train" or split == "test":
        out_name = os.path.join(nuscenes_dir, "drivelm_nus_imgs_train.zip")
        gdown.download(
            id="1fcy_oamCh2WER9ldd790DI39mNSwW9GV",
            output=out_name,
        )
        extract_children(out_name, nuscenes_dir)
        gdown.download(
            id="14tuCTmV63nxkTO3pUeEGib45ddYHmjnV",
            output=os.path.join(drivelm_dir, "v1_1_train_nus.json"),
        )

    else:
        out_name = os.path.join(nuscenes_dir, "drivelm_nus_imgs_val.zip")
        gdown.download(
            id="1IXvFxAKfiM2f5We3Y7jYPw0nG_c7uSyX",
            output=out_name,
        )
        extract_children(out_name, nuscenes_dir / "samples")
        gdown.download(
            id="1DmwJ3EjtSVSl9QAygOeaME0eMGNONx5I",
            output=os.path.join(drivelm_dir, "v1_1_val_nus_q_only.json"),
        )


def load_dataset(split: str):
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

    if split == "test":
        logger.debug("Extracting test dataset from train dataset")
        extract_data(drivelm_train_json, drivelm_test_json)

        base_path = drivelm_test_json

    with open(base_path) as f:
        logger.debug(f"Loading dataset from {base_path}")
        data = load(f)

    return data
