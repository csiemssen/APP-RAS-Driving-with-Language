import unittest

import pytest

from src.data.basic_dataset import DriveLMImageDataset
from src.data.message_formats import QwenMessageFormat


@pytest.mark.dataset
class TestDriveLMImageDataset(unittest.TestCase):
    def test_train_dataset_is_not_empty(self):
        dataset = DriveLMImageDataset(message_format=QwenMessageFormat(), split="train")
        self.assertGreater(len(dataset), 0, "Dataset should not be empty")

    def test_val_dataset_is_not_empty(self):
        dataset = DriveLMImageDataset(message_format=QwenMessageFormat(), split="val")
        self.assertGreater(len(dataset), 0, "Validation dataset should not be empty")

    def test_dataset_with_augmented_questions(self):
        dataset_not_augmented = DriveLMImageDataset(
            message_format=QwenMessageFormat(),
            split="train",
            add_augmented=False,
        )

        dataset = DriveLMImageDataset(
            message_format=QwenMessageFormat(),
            split="train",
            add_augmented=True,
        )

        self.assertGreater(
            len(dataset),
            0,
            "Dataset with augmented questions should not be empty",
        )

        self.assertGreater(
            len(dataset_not_augmented),
            0,
            "Dataset without augmented questions should not be empty",
        )

        self.assertGreater(
            len(dataset),
            len(dataset_not_augmented),
            "Dataset with augmented questions should be larger than without",
        )


if __name__ == "__main__":
    unittest.main()
