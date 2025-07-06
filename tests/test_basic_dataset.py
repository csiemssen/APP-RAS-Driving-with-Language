import unittest

import pytest

from src.data.basic_dataset import DriveLMImageDataset
from src.data.message_formats import QwenMessageFormat
from src.utils.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.dataset
class TestDriveLMImageDataset(unittest.TestCase):
    def test_train_dataset_is_not_empty(self):
        dataset = DriveLMImageDataset(message_format=QwenMessageFormat(), split="train")
        self.assertGreater(len(dataset), 0, "Dataset should not be empty")

    def test_val_dataset_is_not_empty(self):
        dataset = DriveLMImageDataset(message_format=QwenMessageFormat(), split="val")
        self.assertGreater(len(dataset), 0, "Validation dataset should not be empty")

    def test_dataset_with_reasoning_context(self):
        dataset = DriveLMImageDataset(
            message_format=QwenMessageFormat(),
            split="train",
            use_reasoning=True,
        )
        self.assertGreater(
            len(dataset),
            0,
            "Dataset with reasoning context should not be empty",
        )

        has_valid_context_pair = False
        for item in dataset:
            for question, answer in item.context_pairs:
                if len(question) > 0 and len(answer) > 0:
                    has_valid_context_pair = True
                    break
            if has_valid_context_pair:
                break

        self.assertTrue(
            has_valid_context_pair,
            "At least one context pair should have a non-empty question and answer",
        )

    def test_dataset_with_grid(self):
        dataset = DriveLMImageDataset(
            message_format=QwenMessageFormat(),
            split="train",
            use_grid=True,
        )

        self.assertGreater(
            len(dataset),
            0,
            "Dataset with grid should not be empty",
        )

        for item in dataset:
            self.assertIn(
                "GRID",
                item.image_path,
                "Image path should contain grid information",
            )

    def test_dataset_with_augmented_questions(self):
        dataset = DriveLMImageDataset(
            message_format=QwenMessageFormat(),
            split="train",
        )
        dataset_with_augmented = DriveLMImageDataset(
            message_format=QwenMessageFormat(),
            split="train",
            add_augmented=True,
        )

        self.assertGreater(
            len(dataset),
            0,
            "Dataset should not be empty",
        )

        self.assertGreater(
            len(dataset_with_augmented),
            len(dataset),
            "Dataset with augmented questions should be larger than the base dataset",
        )


if __name__ == "__main__":
    unittest.main()
