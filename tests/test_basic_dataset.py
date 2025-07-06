import logging
import unittest
from collections import Counter

import pytest

from src.data.basic_dataset import DriveLMImageDataset
from src.data.message_formats import QwenMessageFormat
from src.utils.logger import get_logger
from src.utils.utils import create_subset_for_testing

logging.getLogger().setLevel(logging.DEBUG)

logger = get_logger(__name__)


@pytest.mark.dataset
class TestDriveLMImageDataset(unittest.TestCase):
    def test_train_dataset_is_not_empty(self):
        dataset = DriveLMImageDataset(message_format=QwenMessageFormat(), split="train")
        self.assertGreater(len(dataset), 0, "Dataset should not be empty")

    def test_val_dataset_is_not_empty(self):
        dataset = DriveLMImageDataset(message_format=QwenMessageFormat(), split="val")
        self.assertGreater(len(dataset), 0, "Validation dataset should not be empty")

    def test_test_dataset_is_not_empty(self):
        dataset = DriveLMImageDataset(message_format=QwenMessageFormat(), split="test")
        self.assertGreater(len(dataset), 0, "Test dataset should not be empty")

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

    def test_subset_dataset(self):
        dataset = DriveLMImageDataset(
            message_format=QwenMessageFormat(),
            split="train",
        )
        test_set_size = min(50, len(dataset))  # Use a small test set for speed
        subset = create_subset_for_testing(dataset, test_set_size)

        qa_type_counts_full = Counter(item.qa_type for item in dataset)
        total_full = sum(qa_type_counts_full.values())
        dist_full = {k: v / total_full for k, v in qa_type_counts_full.items()}

        qa_type_counts_subset = Counter(dataset[i].qa_type for i in subset.indices)
        total_subset = sum(qa_type_counts_subset.values())
        dist_subset = {k: v / total_subset for k, v in qa_type_counts_subset.items()}

        for qa_type in dist_full:
            if qa_type in dist_subset:
                diff = abs(dist_full[qa_type] - dist_subset[qa_type])
                assert diff < 0.10, (
                    f"Distribution for qa_type '{qa_type}' differs by more than 10%: "
                    f"full={dist_full[qa_type]:.2%}, subset={dist_subset[qa_type]:.2%}"
                )
            else:
                assert dist_full[qa_type] < 0.05, (
                    f"qa_type '{qa_type}' missing from subset but present in full dataset"
                )


if __name__ == "__main__":
    unittest.main()
