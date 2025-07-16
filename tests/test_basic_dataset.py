import unittest
from collections import Counter

import pytest

from src.data.basic_dataset import DriveLMImageDataset
from src.data.message_formats import QwenMessageFormat
from src.utils.logger import get_logger
from src.utils.utils import create_subset

logger = get_logger(__name__)


def get_group_counts(dataset, indices, by_tag=False):
    if by_tag:
        return Counter(tag for i in indices for tag in getattr(dataset[i], "tags", []))
    else:
        return Counter(dataset[i].qa_type for i in indices)


def validate_proportional_distribution(
    dataset, subset_indices, by_tag=False, tolerance=0.10
):
    full_counts = get_group_counts(dataset, range(len(dataset)), by_tag)
    subset_counts = get_group_counts(dataset, subset_indices, by_tag)
    total_full = sum(full_counts.values())
    total_subset = sum(subset_counts.values())
    dist_full = {k: v / total_full for k, v in full_counts.items()}
    dist_subset = {k: v / total_subset for k, v in subset_counts.items()}

    for group in dist_full:
        if group in dist_subset:
            diff = abs(dist_full[group] - dist_subset[group])
            assert diff < tolerance, (
                f"Distribution for '{group}' differs by more than {tolerance * 100:.0f}%: "
                f"full={dist_full[group]:.2%}, subset={dist_subset[group]:.2%}"
            )
        else:
            assert dist_full[group] < 0.05, (
                f"'{group}' missing from subset but present in full dataset"
            )


def validate_equal_distribution(dataset, subset_indices, by_tag=False):
    group_type = "tag" if by_tag else "qa_type"
    group_counts = get_group_counts(dataset, subset_indices, by_tag)
    if group_counts:
        min_count = min(group_counts.values())
        max_count = max(group_counts.values())
        assert max_count - min_count <= 1, (
            f"Equal distribution by '{group_type}' failed: "
            f"min={min_count}, max={max_count}, counts={dict(group_counts)}"
        )


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

    def test_subset_proportional_qa_type(self):
        dataset = DriveLMImageDataset(
            message_format=QwenMessageFormat(),
            split="test",
        )
        test_set_size = min(500, len(dataset))
        subset = create_subset(dataset, test_set_size, by_tag=False)
        validate_proportional_distribution(dataset, subset.indices, by_tag=False)

    def test_subset_equal_qa_type(self):
        dataset = DriveLMImageDataset(
            message_format=QwenMessageFormat(),
            split="test",
        )
        test_set_size = min(500, len(dataset))
        subset = create_subset(
            dataset,
            test_set_size,
            by_tag=False,
            equal_distribution=True,
        )
        validate_equal_distribution(dataset, subset.indices, by_tag=False)

    def test_subset_proportional_tag(self):
        dataset = DriveLMImageDataset(
            message_format=QwenMessageFormat(),
            split="test",
        )
        test_set_size = min(500, len(dataset))
        subset = create_subset(dataset, test_set_size, by_tag=True)
        validate_proportional_distribution(dataset, subset.indices, by_tag=True)

    def test_subset_equal_tag(self):
        dataset = DriveLMImageDataset(
            message_format=QwenMessageFormat(),
            split="test",
        )
        test_set_size = min(500, len(dataset))
        subset = create_subset(
            dataset,
            test_set_size,
            by_tag=True,
            equal_distribution=True,
        )
        validate_equal_distribution(dataset, subset.indices, by_tag=True)

    def test_dataset_with_excluded_tags(self):
        exclude_question_tags = [1]

        dataset = DriveLMImageDataset(
            message_format=QwenMessageFormat(),
            split="test",
            exclude_question_tags=exclude_question_tags,
        )

        for item in dataset:
            for tag in item.tags:
                assert tag not in exclude_question_tags, (
                    f"Item {item.qa_id} has excluded tag '{tag}'"
                )

    def test_dataset_with_system_prompt(self):
        dataset = DriveLMImageDataset(
            message_format=QwenMessageFormat(),
            split="train",
            use_system_prompt=True,
        )

        self.assertGreater(
            len(dataset),
            0,
            "Dataset with system prompts should not be empty",
        )

        for item in dataset:
            self.assertIsNotNone(
                item.system_prompt,
                f"Item {item.qa_id} should have a system prompt",
            )


if __name__ == "__main__":
    unittest.main()
