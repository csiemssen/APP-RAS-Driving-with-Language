import unittest
from collections import Counter

import pytest

from src.data.basic_dataset import DriveLMImageDataset
from src.data.message_formats import QwenMessageFormat
from src.utils.logger import get_logger
from src.utils.utils import create_subset, key_object_key_to_dict

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

    def test_dataset_with_excluded_question_types(self):
        exclude_question_types = ["planning"]

        dataset = DriveLMImageDataset(
            message_format=QwenMessageFormat(),
            split="test",
            exclude_question_types=exclude_question_types,
        )

        for item in dataset:
            assert item.qa_type not in exclude_question_types, (
                f"Item {item.qa_id} has excluded question type '{item.qa_type}'"
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

    def test_all_system_prompt_overrides_are_used(self):
        config_path = "tests/test_data/test_system_prompts.yml"
        dataset = DriveLMImageDataset(
            message_format=QwenMessageFormat(),
            split="test",
            use_system_prompt=True,
            use_reasoning=True,
            use_grid=True,
            system_prompt_config_path=config_path,
        )

        # Collect all override strings from the config file
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        override_strings = set()
        # General prompt
        override_strings.add(config.get("general_prompt", ""))
        # Approach prompts
        approach = config.get("approach_prompt", {})
        override_strings.add(approach.get("base", ""))
        grid = approach.get("use_grid", {})
        override_strings.add(grid.get("enabled", ""))
        override_strings.add(approach.get("use_reasoning", ""))
        # Question type prompts
        for v in config.get("question_type_prompts", {}).values():
            override_strings.add(v)
        # Question specific prompts
        for section in config.get("question_specific_prompts", {}).values():
            for v in section.values():
                override_strings.add(v)

        override_strings = {s for s in override_strings if s}

        found = set()
        for item in dataset:
            for s in override_strings:
                if s in item.system_prompt:
                    found.add(s)
            if found == override_strings:
                break

        assert found == override_strings, (
            f"Not all override strings found in system prompts: {override_strings - found}"
        )

    def test_dataset_with_rescaling(self):
        resize_factor = 0.5
        dataset_orig = DriveLMImageDataset(
            message_format=QwenMessageFormat(),
            split="train",
            resize_factor=1.0,
            use_grid=False,
        )
        dataset_rescaled = DriveLMImageDataset(
            message_format=QwenMessageFormat(),
            split="train",
            resize_factor=resize_factor,
            use_grid=False,
        )
        self.assertGreater(
            len(dataset_rescaled),
            0,
            "Dataset with rescaled images should not be empty",
        )
        self.assertEqual(
            len(dataset_orig),
            len(dataset_rescaled),
            "Original and rescaled datasets should have the same number of items",
        )

        for orig_item, rescaled_item in zip(dataset_orig, dataset_rescaled):
            if orig_item.key_object_info:
                for (orig_key, orig_value), (
                    rescaled_key,
                    rescaled_value,
                ) in zip(
                    orig_item.key_object_info.items(),
                    rescaled_item.key_object_info.items(),
                ):
                    orig_koi = key_object_key_to_dict(orig_key)
                    rescaled_koi = key_object_key_to_dict(rescaled_key)
                    assert (
                        abs(orig_koi["x"] * resize_factor - rescaled_koi["x"]) < 1e-2
                    ), (
                        f"x coordinate not rescaled correctly: {orig_koi['x']} -> {rescaled_koi['x']}"
                    )
                    assert (
                        abs(orig_koi["y"] * resize_factor - rescaled_koi["y"]) < 1e-2
                    ), (
                        f"y coordinate not rescaled correctly: {orig_koi['y']} -> {rescaled_koi['y']}"
                    )


if __name__ == "__main__":
    unittest.main()
