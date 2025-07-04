from typing import Any, Dict, List


def get_approach_kwargs(approaches: List[str]) -> Dict[str, Any]:
    approach_kwargs_map = {
        "image_grid": {"use_grid": True},
        "descriptor_qas": {"use_augmented": True},
        "reasoning": {"use_reasoning": True},
        # Add more approaches here as needed
    }
    kwargs = {}
    for a in approaches:
        if a in approach_kwargs_map:
            kwargs.update(approach_kwargs_map[a])
    return kwargs


def get_approach_name(approaches: List[str]) -> str:
    return "_".join(approaches)
