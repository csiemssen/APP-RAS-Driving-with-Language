from typing import Any, Dict, List


def get_approach_kwargs(approaches: List[str]) -> Dict[str, Any]:
    approach_kwargs_map = {
        "front_cam": {"front_cam": True},
        "image_grid": {"use_grid": True},
        "descriptor_qas": {"use_augmented": True},
        "add_kois": {"add_kois": True},
        "add_bev": {"add_bev": True},
        "reasoning": {"use_reasoning": True},
        "system_prompt": {"use_system_prompt": True},
        # Add more approaches here as needed
    }
    kwargs = {}
    for a in approaches:
        if a in approach_kwargs_map:
            kwargs.update(approach_kwargs_map[a])
    return kwargs


def get_approach_name(approaches: List[str]) -> str:
    return "_".join(to_pascal_case(a) for a in approaches)


def to_pascal_case(s: str) -> str:
    return "".join(word.capitalize() for word in s.split("_"))
