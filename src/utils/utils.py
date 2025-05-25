from typing import Any


def flatten(list_of_lists: list[list[Any]]) -> list[Any]:
    return [x for l in list_of_lists for x in l]


def remove_nones(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}
