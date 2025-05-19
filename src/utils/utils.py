from typing import Any

def flatten(list_of_lists: list[list[Any]]) -> list[Any]:
    return [x for l in list_of_lists for x in l]

