import shutil
from pathlib import Path
from typing import Any


def flatten(list_of_lists: list[list[Any]]) -> list[Any]:
    return [x for l in list_of_lists for x in l]


def remove_nones(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def extract_children(zip_path: str, out_path: str):
    tmp = Path(out_path) / "tmp"
    shutil.unpack_archive(zip_path, tmp)
    for parent in tmp.iterdir():
        for child in parent.iterdir():
            shutil.move(child, out_path)
    shutil.rmtree(tmp)
