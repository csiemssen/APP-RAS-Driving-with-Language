import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, TypeVar

import torch
from torch.utils.data import Dataset, Subset

from src.constants import GRID_IMG_SIZE, IMAGE_SIZE, GRID_POSITIONS
from src.data.query_item import QueryItem
from src.utils.logger import get_logger

logger = get_logger(__name__)


def flatten(list_of_lists: list[list[Any]]) -> list[Any]:
    return [x for sublist in list_of_lists for x in sublist]


def remove_nones(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def has_options(question, min_options=2):
    # Looks for at least min_options of A., B., C., D. in the question string
    matches = re.findall(r"\b[A-D]\.", question)
    return len(set(matches)) >= min_options


def sanitize_model_name(model_path: str) -> str:
    return model_path.replace("/", "_")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "auto"


def is_mps() -> bool:
    return torch.backends.mps.is_available()


def is_cuda() -> bool:
    return torch.cuda.is_available()


def extract_children(zip_path: str, out_path: str):
    tmp = Path(out_path) / "tmp"
    shutil.unpack_archive(zip_path, tmp)
    for parent in tmp.iterdir():
        for child in parent.iterdir():
            shutil.move(child, out_path)
    shutil.rmtree(tmp)


def create_subset(
    ds: Dataset, sample_size: int, by_tag=False, equal_distribution=False
) -> Dataset:
    logger.info(f"Creating subset with size {sample_size}")

    def get_keys(item: QueryItem):
        if by_tag:
            if item.tags is None:
                logger.warning(
                    f"Item {item.qa_id} has no tag, returning empty list for 'tag' by parameter."
                )
            return item.tags
        else:
            return [item.qa_type]

    total = len(ds)
    if total == 0:
        logger.warning("Dataset is empty, returning empty subset.")
        return Subset(ds, [])

    indices_by_key: Dict[str, List[int]] = {}
    for idx in range(len(ds)):
        item: QueryItem = ds[idx]
        for key in get_keys(item):
            indices_by_key.setdefault(key, []).append(idx)

    keys = list(indices_by_key.keys())
    n_keys = len(keys)

    sampled_indices = set()
    if equal_distribution:
        n_per_key = max(1, sample_size // n_keys)
        for key in keys:
            indices = indices_by_key[key]
            sampled = sorted(indices)[:n_per_key]
            sampled_indices.update(sampled)
    else:
        for key, indices in indices_by_key.items():
            n = max(1, int(len(indices) * sample_size / total))
            sampled = sorted(indices)[:n]
            sampled_indices.update(sampled)

    sampled_indices = list(sampled_indices)[:sample_size]

    sampled_count_by_key = {
        k: len([i for i in v if i in sampled_indices])
        for k, v in indices_by_key.items()
    }
    count_by_type = {
        qa_type: len(indices) for qa_type, indices in indices_by_key.items()
    }
    percent_by_qa_type = {
        qa_type: count / total for qa_type, count in count_by_type.items()
    }
    sampled_pct_by_type = {
        qa_type: count / sample_size for qa_type, count in sampled_count_by_key.items()
    }

    logger.debug(f"Original distribution (count): {count_by_type}")
    logger.debug(f"Original distribution (percent): {percent_by_qa_type}")
    logger.debug(f"Sampled distribution (count): {sampled_count_by_key}")
    logger.debug(f"Sampled distribution (percent): {sampled_pct_by_type}")

    subset = Subset(ds, sampled_indices)
    logger.info(f"Created subset with size {len(subset)}")
    return subset


T = TypeVar("T", int, float)


def tuple_cast(t: Tuple[float, ...], typ: Type[T]) -> Tuple[T, ...]:
    return tuple(typ(x) for x in t)


def tuple_mul(t: Tuple[float, float], scalar: float) -> Tuple[float, float]:
    return (t[0] * scalar, t[1] * scalar)


def get_resize_image_size(resize_factor: float, grid: bool = False) -> Tuple[int, int]:
    if grid:
        size = tuple_mul(GRID_IMG_SIZE, resize_factor)
    else:
        size = tuple_mul(IMAGE_SIZE, resize_factor)
    return tuple_cast(size, int)


def find_key_objects(text: str) -> List[str]:
    pattern = r"<c\d+,CAM_[A-Z_]+,\d+\.?\d*,\d+\.?\d*>"
    matches = re.findall(pattern, text)
    return matches


def key_object_str_to_dict(text: str) -> Dict[str, Any]:
    pattern = r"<c(\d+),CAM_([A-Z_]+),(\d+\.?\d*),(\d+\.?\d*)>"
    matches = re.findall(pattern, text)
    return (
        {
            "id": f"c{matches[0][0]}",
            "camera": f"CAM_{matches[0][1]}",
            "x": float(matches[0][2]),
            "y": float(matches[0][3]),
        }
        if matches
        else {}
    )


def key_object_dict_to_str(key_object: Dict[str, Any]) -> str:
    return f"<{key_object['id']},{key_object['camera']},{key_object['x']},{key_object['y']}>"


def scale_key_object_point(
    point: tuple[float, float], resize_factor: float
) -> tuple[float, float]:
    return float.__round__(point[0] * resize_factor, 2), float.__round__(point[1] * resize_factor, 2)


def normalise_key_object_infos(
    data,
    resize_factor: float,
    use_grid: bool,
) -> tuple[str, dict[str, Any]]:
    for _, scene_data in data.items():
        for _, key_frame_data in scene_data["key_frames"].items():
            key_object_infos = key_frame_data["key_object_infos"]
            if not key_object_infos:
                continue
            normalised_key_object_infos = {}
            for key, value in key_object_infos.items():
                normalised_key = normalise_key_object_descriptor(
                    key,
                    resize_factor,
                    use_grid,
                )
                new_value = value.copy()
                if "2d_bbox" in value:
                    koi_dict = key_object_str_to_dict(key)
                    x1, y1, x2, y2 = value["2d_bbox"]
                    if use_grid:
                        x1, y1 = map_camera_point_to_grid_point((x1, y1), koi_dict["camera"])
                        x2, y2 = map_camera_point_to_grid_point((x2, y2), koi_dict["camera"])

                    x1, y1 = scale_key_object_point(
                        (x1, y1),
                        resize_factor,
                    )
                    x2, y2 = scale_key_object_point(
                        (x2, y2),
                        resize_factor,
                    )

                    new_value["2d_bbox"] = (x1, y1, x2, y2)
                normalised_key_object_infos[normalised_key] = new_value
            key_frame_data["key_object_infos"] = normalised_key_object_infos

    return data


def normalise_key_object_descriptor(
    key_object_descriptor: str, resize_factor: float, use_grid: bool
):
    koi_dict = key_object_str_to_dict(key_object_descriptor)

    if not koi_dict:
        logger.warning(
            f"Key object string '{key_object_descriptor}' could not be parsed."
        )
        return key_object_descriptor

    # Map to grid coordinates first as it uses the orginal image size
    if use_grid:
        new_x, new_y = map_camera_point_to_grid_point(
            (koi_dict["x"], koi_dict["y"]), koi_dict["camera"]
        )
    else:
        new_x, new_y = koi_dict["x"], koi_dict["y"]

    new_x, new_y = scale_key_object_point((new_x, new_y), resize_factor)

    koi_dict["x"] = new_x
    koi_dict["y"] = new_y

    return key_object_dict_to_str(koi_dict)


def normalise_key_objects_in_text(
    text: str,
    resize_factor: float,
    use_grid: bool,
) -> str:
    descriptors = find_key_objects(text)
    for desc in descriptors:
        norm_desc = normalise_key_object_descriptor(
            desc,
            resize_factor,
            use_grid,
        )
        text = text.replace(desc, norm_desc)
    return text


def map_camera_point_to_grid_point(
    point: Tuple[float, float],
    cam_name: str,
) -> Tuple[float, float]:
    # The model does not always output existing cam names
    if cam_name not in GRID_POSITIONS:
        return point
    col, row = GRID_POSITIONS[cam_name]
    img_height, img_width = IMAGE_SIZE
    x_offset = col * img_width
    y_offset = row * img_height
    return (point[0] + x_offset, point[1] + y_offset)
