import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Type, TypeVar

import torch
from torch.utils.data import Dataset, Subset

from src.constants import (
    BEV_AND_FRONT_CAM_IMG_SIZE,
    BEV_IMG_SIZE,
    GRID_IMG_SIZE,
    GRID_POSITIONS,
    IMAGE_SIZE,
)
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


def get_resize_image_size(
    resize_factor: float,
    grid: bool = False,
    bev: bool = False,
    front_cam: bool = False,
) -> Tuple[int, int]:
    if grid:
        size = tuple_mul(GRID_IMG_SIZE, resize_factor)
    elif bev and not front_cam:
        size = tuple_mul(BEV_IMG_SIZE, resize_factor)
    elif bev and front_cam:
        size = tuple_mul(BEV_AND_FRONT_CAM_IMG_SIZE, resize_factor)
    else:
        size = tuple_mul(IMAGE_SIZE, resize_factor)
    return tuple_cast(size, int)


def find_key_objects(text: str) -> List[str]:
    return re.findall(r"<c\d+,CAM_[A-Z_]+,\d+\.?\d*,\d+\.?\d*>", text)


def key_object_key_to_dict(key: str) -> Dict[str, Any]:
    match = re.match(r"<c(\d+),CAM_([A-Z_]+),(\d+\.?\d*),(\d+\.?\d*)>", key)
    return (
        {
            "id": f"c{match.group(1)}",
            "camera": f"CAM_{match.group(2)}",
            "x": float(match.group(3)),
            "y": float(match.group(4)),
        }
        if match
        else {}
    )


def key_object_dict_to_key(key_obj: Dict[str, Any]) -> str:
    return f"<{key_obj['id']},{key_obj['camera']},{key_obj['x']},{key_obj['y']}>"


def scale_point(point: Tuple[float, float], factor: float) -> Tuple[float, float]:
    return round(point[0] * factor, 2), round(point[1] * factor, 2)


def _transform_key_object_infos(
    key_object_infos: Dict[str, Any],
    key_transform: Callable[[str], str],
    bbox_transform: Callable[
        [Tuple[float, float, float, float]], Tuple[float, float, float, float]
    ] = None,
) -> Dict[str, Any]:
    result = {}
    for key, value in key_object_infos.items():
        new_key = key_transform(key)
        new_value = value.copy()
        if "2d_bbox" in value and bbox_transform:
            new_value["2d_bbox"] = bbox_transform(value["2d_bbox"])
        result[new_key] = new_value
    return result


def _transform_key_objects_in_text(
    text: str, transform_fn: Callable[[str], str]
) -> str:
    keys = find_key_objects(text)
    transformed_keys = [transform_fn(key) for key in keys]
    for orig, new in zip(keys, transformed_keys):
        text = text.replace(orig, new)
    return text


def scale_key_object_key(key: str, factor: float) -> str:
    obj = key_object_key_to_dict(key)
    if not obj:
        logger.warning(f"Key object string '{key}' could not be parsed.")
        return key
    obj["x"], obj["y"] = scale_point((obj["x"], obj["y"]), factor)
    return key_object_dict_to_key(obj)


def scale_bbox(
    bbox: Tuple[float, float, float, float], factor: float
) -> Tuple[float, float, float, float]:
    x1, y1 = scale_point((bbox[0], bbox[1]), factor)
    x2, y2 = scale_point((bbox[2], bbox[3]), factor)
    return x1, y1, x2, y2


def scale_key_object_infos(
    key_object_infos: Dict[str, Any], resize_factor: float
) -> Dict[str, Any]:
    return _transform_key_object_infos(
        key_object_infos,
        lambda k: scale_key_object_key(k, resize_factor),
        lambda bbox: scale_bbox(bbox, resize_factor),
    )


def camera_key_object_infos_to_grid(
    key_object_infos: Dict[str, Any],
) -> Dict[str, Any]:
    return _transform_key_object_infos(
        key_object_infos,
        camera_key_object_key_to_grid,
        lambda bbox: (
            *camera_point_to_grid_point(
                (bbox[0], bbox[1]),
                key_object_key_to_dict(list(key_object_infos.keys())[0])["camera"],
            ),
            *camera_point_to_grid_point(
                (bbox[2], bbox[3]),
                key_object_key_to_dict(list(key_object_infos.keys())[0])["camera"],
            ),
        ),
    )


def grid_key_object_infos_to_camera(
    key_object_infos: Dict[str, Any],
) -> Dict[str, Any]:
    return _transform_key_object_infos(
        key_object_infos,
        grid_key_objects_key_to_camera,
        lambda bbox: (
            *grid_point_to_camera_point(
                (bbox[0], bbox[1]),
                key_object_key_to_dict(list(key_object_infos.keys())[0])["camera"],
            ),
            *grid_point_to_camera_point(
                (bbox[2], bbox[3]),
                key_object_key_to_dict(list(key_object_infos.keys())[0])["camera"],
            ),
        ),
    )


def camera_key_object_key_to_grid(key_object_key: str) -> str:
    obj = key_object_key_to_dict(key_object_key)
    if not obj:
        logger.warning(f"Key object string '{key_object_key}' could not be parsed.")
        return key_object_key
    obj["x"], obj["y"] = camera_point_to_grid_point((obj["x"], obj["y"]), obj["camera"])
    return key_object_dict_to_key(obj)


def grid_key_objects_key_to_camera(key_object_key: str) -> str:
    obj = key_object_key_to_dict(key_object_key)
    if not obj:
        logger.warning(f"Key object string '{key_object_key}' could not be parsed.")
        return key_object_key
    obj["x"], obj["y"] = grid_point_to_camera_point((obj["x"], obj["y"]), obj["camera"])
    return key_object_dict_to_key(obj)


def normalize_key_object_infos(
    key_object_infos: Dict[str, Any],
    resize_factor: float = 1.0,
    use_grid: bool = False,
) -> Dict[str, Any]:
    if use_grid:
        key_object_infos = camera_key_object_infos_to_grid(key_object_infos)
    return scale_key_object_infos(key_object_infos, resize_factor)


def denormalize_key_object_infos(
    key_object_infos: Dict[str, Any],
    resize_factor: float = 1.0,
    use_grid: bool = False,
) -> Dict[str, Any]:
    infos = scale_key_object_infos(key_object_infos, 1.0 / resize_factor)
    if use_grid:
        infos = grid_key_object_infos_to_camera(infos)
    return infos


def normalize_key_objects_in_text(
    text: str,
    resize_factor: float = 1.0,
    use_grid: bool = False,
) -> str:
    def transform(key: str) -> str:
        norm_key = key
        if use_grid:
            norm_key = camera_key_object_key_to_grid(norm_key)
        norm_key = scale_key_object_key(norm_key, resize_factor)
        return norm_key

    return _transform_key_objects_in_text(text, transform)


def denormalize_key_objects_in_text(
    text: str,
    resize_factor: float = 1.0,
    use_grid: bool = False,
) -> str:
    def transform(key: str) -> str:
        denorm_key = key
        denorm_key = scale_key_object_key(denorm_key, 1.0 / resize_factor)
        if use_grid:
            denorm_key = grid_key_objects_key_to_camera(denorm_key)
        return denorm_key

    return _transform_key_objects_in_text(text, transform)


def camera_point_to_grid_point(
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


def grid_point_to_camera_point(
    point: Tuple[float, float],
    cam_name: str,
) -> Tuple[float, float]:
    if cam_name not in GRID_POSITIONS:
        return point
    col, row = GRID_POSITIONS[cam_name]
    img_height, img_width = IMAGE_SIZE
    x_offset = col * img_width
    y_offset = row * img_height
    return (point[0] - x_offset, point[1] - y_offset)
