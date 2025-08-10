import unittest

from src.constants import GRID_POSITIONS, IMAGE_SIZE
from src.utils.utils import (
    denormalize_key_object_infos,
    denormalize_key_objects_in_text,
    find_key_objects,
    key_object_key_to_dict,
    normalize_key_object_infos,
    normalize_key_objects_in_text,
)


class TestDriveLMImageDataset(unittest.TestCase):
    def test_key_object_descriptor_scaling_and_inverse(self):
        text = "What object would consider <c1,CAM_BACK,1088.3,497.5> to be most relevant to its decision?"
        resize_factor = 0.5

        scaled_text = normalize_key_objects_in_text(
            text,
            resize_factor=resize_factor,
            use_grid=False,
        )

        scaled_koi = key_object_key_to_dict(find_key_objects(scaled_text)[0])
        orig_koi = key_object_key_to_dict(find_key_objects(text)[0])

        assert abs(orig_koi["x"] * resize_factor - scaled_koi["x"]) < 1e-2, (
            f"x coordinate not scaled correctly: {orig_koi['x']} -> {scaled_koi['x']}"
        )
        assert abs(orig_koi["y"] * resize_factor - scaled_koi["y"]) < 1e-2, (
            f"y coordinate not scaled correctly: {orig_koi['y']} -> {scaled_koi['y']}"
        )

        unscaled_text = denormalize_key_objects_in_text(
            scaled_text,
            resize_factor=resize_factor,
            use_grid=False,
        )
        unscaled_koi = key_object_key_to_dict(find_key_objects(unscaled_text)[0])
        assert abs(unscaled_koi["x"] - orig_koi["x"]) < 1e-2, (
            f"x coordinate not unscaled correctly: {unscaled_koi['x']} -> {orig_koi['x']}"
        )
        assert abs(unscaled_koi["y"] - orig_koi["y"]) < 1e-2, (
            f"y coordinate not unscaled correctly: {unscaled_koi['y']} -> {orig_koi['y']}"
        )

    def test_key_object_descriptor_scaling_with_grid(self):
        text = "What object would consider <c1,CAM_BACK,1088.3,497.5> to be most relevant to its decision?"
        resize_factor = 0.5

        scaled_text = normalize_key_objects_in_text(
            text,
            resize_factor=resize_factor,
            use_grid=True,
        )

        scaled_koi = key_object_key_to_dict(find_key_objects(scaled_text)[0])
        orig_koi = key_object_key_to_dict(find_key_objects(text)[0])

        cam_name = orig_koi["camera"]
        col, row = GRID_POSITIONS[cam_name]
        img_height, img_width = IMAGE_SIZE
        x_offset = col * img_width
        y_offset = row * img_height

        expected_x = (orig_koi["x"] + x_offset) * resize_factor
        expected_y = (orig_koi["y"] + y_offset) * resize_factor

        assert abs(scaled_koi["x"] - expected_x) < 1e-2, (
            f"x coordinate not grid-mapped and scaled correctly: {scaled_koi['x']} != {expected_x}"
        )
        assert abs(scaled_koi["y"] - expected_y) < 1e-2, (
            f"y coordinate not grid-mapped and scaled correctly: {scaled_koi['y']} != {expected_y}"
        )

        unscaled_text = denormalize_key_objects_in_text(
            scaled_text,
            resize_factor=resize_factor,
            use_grid=True,
        )

        unscaled_koi = key_object_key_to_dict(find_key_objects(unscaled_text)[0])
        assert abs(unscaled_koi["x"] - orig_koi["x"]) < 1e-2, (
            f"x coordinate not unscaled correctly: {unscaled_koi['x']} -> {orig_koi['x']}"
        )
        assert abs(unscaled_koi["y"] - orig_koi["y"]) < 1e-2, (
            f"y coordinate not unscaled correctly: {unscaled_koi['y']} -> {orig_koi['y']}"
        )

    def test_key_object_infos_scaling_and_inverse_with_grid(self):
        key_object_infos = {
            "<c1,CAM_BACK,1088.3,497.5>": {
                "Category": "Vehicle",
                "Status": "Moving",
                "Visual_description": "Brown SUV.",
                "2d_bbox": [966.6, 403.3, 1224.1, 591.7],
            }
        }
        resize_factor = 1
        cam_name = "CAM_BACK"
        col, row = GRID_POSITIONS[cam_name]
        img_height, img_width = IMAGE_SIZE
        x_offset = col * img_width
        y_offset = row * img_height

        normalized_infos = normalize_key_object_infos(
            key_object_infos,
            resize_factor=resize_factor,
            use_grid=True,
        )

        norm_key = list(normalized_infos.keys())[0]
        norm_koi = key_object_key_to_dict(norm_key)
        orig_koi = key_object_key_to_dict(list(key_object_infos.keys())[0])

        expected_x = (orig_koi["x"] + x_offset) * resize_factor
        expected_y = (orig_koi["y"] + y_offset) * resize_factor
        assert abs(norm_koi["x"] - expected_x) < 1e-2
        assert abs(norm_koi["y"] - expected_y) < 1e-2

        orig_bbox = key_object_infos[list(key_object_infos.keys())[0]]["2d_bbox"]
        norm_bbox = normalized_infos[norm_key]["2d_bbox"]
        expected_bbox = [
            (orig_bbox[0] + x_offset) * resize_factor,
            (orig_bbox[1] + y_offset) * resize_factor,
            (orig_bbox[2] + x_offset) * resize_factor,
            (orig_bbox[3] + y_offset) * resize_factor,
        ]
        for a, b in zip(norm_bbox, expected_bbox):
            assert abs(a - b) < 1e-2

        denormalized_infos = denormalize_key_object_infos(
            normalized_infos,
            resize_factor=resize_factor,
            use_grid=True,
        )

        denorm_key = list(denormalized_infos.keys())[0]
        denorm_koi = key_object_key_to_dict(denorm_key)
        assert abs(denorm_koi["x"] - orig_koi["x"]) < 1e-2
        assert abs(denorm_koi["y"] - orig_koi["y"]) < 1e-2

        denorm_bbox = denormalized_infos[denorm_key]["2d_bbox"]
        for a, b in zip(denorm_bbox, orig_bbox):
            assert abs(a - b) < 1e-2
