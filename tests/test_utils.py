import unittest
from src.utils.utils import (
    normalise_key_objects_in_text,
    key_object_str_to_dict,
    find_key_objects,
)


class TestDriveLMImageDataset(unittest.TestCase):
    def test_key_object_descriptor_scaling_and_inverse(self):
        text = "What object would consider <c1,CAM_BACK,1088.3,497.5> to be most relevant to its decision?"
        resize_factor = 0.5

        scaled_text = normalise_key_objects_in_text(
            text,
            resize_factor=resize_factor,
            use_grid=False,
        )

        scaled_koi = key_object_str_to_dict(find_key_objects(scaled_text)[0])
        orig_koi = key_object_str_to_dict(find_key_objects(text)[0])

        assert abs(orig_koi["x"] * resize_factor - scaled_koi["x"]) < 1e-2, (
            f"x coordinate not scaled correctly: {orig_koi['x']} -> {scaled_koi['x']}"
        )
        assert abs(orig_koi["y"] * resize_factor - scaled_koi["y"]) < 1e-2, (
            f"y coordinate not scaled correctly: {orig_koi['y']} -> {scaled_koi['y']}"
        )

        unscaled_text = normalise_key_objects_in_text(
            scaled_text,
            resize_factor=1 / resize_factor,
            use_grid=False,
        )
        unscaled_koi = key_object_str_to_dict(find_key_objects(unscaled_text)[0])
        assert abs(unscaled_koi["x"] - orig_koi["x"]) < 1e-2, (
            f"x coordinate not unscaled correctly: {unscaled_koi['x']} -> {orig_koi['x']}"
        )
        assert abs(unscaled_koi["y"] - orig_koi["y"]) < 1e-2, (
            f"y coordinate not unscaled correctly: {unscaled_koi['y']} -> {orig_koi['y']}"
        )
