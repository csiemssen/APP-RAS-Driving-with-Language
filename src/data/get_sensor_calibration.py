import polars as pl
from tqdm import tqdm

from src.constants import nuscenes_dir


def get_sample_data_and_calibrated_camera_lf() -> pl.LazyFrame:
    sample_data_lf = pl.read_json(nuscenes_dir / "sample_data.json").lazy()
    sample_data_lf = sample_data_lf.filter(
        pl.col("is_key_frame") == True  # noqa: E712
    ).select([
        "token",
        "sample_token",
        "calibrated_sensor_token"
    ])
    calibrated_camera_lf = pl.read_json(nuscenes_dir / "calibrated_sensor.json").lazy()
    calibrated_camera_lf = calibrated_camera_lf.filter(
        pl.col("camera_intrinsic").len() != 0
    )
    sensor_lf = pl.read_json(nuscenes_dir / "sensor.json").lazy()
    calibrated_camera_with_sensor_type_lf = calibrated_camera_lf.join(
        sensor_lf,
        left_on="sensor_token",
        right_on="token",
        suffix="_sensor",
    )
    return sample_data_lf.join(
        calibrated_camera_with_sensor_type_lf, 
        left_on="calibrated_sensor_token", 
        right_on="token", 
        suffix="_calibrated"
    )


cameras = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

class CameraCalibration:
    camera_intrinsic: list[list[float]]
    translation: list[float]
    rotation: list[float]

    def __init__(self, camera_intrinsic, translation, rotation):
        self.camera_intrinsic = camera_intrinsic
        self.translation = translation
        self.rotation = rotation


def get_camera_calibration(lf: pl.LazyFrame, key_frame_id) -> dict[str, CameraCalibration]:
    calibration_per_camera = {}
    for cam in cameras:
        calibration = lf.filter(
            pl.col("channel") == cam,
            pl.col("sample_token") == key_frame_id
        ).select(
            "translation",
            "rotation",
            "camera_intrinsic"
        ).collect().to_dict()
        assert len(calibration["translation"]) == 1
        calibration_per_camera[cam] = CameraCalibration(
            camera_intrinsic=calibration["camera_intrinsic"][0].to_list(),
            translation=calibration["translation"][0].to_list(),
            rotation=calibration["rotation"][0].to_list(),
        )
    return calibration_per_camera

def get_calibration(data: dict):
    lf = get_sample_data_and_calibrated_camera_lf()
    for _, scene in tqdm(data.items(), desc="Fetching camera calibration data"):
        for key_frame_id, key_frame in scene["key_frames"].items():
            key_frame["camera_calibration"] = get_camera_calibration(lf, key_frame_id)
    return data
