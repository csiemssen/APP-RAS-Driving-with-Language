import os

from ultralytics import YOLO

from src.constants import drivelm_dir


def generate_yolo_kois(data, max_results_per_cam: int = 5):
    model = YOLO("yolo11n.pt")
    for _, scene_obj in data.items():
        for _, key_frame in scene_obj["key_frames"].items():
            image_paths_raw = key_frame["image_paths"]
            i = 0
            kois = []
            for camera, image_path in image_paths_raw.items():
                results = model(os.path.join(drivelm_dir, image_path))[:max_results_per_cam]
                center_points = [
                    (xywh[0], xywh[1]) for res in results for xywh in res.boxes.xywh
                ]
                categories = [
                    res.names[cls.item()]
                    for res in results
                    for cls in res.boxes.cls.int()
                ]
                for j in range(len(center_points)):
                    i += 1
                    kois.append(
                        (
                            f"<c{i},{camera},{center_points[j][0]},{center_points[j][1]}>",
                            categories[j],
                        )
                    )
            key_frame["key_object_infos"] = {
                descriptor: {
                    "Category": category,
                }
                for descriptor, category in kois
            }

    return data
