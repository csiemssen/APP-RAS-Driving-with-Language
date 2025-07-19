import json
import os
import re
from typing import Optional
from tqdm import tqdm

import numpy as np
from ultralytics import YOLO

from src.constants import drivelm_dir
from src.data.load_dataset import load_dataset


model = YOLO("yolo11n.pt")


def calculate_error(kois_per_key_frame_gt: dict, kois_per_key_frame_yolo: dict):
    num_gt_nums = 0
    true_positives = 0
    for key_frame_id in kois_per_key_frame_gt.keys():
        kois_gt = kois_per_key_frame_gt[key_frame_id].keys()
        kois_yolo = kois_per_key_frame_yolo[key_frame_id].keys()
        gt_nums = [num for koi in kois_gt for num in re.findall(r"\d+\.\d+", koi)]
        yolo_nums = [num for koi in kois_yolo for num in re.findall(r"\d+\.\d+", koi)]

        if len(yolo_nums) % 2 != 0:
            yolo_nums = yolo_nums[:-1]
        yolo_nums = np.array(
            [list(map(float, x.split()))[0] for x in yolo_nums]
        ).reshape(-1, 2)
        gt_nums = np.array([list(map(float, x.split()))[0] for x in gt_nums]).reshape(
            -1, 2
        )
        num_gt_nums += len(gt_nums)
        # NOTE: We want to attain something close to true_positives == len(kois_gt)
        for pred in yolo_nums:
            closest_distance = float("inf")
            for gt in gt_nums:
                distance = np.sum(np.abs(pred - gt))
                if distance < closest_distance:
                    closest_distance = distance

            if closest_distance < 16:
                true_positives += 1

    return true_positives, num_gt_nums


def key_frame_id_and_key_frames():
    data = load_dataset("train")
    for scene_id in data.keys():
        scene_obj = data[scene_id]["key_frames"]
        for key_frame_id in scene_obj.keys():
            yield key_frame_id, scene_obj[key_frame_id]


def get_image_paths_and_kois_per_key_frame_yolo(max_samples: Optional[int] = None):
    kois_per_key_frame = {}
    num_samples = 0
    print("Generating KOIs")
    for key_frame_id, key_frame in tqdm(key_frame_id_and_key_frames()):
        if max_samples and num_samples == max_samples:
            break
        image_paths_raw = key_frame["image_paths"]
        i = 0
        kois = []
        for camera, image_path in image_paths_raw.items():
            results = model(os.path.join(drivelm_dir, image_path))
            center_points = [(xywh[0], xywh[1]) for res in results for xywh in res.boxes.xywh][:5]
            categories = [res.names[cls.item()] for res in results for cls in res.boxes.cls.int()][:5]
            for j in range(len(center_points)):
                i += 1
                kois.append((f"<c{i},{camera},{center_points[j][0]},{center_points[j][1]}>",categories[j]))
        kois_per_key_frame[key_frame_id] = {
            descriptor: {
                "Category": category,
            } for descriptor, category in kois
        }
        num_samples += 1
    return kois_per_key_frame


def get_image_paths_and_kois_per_key_frame_gt(max_samples: Optional[int] = None):
    kois_per_key_frame = {}
    num_samples = 0
    for key_frame_id, key_frame in key_frame_id_and_key_frames():
        if max_samples and num_samples == max_samples:
            break
        key_object_infos = key_frame["key_object_infos"]
        kois_per_key_frame[key_frame_id] = key_object_infos
        num_samples += 1
    return kois_per_key_frame


if __name__=="__main__":
    kois_per_key_frame_gt = get_image_paths_and_kois_per_key_frame_gt()
    kois_per_key_frame_yolo = get_image_paths_and_kois_per_key_frame_yolo()
    matches, num_kois_gt = calculate_error(kois_per_key_frame_gt, kois_per_key_frame_yolo)
    print(f"Number of KOIs in GT: {num_kois_gt}, Number of matches: {matches}")
