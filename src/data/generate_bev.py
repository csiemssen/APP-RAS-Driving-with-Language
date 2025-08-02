import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from tqdm import tqdm

from src.constants import bev_dir, drivelm_dir
from src.data.get_sensor_calibration import CameraCalibration
from src.utils.utils import get_logger, key_object_str_to_dict


logger = get_logger(__name__)


def generate_bev_from_detections(
        calibration: dict[str, CameraCalibration], 
        kois: dict,
    ) -> np.ndarray:
    """
    Generates a Bird's-Eye View (BEV) map from detected objects for a keyframe,
    using nuScenes camera calibration information.

    Args:
        calibration: A dictionary where keys are camera names (e.g., 'CAM_FRONT')
                     and values are CameraCalibration objects.
        kois: A dictionary where keys contain camera names and values contain
              object detection information including 2d_bbox and Category.
    Returns:
        A NumPy array representing the BEV map (H, W, 3).
    """
    bev_map_res_m_per_pixel = 0.1
    bev_map_x_range = 50.0
    bev_map_y_range = 50.0

    # --- BEV Map Initialization ---
    # Calculate min/max extents based on ranges to center ego (0,0)
    x_min_m = -bev_map_x_range / 2.0
    y_min_m = -bev_map_y_range / 2.0

    bev_map_width_pixels = int(bev_map_x_range / bev_map_res_m_per_pixel)
    bev_map_height_pixels = int(bev_map_y_range / bev_map_res_m_per_pixel)

    bev_map = np.zeros((bev_map_height_pixels, bev_map_width_pixels, 3), dtype=np.uint8)
    bev_map.fill(20)

    all_projected_objects = []

    total_items = 0
    for camera_name, cam_calib in calibration.items():
        current_keys = [koi_key for koi_key in kois.keys() if camera_name == key_object_str_to_dict(koi_key)["camera"]]
        current_identifiers = [key_object_str_to_dict(k)["id"] for k in current_keys]
        current_camera_kois = [kois[k] for k in current_keys]
        current_camera_boxes = [koi["2d_bbox"] for koi in current_camera_kois]
        total_items += len(current_camera_boxes)
        current_camera_names = [koi["Category"] for koi in current_camera_kois]

        K = np.array(cam_calib.camera_intrinsic, dtype=np.float64)
        t_camera_to_ego = np.array(cam_calib.translation, dtype=np.float64) # (x, y, z)
        q_camera_to_ego = np.array(cam_calib.rotation, dtype=np.float64) # (w, x, y, z)

        # 1. Convert quaternion to rotation matrix: R_ego_from_camera
        # nuScenes quaternion is (w, x, y, z) -> scipy Rotation.from_quat expects (x, y, z, w)
        r_ego_from_camera_scipy = R_scipy.from_quat([q_camera_to_ego[1], q_camera_to_ego[2], q_camera_to_ego[3], q_camera_to_ego[0]])
        R_ego_from_camera = r_ego_from_camera_scipy.as_matrix() # 3x3 rotation matrix from camera to ego

        for i in range(len(current_camera_boxes)):
            bbox = current_camera_boxes[i]
            obj_name = current_camera_names[i]

            # Use the bottom-center of the 2D bounding box as the ground contact point heuristic.
            x1, y1, x2, y2 = bbox
            bottom_center_2d = np.array([(x1 + x2) / 2, y2], dtype=np.float64)

            # --- Project 2D image point back to 3D on the ground plane (Z=0 in ego frame) ---

            # Convert 2D image point to a 3D ray direction in the camera frame (normalized coordinates).
            uv_hom = np.array([bottom_center_2d[0], bottom_center_2d[1], 1.0], dtype=np.float64).reshape(3, 1)
            K_inv = np.linalg.inv(K)
            ray_direction_camera_frame = np.dot(K_inv, uv_hom).flatten()

            # Transform the ray from the camera frame to the ego vehicle frame.
            ray_origin_ego = t_camera_to_ego
            ray_direction_ego = np.dot(R_ego_from_camera, ray_direction_camera_frame)

            # Intersect the ray with the ground plane (Z_ego = 0).
            if np.isclose(ray_direction_ego[2], 0.0):
                continue # Ray is parallel or near-parallel to ground plane

            lam = -ray_origin_ego[2] / ray_direction_ego[2]

            # Ensure the intersection point is in front of the camera (positive lambda).
            if lam < 0:
                continue

            point_3d_ego = ray_origin_ego + lam * ray_direction_ego

            # Store the projected object's information
            projected_object_info = {
                'class': obj_name,
                'x_ego': -point_3d_ego[1],
                'y_ego': point_3d_ego[0],
                'z_ego': point_3d_ego[2], # Should be close to 0
                'camera_name': camera_name,
                'original_bbox': bbox,
                'identifier': current_identifiers[i],
            }
            all_projected_objects.append(projected_object_info)

    # --- Remove Duplicate Objects ---
    # Group objects by spatial proximity and class, keep the one with best visibility
    unique_objects = []
    proximity_threshold = 10.0
    duplicates_removed = 0
    
    for obj in all_projected_objects:
        is_duplicate = False
        for unique_obj in unique_objects:
            # Check if objects are of same class and spatially close
            if (obj['class'] == unique_obj['class'] and
                # TODO: Tune the prox threshold
                np.sqrt((obj['x_ego'] - unique_obj['x_ego'])**2 + 
                       (obj['y_ego'] - unique_obj['y_ego'])**2) < proximity_threshold):
                
                # Keep the object from the camera that provides better view
                # Prefer front cameras for forward objects, side cameras for side objects, etc.
                current_distance = np.sqrt(obj['x_ego']**2 + obj['y_ego']**2)
                unique_distance = np.sqrt(unique_obj['x_ego']**2 + unique_obj['y_ego']**2)
                
                # Replace if current object is closer or from a more appropriate camera
                if (current_distance < unique_distance or 
                    _is_better_camera_view(obj, unique_obj)):
                    unique_objects.remove(unique_obj)
                    unique_objects.append(obj)
                else:
                    duplicates_removed += 1
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_objects.append(obj)

    # --- Render Projected Objects onto the BEV Map ---
    for obj_info in unique_objects:
        x_ego = obj_info['x_ego']
        y_ego = obj_info['y_ego']
        obj_class = obj_info['class']
        identifier = obj_info['identifier']

        col_bev = int((x_ego - x_min_m) / bev_map_res_m_per_pixel)
        row_bev = int(bev_map_height_pixels - 1 - ((y_ego - y_min_m) / bev_map_res_m_per_pixel))

        if 0 <= col_bev < bev_map_width_pixels and 0 <= row_bev < bev_map_height_pixels:
            if 'car' in obj_class.lower() or 'truck' in obj_class.lower() or 'bus' in obj_class.lower() or \
               'bicycle' in obj_class.lower() or 'motorcycle' in obj_class.lower():
                car_width_bev = int(1.5 / bev_map_res_m_per_pixel)
                car_length_bev = int(3.0 / bev_map_res_m_per_pixel)
                color = (0, 255, 255) # Yellow (BGR)

                if (0 <= (col_bev - car_width_bev) and (col_bev + car_width_bev) < bev_map_width_pixels
                    and 0 <= (row_bev - car_length_bev) and (row_bev + car_length_bev) < bev_map_height_pixels):
                    cv2.rectangle(bev_map,
                                (col_bev - car_width_bev // 2, row_bev - car_length_bev // 2),
                                (col_bev + car_width_bev // 2, row_bev + car_length_bev // 2),
                                color, -1)
                    cv2.putText(bev_map, identifier, (col_bev - car_width_bev // 2, row_bev - car_length_bev // 2 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            elif 'person' in obj_class.lower():
                ped_width_bev = int(1.0 / bev_map_res_m_per_pixel)
                ped_length_bev = int(1.0 / bev_map_res_m_per_pixel)
                color = (255, 0, 0) # Blue (BGR)

                if (0 <= (col_bev - ped_width_bev) and (col_bev + ped_width_bev) < bev_map_width_pixels
                    and 0 <= (row_bev - ped_length_bev) and (row_bev + ped_length_bev) < bev_map_height_pixels):
                    cv2.rectangle(bev_map,
                                    (col_bev - ped_width_bev // 2, row_bev - ped_length_bev // 2),
                                    (col_bev + ped_width_bev // 2, row_bev + ped_length_bev // 2),
                                    color, -1)
                    cv2.putText(bev_map, identifier, (col_bev - ped_width_bev // 2, row_bev - ped_length_bev // 2 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # --- Draw Ego Vehicle ---
    ego_x_m = 0.0 # Ego vehicle is at (0,0) in its own frame
    ego_y_m = 0.0

    # Convert ego (0,0) to BEV map pixel coordinates
    ego_col_bev = int((ego_x_m - x_min_m) / bev_map_res_m_per_pixel)
    ego_row_bev = int(bev_map_height_pixels - 1 - ((ego_y_m - y_min_m) / bev_map_res_m_per_pixel))
    
    # Ego vehicle dimensions (approximate typical car size)
    ego_width_pixels = int(1.5 / bev_map_res_m_per_pixel)
    ego_length_pixels = int(3.0 / bev_map_res_m_per_pixel)
    
    ego_color = (0, 0, 255) # Red (BGR)
    cv2.rectangle(bev_map,
                  (ego_col_bev - ego_width_pixels // 2, ego_row_bev - ego_length_pixels // 2),
                  (ego_col_bev + ego_width_pixels // 2, ego_row_bev + ego_length_pixels // 2),
                  ego_color, -1)
    cv2.putText(bev_map, 'Ego', (ego_col_bev - ego_width_pixels // 2, ego_row_bev - ego_length_pixels // 2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    marker_color = (255, 255, 255)  # White
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    cv2.putText(bev_map, 'FRONT', (bev_map_width_pixels//2 - 30, 25), font, font_scale, marker_color, thickness)
    cv2.putText(bev_map, 'BACK', (bev_map_width_pixels//2 - 25, bev_map_height_pixels - 10), font, font_scale, marker_color, thickness)
    cv2.putText(bev_map, 'LEFT', (10, bev_map_height_pixels//2), font, font_scale, marker_color, thickness)
    cv2.putText(bev_map, 'RIGHT', (bev_map_width_pixels - 60, bev_map_height_pixels//2), font, font_scale, marker_color, thickness)

    return bev_map


def _is_better_camera_view(obj1, obj2):
    """
    Determine if obj1 has a better camera view than obj2 based on object position and camera type.
    """
    x1, y1 = obj1['x_ego'], obj1['y_ego']
    x2, y2 = obj2['x_ego'], obj2['y_ego']
    cam1 = obj1['camera_name']
    cam2 = obj2['camera_name']
    
    def get_camera_score(x, y, camera_name):
        score = 0
        # Front cameras are best for forward objects (y > 0)
        if 'FRONT' in camera_name and y > 0:
            score += 3
        # Back cameras are best for rear objects (y < 0)
        elif 'BACK' in camera_name and y < 0:
            score += 3
        # Left cameras are best for left objects (x < 0)
        if 'LEFT' in camera_name and x < 0:
            score += 2
        # Right cameras are best for right objects (x > 0)
        elif 'RIGHT' in camera_name and x > 0:
            score += 2
        # Center cameras (FRONT, BACK) are good for center objects
        if camera_name in ['CAM_FRONT', 'CAM_BACK'] and abs(x) < 5:
            score += 1
        return score
    
    score1 = get_camera_score(x1, y1, cam1)
    score2 = get_camera_score(x2, y2, cam2)
    
    return score1 > score2


def generate_bevs(data):
    bev_dir.mkdir(parents=True, exist_ok=True)

    for scene_id, scene_obj in tqdm(data.items(), desc="Generating BEVs"):
        for key_frame_id, key_frame in scene_obj["key_frames"].items():
            image_paths = key_frame["image_paths"]
            image_name = f"{scene_id}_{key_frame_id}__BEV.jpg"
            bev_path = bev_dir / image_name
            image_paths["BEV"] = "../nuscenes/samples/BEV/" + image_name

            if not bev_path.exists():
                image_paths = {
                    key: os.path.join(drivelm_dir, path)
                    for key, path in image_paths.items()
                }
                kois = key_frame["key_object_infos"]
                calibration = key_frame["camera_calibration"]
                bev_img = generate_bev_from_detections(
                    kois=kois,
                    calibration=calibration,
                )
                cv2.imwrite(bev_path, bev_img)
                logger.debug(f"Saved bev image: {bev_img}")
    return data
