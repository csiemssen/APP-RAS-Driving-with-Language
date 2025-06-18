import json

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_relative_position(cam_name: str) -> str:
    mapping = {
        "CAM_FRONT": "front",
        "CAM_FRONT_LEFT": "front left",
        "CAM_FRONT_RIGHT": "front right",
        "CAM_BACK": "back",
        "CAM_BACK_LEFT": "back left",
        "CAM_BACK_RIGHT": "back right",
    }
    return mapping.get(cam_name, "unknown")


def generate_descriptor_qa(obj_id, obj_info):
    desc = obj_info["Visual_description"]
    status = obj_info["Status"] or "unknown"
    cam = obj_id.split(",")[1]
    coords = obj_id.split(",")[2:]
    pos = get_relative_position(cam)
    coord_str = f"({coords[0]},{coords[1]})"
    question = (
        f"The width and height of the image are 1600 and 900 respectively. "
        f"{obj_id} represents the key object that the center coordinates of the bounding box "
        f"in the {cam} image are {coord_str}. "
        f"What is the object {obj_id}? What is the state of it?"
    )

    answer = (
        f"{obj_id} is a {desc.lower()} to the {pos} of the ego vehicle. "
        f"It is {status.lower()}."
    )

    return {"Q": question, "A": answer}


def augment_frame_with_qas(frame):
    koi = frame.get("key_object_infos", {})
    qa_aug = [
        generate_descriptor_qa(obj_id, obj_info) for obj_id, obj_info in koi.items()
    ]
    frame["QA_augmented"] = qa_aug
    return frame


def generate_descriptor_qas(input_dir, output_dir):
    with open(input_dir, "r") as f:
        data = json.load(f)

    for scene_id, scene_obj in data.items():
        for key_frame_id, frame in scene_obj["key_frames"].items():
            scene_obj["key_frames"][key_frame_id] = augment_frame_with_qas(frame)

    with open(output_dir, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Descriptor QAs generated and saved to {output_dir}")
