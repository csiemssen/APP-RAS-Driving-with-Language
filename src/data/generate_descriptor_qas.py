from typing import List

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


def generate_descriptor_core_question(obj_ids: List[str]) -> str:
    if not obj_ids:
        raise ValueError("Object IDs list cannot be empty.")

    key_objects = ", ".join([f"{obj_id}" for obj_id in obj_ids])
    if len(obj_ids) == 1:
        return f"What is the object {key_objects}? What is its state?"
    return f"What are the objects {key_objects}? What are their states?"


def generate_descriptor_question(obj_id: str, obj_info: dict) -> str:
    obj_id_parts = obj_id.split(",")
    cam = obj_id_parts[1]
    coords = obj_id_parts[2:]
    coords = [coord.strip(">").strip() for coord in coords]
    coord_str = f"({coords[0]},{coords[1]})"
    core_question = generate_descriptor_core_question([obj_id])
    question = (
        f"The key object, identified as {obj_id}, has its bounding box center coordinates at {coord_str} in the {cam} image. "
        f"{core_question} "
    )
    return question


def generate_descriptor_answer(obj_id: str, obj_info: dict) -> str:
    desc = obj_info["Visual_description"]
    status = obj_info["Status"] or "unknown"
    obj_id_parts = obj_id.split(",")
    cam = obj_id_parts[1]
    pos = get_relative_position(cam)
    answer = f"{obj_id} is a {desc} to the {pos} of the ego vehicle. It is {status}."
    return answer


def generate_descriptor_qa(obj_id: str, obj_info: dict) -> dict:
    question = generate_descriptor_question(obj_id, obj_info)
    answer = generate_descriptor_answer(obj_id, obj_info)
    return {"Q": question, "A": answer}


def augment_frame_with_qas(frame):
    koi = frame.get("key_object_infos", {})
    qa_aug = [
        generate_descriptor_qa(obj_id, obj_info) for obj_id, obj_info in koi.items()
    ]
    frame["QA"]["augmented"] = qa_aug
    return frame


def generate_descriptor_qas(data):
    for scene_id, scene_obj in data.items():
        for key_frame_id, frame in scene_obj["key_frames"].items():
            scene_obj["key_frames"][key_frame_id] = augment_frame_with_qas(frame)

    logger.info("Descriptor QAs generated")
    return data
