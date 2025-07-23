import os
from typing import Any, List

from torch.utils.data import Dataset

from src.constants import GRID_IMG_SIZE, IMAGE_SIZE, drivelm_dir
from src.data.create_image_grid_dataset import (
    create_image_grid_dataset,
    map_camera_point_to_grid_point,
)
from src.data.generate_descriptor_qas import (
    generate_descriptor_qas,
)
from src.data.generate_reasoning_context import generate_reasoning_context
from src.data.load_dataset import load_dataset
from src.data.message_formats import MessageFormat
from src.data.query_item import QueryItem
from src.data.system_prompts import SystemPromptProvider
from src.utils.logger import get_logger
from src.utils.utils import (
    find_key_objects,
    key_object_dict_to_str,
    key_object_str_to_dict,
    remove_nones,
    rescale_point,
)

logger = get_logger(__name__)


# With the current dataset structure and the specific qugestioning of bounding boxes, it is unclear whether the
# eval will even be transferable to video.
def simple_dict_collate(batch: List[QueryItem]) -> List[QueryItem]:
    return batch


def prune_key_object_info(koi: dict[str, Any]):
    # TODO: Change this to something else if we really need all keys and values from koi
    return {
        km: {k: v}
        for km, _ in koi.items()
        for k, v in koi[km].items()
        if k != "2d_bbox"
    }


def normalise_key_object_point(
    point: tuple[float, float], resize_factor: float, use_grid: bool
) -> tuple[float, float]:
    image_size = GRID_IMG_SIZE if use_grid else IMAGE_SIZE
    return rescale_point(point, image_size, resize_factor)


def normalise_key_object_info_value(
    key: str,
    value: dict[str, Any],
    resize_factor: float,
    use_grid: bool,
) -> dict[str, Any]:
    koi_dict = key_object_str_to_dict(key)
    new_value = value.copy()
    if "2d_bbox" in new_value:
        x1, y1, x2, y2 = new_value["2d_bbox"]
        if use_grid:
            x1, y1 = map_camera_point_to_grid_point((x1, y1), koi_dict["camera"])
            x2, y2 = map_camera_point_to_grid_point((x2, y2), koi_dict["camera"])

        x1, y1 = normalise_key_object_point(
            (x1, y1),
            resize_factor,
            use_grid,
        )

        x2, y2 = normalise_key_object_point(
            (x2, y2),
            resize_factor,
            use_grid,
        )

        new_value["2d_bbox"] = (x1, y1, x2, y2)

    return new_value


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

    new_x, new_y = normalise_key_object_point((new_x, new_y), resize_factor, use_grid)

    koi_dict["x"] = new_x
    koi_dict["y"] = new_y

    return key_object_dict_to_str(koi_dict)


def normalise_key_object_descriptors_in_question(
    question: str,
    resize_factor: float,
    use_grid: bool,
) -> str:
    descriptors = find_key_objects(question)
    for desc in descriptors:
        norm_desc = normalise_key_object_descriptor(
            desc,
            resize_factor,
            use_grid,
        )
        question = question.replace(desc, norm_desc)
    return question


# NOTE: This DS does not consider any direct dependencies between the questions
class DriveLMImageDataset(Dataset):
    def __init__(
        self,
        message_format: MessageFormat,
        split="train",
        add_augmented=False,
        use_grid=False,
        use_reasoning=False,
        use_system_prompt=False,
        system_prompt_config_path=None,
        exclude_question_tags: List[int] = [],
        exclude_question_types: List[str] = [],
        resize_factor: float = 1.0,
    ):
        self.message_format = message_format
        self.split = split
        self.use_reasoning = use_reasoning
        self.use_grid = use_grid
        self.system_prompt_provider = (
            SystemPromptProvider(config_path=system_prompt_config_path)
            if use_system_prompt
            else None
        )

        data = load_dataset(split)

        if split == "train":
            for scene_id, scene_data in data.items():
                for key_frame_id, key_frame_data in scene_data["key_frames"].items():
                    key_object_infos = key_frame_data["key_object_infos"]
                    normalised_key_object_infos = {}
                    if key_object_infos:
                        for key, value in key_object_infos.items():
                            new_key = normalise_key_object_descriptor(
                                key,
                                resize_factor,
                                use_grid,
                            )
                            new_value = normalise_key_object_info_value(
                                key,
                                value,
                                resize_factor,
                                use_grid,
                            )
                            normalised_key_object_infos[new_key] = new_value

                    key_frame_data["key_object_infos"] = normalised_key_object_infos

        if split == "train" and add_augmented:
            data = generate_descriptor_qas(data)

        if use_grid:
            data = create_image_grid_dataset(data)

        removed = 0
        qa_list = []
        for scene_id in data.keys():
            scene_obj = data[scene_id]["key_frames"]
            for key_frame_id in scene_obj.keys():
                # NOTE: Only consider FRONT camera images or GRID images for now
                image_paths = scene_obj[key_frame_id]["image_paths"]
                if use_grid:
                    image_path = os.path.join(
                        drivelm_dir,
                        image_paths["GRID"],
                    )
                else:
                    image_path = os.path.join(
                        drivelm_dir,
                        image_paths["CAM_FRONT"],
                    )

                # NOTE: This is a simple workaround if we do not have all files available
                if not os.path.isfile(image_path):
                    removed += 1
                    continue

                key_object_infos = (
                    scene_obj[key_frame_id]["key_object_infos"]
                    if split == "train"
                    else None
                )

                qas = scene_obj[key_frame_id]["QA"]

                qas_perception = qas["perception"]
                qas_prediction = qas["prediction"]
                qas_planning = qas["planning"]
                qas_behavior = qas["behavior"]

                qa_types = (
                    ["perception" for _ in range(len(qas_perception))]
                    + ["prediction" for _ in range(len(qas_prediction))]
                    + ["planning" for _ in range(len(qas_planning))]
                    + ["behavior" for _ in range(len(qas_behavior))]
                )

                qas_augmented = qas.get("augmented", [])
                qa_types += ["augmented" for _ in range(len(qas_augmented))]

                for i, qa in enumerate(
                    qas_perception
                    + qas_prediction
                    + qas_planning
                    + qas_behavior
                    + qas_augmented
                ):
                    qa["Q"] = normalise_key_object_descriptors_in_question(
                        qa["Q"],
                        resize_factor=resize_factor,
                        use_grid=use_grid,
                    )

                    tags = qa.get("tag", [])
                    qa_type = qa_types[i]
                    if (
                        any(tag in exclude_question_tags for tag in tags)
                        or qa_type in exclude_question_types
                    ):
                        continue

                    qa_list.append(
                        {
                            "qa": remove_nones(qa),
                            "qa_type": qa_types[i],
                            "id": scene_id + "_" + key_frame_id + "_" + str(i),
                            "key_object_info": key_object_infos
                            if qa_types[i] != "perception"
                            else None,
                            "image_path": image_path,
                        }
                    )

        logger.info(f"Removed {removed} scenes due to missing image files.")
        logger.info(f"Loaded {len(qa_list)} QAs from the DriveLM dataset.")

        qa_list.sort(key=lambda qa: qa["qa_type"])
        self.qas = qa_list

    def __len__(self):
        return len(self.qas)

    def __getitem__(self, idx):
        qa = self.qas[idx]
        question = qa["qa"]["Q"]
        answer = qa["qa"]["A"]
        tags = qa["qa"].get("tag", [])
        key_object_info = qa["key_object_info"]
        image_path = qa["image_path"]
        system_prompt = (
            self.system_prompt_provider.get_system_prompt(
                question_type=qa["qa_type"],
                question=question,
                use_grid=self.use_grid,
                use_reasoning=self.use_reasoning,
            )
            if self.system_prompt_provider
            else None
        )

        query_item = QueryItem(
            question=question,
            image_path=image_path,
            qa_id=qa["id"],
            qa_type=qa["qa_type"],
            tags=tags,
            key_object_info=key_object_info,
            system_prompt=system_prompt,
            ground_truth_answer=answer,
        )

        if self.use_reasoning and self.split == "train":
            query_item.context_pairs = generate_reasoning_context(query_item)

        query_item.formatted_message = query_item.format_message(self.message_format)

        return query_item
