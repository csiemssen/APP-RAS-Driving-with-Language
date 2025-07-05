import os
from typing import Any, List

from torch.utils.data import Dataset

from src.constants import drivelm_dir
from src.data.generate_reasoning_context import generate_reasoning_context
from src.data.load_dataset import load_dataset
from src.data.message_formats import MessageFormat
from src.data.system_prompts import get_system_prompt
from src.data.query_item import QueryItem
from src.utils.logger import get_logger
from src.utils.utils import remove_nones

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


# NOTE: This DS does not consider any direct dependencies between the questions
class DriveLMImageDataset(Dataset):
    def __init__(
        self,
        message_format: MessageFormat,
        resize_factor: float = 0.5,
        split="train",
        add_augmented=False,
        use_grid=False,
        use_reasoning=False,
        use_system_prompt=False,
    ):
        self.message_format = message_format
        self.split = split
        self.use_reasoning = use_reasoning
        self.use_grid = use_grid
        self.use_system_prompt = use_system_prompt

        data = load_dataset(
            split,
            add_augmented=add_augmented,
            use_grid=use_grid,
            resize_factor=resize_factor,
        )

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
        key_object_info = qa["key_object_info"]
        image_path = qa["image_path"]
        system_prompt = (
            get_system_prompt(
                qa["qa_type"],
                use_grid=self.use_grid,
                use_reasoning=self.use_reasoning,
            ) if self.use_system_prompt else None
        )

        query_item = QueryItem(
            question=question,
            image_path=image_path,
            qa_id=qa["id"],
            qa_type=qa["qa_type"],
            key_object_info=key_object_info,
            system_prompt=system_prompt,
            ground_truth_answer=answer,
        )

        if self.use_reasoning and self.split == "train":
            query_item.context_pairs = generate_reasoning_context(query_item)
        query_item.formatted_message = query_item.format_message(self.message_format)

        return query_item
