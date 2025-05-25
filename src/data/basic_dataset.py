from collections.abc import Callable
import os
from json import load
from typing import Any

from torch.utils.data import Dataset
from transformers import AutoProcessor

from src.constants import drivelm_dir, drivelm_json
from src.data.message_formats import MessageFormat
from src.utils.utils import remove_nones


# With the current dataset structure and the specific questioning of bounding boxes, it is unclear whether the
# eval will even be transferable to video.

def simple_dict_collate(batch: Any) -> Any:
    return [[b] for b in batch]

def prune_key_object_info(koi: dict[str, Any]):
    # TODO: Change this to something else if we really need all keys and values from koi
    return {km:{k:v} for km, _ in koi.items() for k, v in koi[km].items() if k != "2d_bbox"}

# NOTE: This DS does not consider any dependecies between the questions
class DriveLMImageDataset(Dataset):
    def __init__(self, message_format: MessageFormat):
        self.message_format = message_format

        data = load(open(drivelm_json))
        key_frames = [data[k]["key_frames"][kf] for k in data.keys() for kf in data[k]["key_frames"].keys()]

        # NOTE: We have to pass None for the koi, as we should not assume that the model already konws about the kois
        qas_perception = [{
            "qa": remove_nones(qa),
            "key_object_info": None,
            "image_path": os.path.join(drivelm_dir, kf["image_paths"]["CAM_FRONT"])
        } for kf in key_frames for qa in kf["QA"]["perception"]]
        qas_prediction = [{
            "qa": remove_nones(qa),
            "key_object_info": prune_key_object_info(kf["key_object_infos"]),
            "image_path": os.path.join(drivelm_dir, kf["image_paths"]["CAM_FRONT"])
        } for kf in key_frames for qa in kf["QA"]["prediction"]]
        qas_planning = [{
            "qa": remove_nones(qa),
            "key_object_info": prune_key_object_info(kf["key_object_infos"]),
            "image_path": os.path.join(drivelm_dir, kf["image_paths"]["CAM_FRONT"])
        } for kf in key_frames for qa in kf["QA"]["planning"]]

        # NOTE: This is a workaround for when we dont have the full dataset available on disk
        #       If the full ds should be used double check that we have ever kf available.
        qas_perception = [d for d in qas_perception if os.path.isfile(d["image_path"])]
        qas_prediction = [d for d in qas_prediction if os.path.isfile(d["image_path"])]
        qas_planning = [d for d in qas_planning if os.path.isfile(d["image_path"])]

        self.qas = qas_perception + qas_prediction + qas_planning

    def __len__(self):
        return len(self.qas)

    def __getitem__(self, idx):
        qa = self.qas[idx]
        question = qa["qa"]["Q"]
        print(question)
        # TODO: Think about how to include the answer here as well
        #answer = qa["qa"]["A"]
        key_object_info = qa["key_object_info"]
        image_path = qa["image_path"]
        return self.message_format.format(question, key_object_info, image_path)
