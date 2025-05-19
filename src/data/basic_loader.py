import os
from json import load

from torch.utils.data import Dataset

from src.constants import drivelm_dir, drivelm_json


# With the current dataset structure and the specific questioning of bounding boxes, it is unclear whether the
# eval will even be transferable to video.


# NOTE: This DS does not consider any dependecies between the questions
class DriveLMImageDataset(Dataset):
    def __init__(self):
        # TODO: Think of a way to optimize this mess
        data = load(open(drivelm_json))
        key_frames = [data[k]["key_frames"][kf] for k in data.keys() for kf in data[k]["key_frames"].keys()]

        qas_perception = [{
            "qa": qa,
            "key_object_info": kf["key_object_infos"],
            "image_path": os.path.join(drivelm_dir, kf["image_paths"]["CAM_FRONT"])
        } for kf in key_frames for qa in kf["QA"]["perception"]]
        qas_prediction = [{
            "qa": qa,
            "key_object_info": kf["key_object_infos"],
            "image_path": os.path.join(drivelm_dir, kf["image_paths"]["CAM_FRONT"])
        } for kf in key_frames for qa in kf["QA"]["prediction"]]
        qas_planning = [{
            "qa": qa,
            "key_object_info": kf["key_object_infos"],
            "image_path": os.path.join(drivelm_dir, kf["image_paths"]["CAM_FRONT"])
        } for kf in key_frames for qa in kf["QA"]["planning"]]

        self.qas = qas_perception + qas_prediction + qas_planning

    def __len__(self):
        return len(self.qas)

    def __getitem__(self, idx):
        print("Called get_item")
        qa = self.qas[idx]
        question = qa["qa"]["Q"]
        answer = qa["qa"]["A"]
        key_object_info = qa["key_object_info"]
        image_path = qa["image_path"]
        return question, answer, key_object_info, image_path
