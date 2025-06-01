import os
import json
import torch
import numpy as np
from typing import Optional
from torch.utils.data import DataLoader, Subset
from src.data.basic_dataset import DriveLMImageDataset, simple_dict_collate
from src.data.message_formats import QwenMessageFormat, InternVLMessageFormat
from src.models.intern_vl_inference import InternVLInferenceEngine
from src.constants import data_dir
from src.utils.utils import sanitize_model_name
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(engine, dataset_split: str = "val", batch_size: int = 2, test_set_size: Optional[int] = None):
    dataset = DriveLMImageDataset(engine.message_formatter, dataset_split)
    if test_set_size is not None:
        num_samples = min(test_set_size, len(dataset))
        dataset = Subset(dataset, np.arange(num_samples))
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=simple_dict_collate)

    engine.load_model()

    results = []

    for batch in dataloader:
        messages, questions, labels, q_ids, qa_types = batch
        batch_results = engine.predict_batch(messages)

        for i, result in enumerate(batch_results):
            results.append({
                "id": q_ids[i],
                "question": questions[i],
                "answer": result,
            })

    os.makedirs(os.path.join(data_dir, "output"), exist_ok=True)
    with open(os.path.join(data_dir, "output", f"{sanitize_model_name(engine.model_path)}_output.json"), "w") as f:
      json.dump(results, f, indent=2)
