import json
import os
from typing import Optional

import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.constants import data_dir
from src.data.basic_dataset import DriveLMImageDataset, simple_dict_collate
from src.utils.logger import get_logger
from src.utils.utils import sanitize_model_name

logger = get_logger(__name__)


def evaluate_model(
    engine,
    dataset_split: str = "val",
    batch_size: int = 2,
    test_set_size: Optional[int] = None,
    use_grid: bool = False,
):
    dataset = DriveLMImageDataset(
        engine.message_formatter, dataset_split, use_grid=use_grid
    )
    if test_set_size is not None:
        num_samples = min(test_set_size, len(dataset))
        dataset = Subset(dataset, np.arange(num_samples))
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=simple_dict_collate
    )

    engine.load_model()

    results = []

    for batch in tqdm(dataloader, desc="Evaluating model", unit="batch"):
        messages, questions, labels, q_ids, qa_types = batch
        batch_results = engine.predict_batch(messages)

        for i, result in enumerate(batch_results):
            results.append(
                {
                    "id": q_ids[i],
                    "question": questions[i],
                    "answer": result,
                }
            )

    os.makedirs(os.path.join(data_dir, "output"), exist_ok=True)
    with open(
        os.path.join(
            data_dir,
            "output",
            f"{sanitize_model_name(engine.model_path)}_output.json",
        ),
        "w",
    ) as f:
        json.dump(results, f, indent=2)

    with open(
        os.path.join(
            data_dir,
            "output",
            f"{sanitize_model_name(engine.model_path)}_submission.json",
        ),
        "w",
    ) as f:
        json.dump(
            {
                "method": "",
                "team": "appras-tub",
                "authors": [
                    "Veit Laule",
                    "Caspar Siemssen",
                ],
                "email": "v.laule@campus.tu-berlin.de",
                "institution": "Technische Universität Berlin",
                "country": "Germany",
                "results": results,
            },
            f,
            indent=2,
        )
