import json
import os
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import data_dir
from src.data.basic_dataset import DriveLMImageDataset, simple_dict_collate
from src.models.base_inference import BaseInferenceEngine
from src.reasoning.reasoning_engine import ReasoningEngine
from src.utils.logger import get_logger
from src.utils.utils import (
    create_subset,
    normalise_key_objects_in_text,
    sanitize_model_name,
)

logger = get_logger(__name__)


def evaluate_model(
    engine: BaseInferenceEngine,
    batch_size: str,
    dataset_split: str = "val",
    test_set_size: Optional[str] = None,
    use_grid: bool = False,
    use_system_prompt: bool = False,
    system_prompt_config_path: Optional[str] = None,
    use_reasoning: bool = False,
    add_kois: bool = False,
    approach_name: Optional[str] = None,
    exclude_question_tags: List[int] = [],
    exclude_question_types: List[str] = [],
    resize_factor: float = 1.0,
):
    dataset = DriveLMImageDataset(
        message_format=engine.message_formatter,
        split=dataset_split,
        add_kois=add_kois,
        use_grid=use_grid,
        use_reasoning=use_reasoning,
        use_system_prompt=use_system_prompt,
        system_prompt_config_path=system_prompt_config_path,
        exclude_question_tags=exclude_question_tags,
        exclude_question_types=exclude_question_types,
        resize_factor=resize_factor,
    )
    if test_set_size is not None:
        dataset = create_subset(
            ds=dataset,
            sample_size=int(test_set_size),
            by_tag=True,
            equal_distribution=True,
        )
    dataloader = DataLoader(
        dataset, batch_size=int(batch_size), collate_fn=simple_dict_collate
    )

    if use_reasoning:
        reasoning_engine = ReasoningEngine(engine)

    engine.load_model()

    results = []

    for batch_idx, batch in enumerate(
        tqdm(dataloader, desc="Evaluating model", unit="batch")
    ):
        if use_reasoning:
            batch = reasoning_engine.process_batch(batch)

        formatted_messages = [[item.formatted_message] for item in batch]

        batch_results = engine.predict_batch(formatted_messages)

        for i, result in enumerate(batch_results):
            results.append(
                {
                    "id": batch[i].qa_id,
                    "question": normalise_key_objects_in_text(
                        batch[i].question,
                        resize_factor=1 / resize_factor,
                        use_grid=use_grid,
                    ),
                    "model_input": batch[i].formatted_message,
                    "answer": normalise_key_objects_in_text(
                        text=result,
                        resize_factor=1 / resize_factor,
                        use_grid=use_grid,
                    ),
                }
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    model_dir = sanitize_model_name(engine.model_path)
    output_dir = os.path.join(data_dir, "output", model_dir)
    os.makedirs(output_dir, exist_ok=True)

    if approach_name is None or approach_name == "":
        output_file = os.path.join(output_dir, "output.json")
        submission_file = os.path.join(output_dir, "submission.json")
    else:
        output_file = os.path.join(output_dir, f"{approach_name}_output.json")
        submission_file = os.path.join(output_dir, f"{approach_name}_submission.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    with open(submission_file, "w") as f:
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
