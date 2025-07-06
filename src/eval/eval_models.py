import json
import os
from typing import Optional

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import data_dir
from src.data.basic_dataset import DriveLMImageDataset, simple_dict_collate
from src.models.base_inference import BaseInferenceEngine
from src.reasoning.reasoning_engine import ReasoningEngine
from src.utils.logger import get_logger
from src.utils.utils import create_subset_for_testing, sanitize_model_name

logger = get_logger(__name__)


def evaluate_model(
    engine: BaseInferenceEngine,
    batch_size: str,
    dataset_split: str = "val",
    test_set_size: Optional[str] = None,
    use_grid: bool = False,
    use_system_prompt: bool = False,
    use_reasoning: bool = False,
):
    dataset = DriveLMImageDataset(
        message_format=engine.message_formatter,
        split=dataset_split,
        use_grid=use_grid,
        use_system_prompt=use_system_prompt,
        use_reasoning=use_reasoning,
    )
    if test_set_size is not None:
        dataset = create_subset_for_testing(dataset, int(test_set_size))
    dataloader = DataLoader(
        dataset, batch_size=int(batch_size), collate_fn=simple_dict_collate
    )

    if use_reasoning:
        reasoning_engine = ReasoningEngine(engine)

    engine.load_model()

    results = []

    for batch in tqdm(dataloader, desc="Evaluating model", unit="batch"):
        if use_reasoning:
            batch = reasoning_engine.process_batch(batch)

        formatted_messages = [[item.formatted_message] for item in batch]

        batch_results = engine.predict_batch(formatted_messages)

        for i, result in enumerate(batch_results):
            results.append(
                {
                    "id": batch[i].qa_id,
                    "question": batch[i].question,
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
