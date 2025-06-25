import unittest

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from src.data.basic_dataset import DriveLMImageDataset, simple_dict_collate
from src.data.message_formats import InternVLMessageFormat
from src.models.intern_vl_inference import InternVLInferenceEngine
from src.utils.logger import get_logger
from src.utils.utils import is_cuda

logger = get_logger(__name__)


@pytest.mark.inference
class TestInternVLInference(unittest.TestCase):
    def test_internvl_model_load(self):
        if is_cuda():
            engine = InternVLInferenceEngine(
                "OpenGVLab/InternVL3-2B",
                use_4bit=True,
                torch_dtype=torch.bfloat16,
            )
        else:
            engine = InternVLInferenceEngine(
                "OpenGVLab/InternVL3-2B",
            )

        engine.load_model()
        self.assertIsNotNone(engine.model, "Model should be loaded successfully.")
        self.assertIsNotNone(
            engine.tokenizer, "Tokenizer should be loaded successfully."
        )

    def test_internvl_model_predict(self):
        if is_cuda():
            engine = InternVLInferenceEngine(
                "OpenGVLab/InternVL3-2B",
                use_4bit=True,
                torch_dtype=torch.bfloat16,
            )
        else:
            engine = InternVLInferenceEngine(
                "OpenGVLab/InternVL3-2B",
            )

        engine.load_model()

        dataset = DriveLMImageDataset(
            InternVLMessageFormat(), split="val", use_grid=True
        )
        test_set = Subset(dataset, np.arange(1))
        dataloader = DataLoader(test_set, batch_size=1, collate_fn=simple_dict_collate)

        results = []
        for batch in dataloader:
            query_items = batch
            formatted_messages = [[item.formatted_message for item in query_items]]

            predictions = engine.predict_batch(formatted_messages)
            results.extend(predictions)
            self.assertEqual(
                len(predictions),
                len(query_items),
                "Predictions should match the number of query items in the batch.",
            )

        self.assertGreater(len(results), 0, "There should be some predictions made.")
