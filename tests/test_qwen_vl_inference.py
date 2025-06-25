import unittest

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from src.data.basic_dataset import DriveLMImageDataset, simple_dict_collate
from src.data.message_formats import QwenMessageFormat
from src.models.qwen_vl_inference import QwenVLInferenceEngine
from src.utils.utils import is_cuda


@pytest.mark.inference
class TestQwenVLInference(unittest.TestCase):
    def test_qwen_model_load(self):
        if is_cuda():
            engine = QwenVLInferenceEngine(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                use_4bit=True,
                torch_dtype=torch.bfloat16,
            )
        else:
            engine = QwenVLInferenceEngine("Qwen/Qwen2.5-VL-3B-Instruct")

        engine.load_model()
        self.assertIsNotNone(engine.model, "Model should be loaded successfully.")

    def test_qwen_model_predict(self):
        if is_cuda():
            engine = QwenVLInferenceEngine(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                use_4bit=True,
                torch_dtype=torch.bfloat16,
            )
        else:
            engine = QwenVLInferenceEngine("Qwen/Qwen2.5-VL-3B-Instruct")

        engine.load_model()

        dataset = DriveLMImageDataset(QwenMessageFormat(), "val")
        test_set = Subset(dataset, np.arange(1))
        dataloader = DataLoader(test_set, batch_size=1, collate_fn=simple_dict_collate)

        results = []
        for batch in dataloader:
            messages = [item.formatted_message for item in batch]
            predictions = engine.predict_batch(messages)
            results.extend(predictions)
            self.assertEqual(
                len(predictions),
                len(batch),
                "Predictions should match the number of query items in the batch.",
            )

        self.assertGreater(len(results), 0, "There should be some predictions made.")
