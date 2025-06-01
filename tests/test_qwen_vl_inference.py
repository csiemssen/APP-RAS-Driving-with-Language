import unittest
import pytest
import numpy as np
from src.data.message_formats import QwenMessageFormat
from src.data.basic_dataset import DriveLMImageDataset, simple_dict_collate
from src.models.qwen_vl_inference import QwenVLInferenceEngine
from torch.utils.data import Subset, DataLoader

@pytest.mark.inference
class TestQwenVLInference(unittest.TestCase):
  def test_qwen_model_load(self):
    engine = QwenVLInferenceEngine("Qwen/Qwen2.5-VL-3B-Instruct")
    engine.load_model()
    self.assertIsNotNone(engine.model, "Model should be loaded successfully.")

  def test_qwen_model_predict(self):
    engine = QwenVLInferenceEngine("Qwen/Qwen2.5-VL-3B-Instruct")
    engine.load_model()

    dataset = DriveLMImageDataset(QwenMessageFormat(), "val")
    test_set = Subset(dataset, np.arange(1))
    dataloader = DataLoader(test_set, batch_size=1, collate_fn=simple_dict_collate)

    results = []
    for batch in dataloader:
      messages, questions, labels, q_ids, qa_types = batch
      predictions = engine.predict_batch(messages)
      results.extend(predictions)
      self.assertEqual(len(predictions), len(messages), "Predictions should match the number of messages.")

    self.assertGreater(len(results), 0, "There should be some predictions made.")
