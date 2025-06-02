import unittest
import pytest
import numpy as np
import torch
from src.data.message_formats import InternVLMessageFormat
from src.data.basic_dataset import DriveLMImageDataset, simple_dict_collate
from src.models.intern_vl_inference import InternVLInferenceEngine
from torch.utils.data import Subset, DataLoader
from src.utils.utils import is_cuda

@pytest.mark.inference
class TestInternVLInference(unittest.TestCase):
  def test_internvl_model_load(self):
    engine = InternVLInferenceEngine("OpenGVLab/InternVL3-2B")

    if is_cuda():
      engine.configure_quantization(use_4bit=True)

    engine.load_model()
    self.assertIsNotNone(engine.model, "Model should be loaded successfully.")
    self.assertIsNotNone(engine.tokenizer, "Tokenizer should be loaded successfully.")

  def test_internvl_model_predict(self):
    engine = InternVLInferenceEngine("OpenGVLab/InternVL3-2B")

    if is_cuda():
      engine.configure_quantization(use_4bit=True, torch_dtype=torch.bfloat16)

    engine.load_model()

    dataset = DriveLMImageDataset(InternVLMessageFormat(), "val")
    test_set = Subset(dataset, np.arange(1))
    dataloader = DataLoader(test_set, batch_size=1, collate_fn=simple_dict_collate)

    results = []
    for batch in dataloader:
      messages, questions, labels, q_ids, qa_types = batch
      predictions = engine.predict_batch(messages)
      results.extend(predictions)
      self.assertEqual(len(predictions), len(messages), "Predictions should match the number of messages.")

    self.assertGreater(len(results), 0, "There should be some predictions made.")
