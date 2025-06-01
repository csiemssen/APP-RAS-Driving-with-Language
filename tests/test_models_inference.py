import unittest
import pytest
import numpy as np
from src.data.message_formats import QwenMessageFormat, InternVLMessageFormat
from src.data.basic_dataset import DriveLMImageDataset, simple_dict_collate
from src.models.intern_vl_inference import InternVLInferenceEngine
from torch.utils.data import Subset, DataLoader

@pytest.mark.inference
class TestModelInference(unittest.TestCase):
  def test_internvl_model_load(self):
    engine = InternVLInferenceEngine("OpenGVLab/InternVL3-2B")
    engine.load_model()
    self.assertIsNotNone(engine.model, "Model should be loaded successfully.")
    self.assertIsNotNone(engine.tokenizer, "Tokenizer should be loaded successfully.")

  def test_internvl_model_predict(self):
    engine = InternVLInferenceEngine("OpenGVLab/InternVL3-2B")
    engine.load_model()

    dataset = DriveLMImageDataset(InternVLMessageFormat(), "val")
    test_set = Subset(dataset, np.arange(4))
    dataloader = DataLoader(test_set, batch_size=2, collate_fn=simple_dict_collate)

    results = []
    for batch in dataloader:
      messages, questions, labels, q_ids, qa_types = batch
      predictions = engine.predict_batch(messages)
      results.extend(predictions)
      self.assertEqual(len(predictions), len(messages), "Predictions should match the number of messages.")

    self.assertGreater(len(results), 0, "There should be some predictions made.")
