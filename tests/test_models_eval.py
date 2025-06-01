import os
import json
import unittest
import pytest
from src.eval.eval_models import evaluate_model
from src.models.intern_vl_inference import InternVLInferenceEngine
from src.data.message_formats import QwenMessageFormat, InternVLMessageFormat
from src.constants import data_dir
from src.utils.utils import sanitize_model_name
import torch


@pytest.mark.eval
class TestModelEvaluation(unittest.TestCase):
  def test_intern_vl_eval(self):
    engine = InternVLInferenceEngine("OpenGVLab/InternVL3-2B")
    evaluate_model(
        engine=engine,
        dataset_split="val",
        batch_size=1,
        test_set_size=1
    )

    output_file = os.path.join(data_dir, "output", f"{sanitize_model_name(engine.model_path)}output.json")
    self.assertTrue(os.path.exists(output_file), "Output file should be created.")

    with open(output_file, 'r') as f:
      results = json.load(f)
      self.assertGreater(len(results), 0, "Results should not be empty.")
      for result in results:
        self.assertIn("id", result)
        self.assertIn("question", result)
        self.assertIn("answer", result)
