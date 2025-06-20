import json
import os
import unittest

import pytest
import torch

from src.constants import data_dir
from src.eval.eval_models import evaluate_model
from src.models.intern_vl_inference import InternVLInferenceEngine
from src.models.qwen_vl_inference import QwenVLInferenceEngine
from src.utils.utils import is_cuda, sanitize_model_name


@pytest.mark.eval
class TestModelEvaluation(unittest.TestCase):
    def test_intern_vl_eval(self):
        if is_cuda():
            engine = InternVLInferenceEngine(
                "OpenGVLab/InternVL3-2B",
                use_4bit=True,
                torch_dtype=torch.bfloat16,
                use_grid=True,
            )
        else:
            engine = InternVLInferenceEngine("OpenGVLab/InternVL3-2B")

        evaluate_model(
            engine=engine, dataset_split="val", batch_size=1, test_set_size=1
        )

        output_file = os.path.join(
            data_dir,
            "output",
            f"{sanitize_model_name(engine.model_path)}_output.json",
        )
        submission_file = os.path.join(
            data_dir,
            "output",
            f"{sanitize_model_name(engine.model_path)}_submission.json",
        )
        self.assertTrue(os.path.exists(output_file), "Output file should be created.")
        self.assertTrue(
            os.path.exists(submission_file),
            "Submission file should be created.",
        )

        with open(output_file, "r") as f:
            results = json.load(f)
            self.assertGreater(len(results), 0, "Results should not be empty.")
            for result in results:
                self.assertIn("id", result)
                self.assertIn("question", result)
                self.assertIn("answer", result)

    def test_qwen_eval(self):
        if is_cuda():
            engine = QwenVLInferenceEngine(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                use_4bit=True,
                torch_dtype=torch.bfloat16,
                use_grid=True,
            )
        else:
            engine = QwenVLInferenceEngine("Qwen/Qwen2.5-VL-3B-Instruct")

        evaluate_model(
            engine=engine, dataset_split="val", batch_size=1, test_set_size=1
        )

        output_file = os.path.join(
            data_dir,
            "output",
            f"{sanitize_model_name(engine.model_path)}_output.json",
        )
        submission_file = os.path.join(
            data_dir,
            "output",
            f"{sanitize_model_name(engine.model_path)}_submission.json",
        )
        self.assertTrue(os.path.exists(output_file), "Output file should be created.")
        self.assertTrue(
            os.path.exists(submission_file),
            "Submission file should be created.",
        )

        with open(output_file, "r") as f:
            results = json.load(f)
            self.assertGreater(len(results), 0, "Results should not be empty.")
            for result in results:
                self.assertIn("id", result)
                self.assertIn("question", result)
                self.assertIn("answer", result)
