import json
import os
import unittest

import pytest
import torch

from src.constants import data_dir
from src.eval.eval_models import evaluate_model
from src.models.anthropic_inference import AnthropicInferenceEngine
from src.models.gemini_inference import GeminiInferenceEngine
from src.models.intern_vl_inference import InternVLInferenceEngine
from src.models.qwen_vl_inference import QwenVLInferenceEngine
from src.utils.utils import is_cuda, sanitize_model_name


def check_eval_files(output_dir, output_file_name, submission_file_name):
    output_file = os.path.join(output_dir, output_file_name)
    submission_file = os.path.join(output_dir, submission_file_name)

    assert os.path.exists(output_file), "Output file should be created."
    assert os.path.exists(submission_file), "Submission file should be created."

    with open(output_file, "r") as f:
        results = json.load(f)
        assert len(results) > 0, "Results should not be empty."
        for result in results:
            assert "id" in result
            assert "question" in result
            assert "answer" in result


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
            engine=engine,
            dataset_split="val",
            batch_size=1,
            test_set_size=1,
            use_reasoning=True,
            use_grid=True,
            approach_name="test_intern_vl_eval",
        )

        model_dir = sanitize_model_name(engine.model_path)
        output_dir = os.path.join(data_dir, "output", model_dir)
        check_eval_files(
            output_dir,
            "test_intern_vl_eval_output.json",
            "test_intern_vl_eval_submission.json",
        )

    @unittest.skip("Skipping Qwen evaluation test")
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
            engine=engine,
            dataset_split="val",
            batch_size=1,
            test_set_size=1,
            approach_name="test_qwen_eval",
        )

        model_dir = sanitize_model_name(engine.model_path)
        output_dir = os.path.join(data_dir, "output", model_dir)
        check_eval_files(
            output_dir,
            "test_qwen_eval_output.json",
            "test_qwen_eval_submission.json",
        )

    def test_gemini_eval(self):
        engine = GeminiInferenceEngine(model="gemini-2.0-flash")
        evaluate_model(
            engine=engine,
            dataset_split="val",
            batch_size=1,
            test_set_size=1,
            use_grid=True,
            use_system_prompt=True,
            approach_name="test_gemini_eval",
        )

        model_dir = engine.model
        output_dir = os.path.join(data_dir, "output", model_dir)
        check_eval_files(
            output_dir,
            "test_gemini_eval_output.json",
            "test_gemini_eval_submission.json",
        )

    def test_anthropic_eval(self):
        engine = AnthropicInferenceEngine(model="claude-3-5-haiku-20241022")
        evaluate_model(
            engine=engine,
            dataset_split="val",
            batch_size=1,
            test_set_size=1,
            use_grid=True,
            use_system_prompt=True,
            approach_name="test_anthropic_eval",
        )

        model_dir = sanitize_model_name(engine.model_path)
        output_dir = os.path.join(data_dir, "output", model_dir)
        check_eval_files(
            output_dir,
            "test_anthropic_eval_output.json",
            "test_anthropic_eval_submission.json",
        )
