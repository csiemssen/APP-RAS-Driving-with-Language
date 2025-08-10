import unittest

import pytest

import src.data.message_formats as message_formats
from src.models.anthropic_inference import AnthropicInferenceEngine
from src.models.gemini_inference import GeminiInferenceEngine
from src.models.openai_inference import OpenAIInferenceEngine


@pytest.mark.inference
class TestRemoteInferenceEngine(unittest.TestCase):
    @unittest.skip("Skipping OpenAI inference test")
    def test_openai_predict_batch(self):
        engine = OpenAIInferenceEngine()
        messages = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ]
        ]
        results = engine.predict_batch(messages)

        self.assertTrue(len(results) > 0, "Results should not be empty")

    def test_gemini_predict_batch(self):
        engine = GeminiInferenceEngine(model="gemini-2.0-flash")
        message = message_formats.GeminiMessageFormat().format(
            question="What is the capital of France?",
            image_path=None,
            system_prompt="You are a helpful assistant.",
        )

        results = engine.predict_batch([message])

        self.assertTrue(len(results) > 0, "Results should not be empty")

    def test_anthropic_predict_batch(self):
        engine = AnthropicInferenceEngine(model="claude-3-5-haiku-20241022")
        messages = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ]
        ]
        results = engine.predict_batch(messages)

        self.assertTrue(len(results) > 0, "Results should not be empty")


if __name__ == "__main__":
    unittest.main()
