import unittest

import src.data.message_formats as message_formats
from src.models.gemini_inference import GeminiInferenceEngine
from src.models.openai_inference import OpenAIInferenceEngine


class TestRemoteInferenceEngine(unittest.TestCase):
    @unittest.skip("Skipping OpenAI inference test")
    def test_openai_predict_batch(self):
        engine = OpenAIInferenceEngine()
        # Prepare a batch of messages in OpenAI format
        messages = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ]
        ]
        results = engine.predict_batch(messages)
        print(results)

    def test_gemini_predict_batch(self):
        engine = GeminiInferenceEngine(model="gemini-2.0-flash")
        # Prepare a batch of messages in Gemini format
        message = message_formats.GeminiMessageFormat().format(
            question="What is the capital of France?",
            image_path=None,
            system_prompt="You are a helpful assistant.",
        )

        results = engine.predict_batch([message])
        print(results)


if __name__ == "__main__":
    unittest.main()
