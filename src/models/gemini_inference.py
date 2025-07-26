from typing import Dict, List, Optional

import dotenv
from google import genai
from google.genai import types

from src.data.message_formats import GeminiMessageFormat
from src.models.remote_inference import RemoteInferenceEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiInferenceEngine(RemoteInferenceEngine):
    def __init__(self, model: Optional[str] = "gemini-2.0-flash"):
        super().__init__(model_path=model)
        dotenv.load_dotenv()
        self.client = genai.Client()
        self.model = model
        self.message_formatter = GeminiMessageFormat()

    def predict_batch(self, messages: List[List[Dict]]):
        responses = []
        total_tokens = 0

        for msg in messages:
            response = self.client.models.generate_content(
                model=self.model,
                contents=msg,
                config=types.GenerateContentConfig(
                    temperature=0.6,
                    max_output_tokens=128,
                ),
            )

            responses.append(response.text.strip())
            total_tokens += response.usage_metadata.total_token_count

        logger.debug(
            f"Generated {len(responses)} responses for batch of size {len(messages)}"
        )
        logger.debug(f"Total tokens used: {total_tokens}")

        return responses
