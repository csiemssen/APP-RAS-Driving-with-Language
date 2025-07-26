import os
from typing import Dict, List, Optional

import dotenv
from openai import OpenAI

from src.data.message_formats import OpenAIMessageFormat
from src.models.remote_inference import RemoteInferenceEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIInferenceEngine(RemoteInferenceEngine):
    def __init__(self, model: Optional[str] = "gpt-4.1"):
        super().__init__(model_path=model)
        dotenv.load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.message_formatter = OpenAIMessageFormat()

    def predict_batch(self, messages: List[List[Dict]]):
        responses = []
        total_tokens = 0
        for msg in messages:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=msg,
                temperature=0.6,
                max_tokens=128,
            )
            responses.append(response.output_text)
            total_tokens += response.usage.total_tokens

        logger.debug(
            f"Generated {len(responses)} responses for batch of size {len(messages)}"
        )
        logger.debug(f"Total tokens used: {total_tokens}")

        return responses
