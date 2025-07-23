import os
from typing import Dict, List, Optional

import anthropic
import dotenv

from src.data.message_formats import AnthropicMessageFormat
from src.models.remote_inference import RemoteInferenceEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnthropicInferenceEngine(RemoteInferenceEngine):
    def __init__(self, model: Optional[str] = "claude-3-5-haiku-20241022"):
        super().__init__(model_path=model)
        dotenv.load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.message_formatter = AnthropicMessageFormat()

    def predict_batch(self, messages: List[List[Dict]]):
        responses = []
        total_tokens = 0
        for msg in messages:
            system_prompt = None
            filtered_msg = []
            for m in msg:
                if m["role"] == "system":
                    if system_prompt is None:
                        system_prompt = m["content"]
                    else:
                        system_prompt += "\n" + m["content"]
                else:
                    filtered_msg.append(m)

            response = self.client.messages.create(
                model=self.model,
                messages=filtered_msg,
                temperature=0.6,
                max_tokens=128,
            )
            responses.append(response.content[0].text.strip())
            total_tokens += response.usage.input_tokens + response.usage.output_tokens

        logger.debug(
            f"Generated {len(responses)} responses for batch of size {len(messages)}"
        )
        logger.debug(f"Total tokens used: {total_tokens}")

        return responses
