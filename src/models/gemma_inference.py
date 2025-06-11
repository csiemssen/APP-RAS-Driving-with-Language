from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from src.data.message_formats import GemmaMessageFormat
from src.models.base_inference import BaseInferenceEngine
from src.utils.logger import get_logger
from src.utils.utils import is_mps

logger = get_logger(__name__)


# NOTE: Adapted from https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora
def process_vision_info(message: list[dict]) -> list[Image.Image]:
    image_inputs = []
    # Iterate through each conversation
    for msg in message:
        # Get content (ensure it's a list)
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Check each content element for images
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                # Get the image and convert to RGB
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(Image.open(image).convert("RGB"))
    return image_inputs


class GemmaInferenceEngine(BaseInferenceEngine):
    def __init__(
        self,
        model_path: str = "google/gemma-3-4b-it",
        revision: Optional[str] = None,
        device: Optional[str] = None,
        use_4bit: bool = True,
    ):
        super().__init__(model_path, revision, device, use_4bit)
        self.model = None
        self.tokenizer = None
        self.message_formatter = GemmaMessageFormat()

    def load_model(self) -> None:
        # NOTE: flash-attention-2 leads to errors here, so we use sdpa for now
        attn_implementation = "eager" if is_mps() else "sdpa"

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_path,
            revision=self.revision,
            torch_dtype=self.torch_dtype,
            quantization_config=self.quantization_config,
            attn_implementation=attn_implementation,
            device_map="auto",
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, revision=self.revision
        )

        logger.info(f"{self.model_path} loaded and ready.")

    def predict_batch(self, messages: List[List[Dict]]) -> list[str]:
        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages
        ]
        image_inputs = [process_vision_info(message) for message in messages]

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        logger.info(
            f"Generated {len(output_text)} responses for batch of size {len(messages)}"
        )

        return output_text
