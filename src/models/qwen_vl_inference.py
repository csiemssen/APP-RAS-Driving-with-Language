from typing import Dict, List, Optional

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.data.message_formats import (
    QwenMessageFormat,
    QwenTrainingMessageFormat,
)
from src.models.base_inference import BaseInferenceEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QwenVLInferenceEngine(BaseInferenceEngine):
    def __init__(
        self,
        resize_factor: float,
        model_path: Optional[str] = None,
        use_4bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        revision: Optional[str] = None,
        device: Optional[str] = None,
        training: bool = False,
    ):
        super().__init__(
            model_path=model_path,
            use_4bit=use_4bit,
            revision=revision,
            device=device,
        )
        self.resize_factor = resize_factor
        self.processor_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.model_path = self.processor_path if model_path is None else model_path
        self.model = None
        self.torch_dtype = torch_dtype if torch_dtype is not None else self.torch_dtype
        self.tokenizer = None
        self.message_formatter = QwenMessageFormat()
        self.training_message_formatter = QwenTrainingMessageFormat()
        self.training = training

    def load_model(self, flash_attn: bool = True) -> None:
        attn_implementation = "flash_attention_2" if flash_attn else None

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            revision=self.revision,
            torch_dtype=self.torch_dtype,
            attn_implementation=attn_implementation,
            quantization_config=self.quantization_config,
            device_map="auto",
        )

        if not self.training:
            self.model = self.model.eval()

        h = 900 * 2 * self.resize_factor
        w = 1600 * 3 * self.resize_factor
        patch_size = 28
        num_img_tokens = (h // patch_size) * (w // patch_size)
        num_img_pixel = num_img_tokens * patch_size * patch_size

        self.processor = AutoProcessor.from_pretrained(
            self.processor_path,
            revision=self.revision,
            min_pixels=num_img_pixel
            - (num_img_pixel * 0.1),  # Allow for some leeway to be sure
            max_pixels=num_img_pixel + (num_img_pixel * 0.1),
        )

        logger.info(f"{self.model_path} loaded and ready.")

    def predict_batch(self, messages: List[List[Dict]]):
        texts = [
            self.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            for message in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            padding_side="left",
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
