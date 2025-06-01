import torch
import numpy as np
import platform
from typing import List, Dict
from src.utils.utils import is_mps, is_cuda
from src.data.message_formats import QwenMessageFormat
from src.models.base_inference import BaseInferenceEngine
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.utils.logger import get_logger

BitsAndBytesConfig = None
if platform.system() != "Darwin" and is_cuda():
    from transformers import BitsAndBytesConfig

logger = get_logger(__name__)

class QwenVLInferenceEngine(BaseInferenceEngine):
  def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: torch.device = None):
    super().__init__(model_path, device)
    self.model = None
    self.tokenizer = None
    self.message_formatter = QwenMessageFormat()

  def load_model(self) -> None:
    if is_cuda():
        torch.cuda.empty_cache()

    torch_dtype = torch.float32 if is_mps() else torch.float16
    attn_implementation = "eager" if is_mps() else "flash_attention_2"

    nf4_config = None
    if BitsAndBytesConfig is not None:
      nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
      )

    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        self.model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        quantization_config=nf4_config,
    ).to(self.device).eval()

    self.processor = AutoProcessor.from_pretrained(self.model_path)

    logger.info(f"{self.model_path} loaded and ready.")

  def predict_batch(self, messages: List[List[Dict]]):

    texts = [
        self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        ) for message in messages
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

    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    generated_ids = self.model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = self.processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text
