import torch
import platform
from typing import List, Dict, Optional
from src.utils.utils import is_mps, is_cuda
from src.data.message_formats import QwenMessageFormat
from src.models.base_inference import BaseInferenceEngine
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.utils.logger import get_logger

logger = get_logger(__name__)

class QwenVLInferenceEngine(BaseInferenceEngine):
  def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: Optional[str] = None, torch_dtype: Optional[torch.dtype] = None):
    super().__init__(model_path, device)
    self.model = None
    self.tokenizer = None
    self.message_formatter = QwenMessageFormat()

  def load_model(self) -> None:

    attn_implementation = "eager" if is_mps() else "flash_attention_2"

    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        self.model_path,
        torch_dtype=self.torch_dtype,
        attn_implementation=attn_implementation,
        quantization_config=self.quantization_config,
        device_map=self.device,
    ).eval()

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
    with torch.no_grad():
      generated_ids = self.model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = self.processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text
