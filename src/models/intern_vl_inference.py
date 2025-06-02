import torch
from transformers import AutoTokenizer, AutoModel
from src.data.message_formats import InternVLMessageFormat
from src.utils.utils import flatten, is_mps, is_cuda
from src.utils.logger import get_logger
from src.utils.intern_vl_image_utils import load_image, get_num_patches
from src.models.base_inference import BaseInferenceEngine
from typing import List, Dict, Optional

logger = get_logger(__name__)

# ToDo: Implement way easier LMDeploy Pipeline inference for InternVL model as described in documentation (linux only)

class InternVLInferenceEngine(BaseInferenceEngine):
  def __init__(self, model_path: str = "OpenGVLab/InternVL3-2B", device: Optional[str] = None, torch_dtype: Optional[torch.dtype] = None):
    super().__init__(model_path, device)
    self.model = None
    self.tokenizer = None
    self.message_formatter = InternVLMessageFormat()


  def load_model(self) -> None:

    # flash_attention_2 is not supported, flash_attention configured by default, if not available, it will use eager
    self.model = AutoModel.from_pretrained(
      self.model_path,
      trust_remote_code=True,
      low_cpu_mem_usage=True,
      torch_dtype=self.torch_dtype,
      quantization_config=self.quantization_config,
      device_map=self.device
    ).eval()

    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    self.generation_config = dict(max_new_tokens=128, do_sample=False)

    logger.info(f"{self.model_path} loaded and ready.")

  def predict_batch(self, messages: List[List[Dict]]):

    flat_messages = flatten(messages)

    texts = [msg["text"] for msg in flat_messages]

    pixel_values_list = []
    num_patches_list = []

    for msg in flat_messages:
      pixel_tensor = load_image(msg["image_path"]).to(self.device).to(self.torch_dtype)
      num_patches_list.append(get_num_patches(pixel_tensor))
      pixel_values_list.append(pixel_tensor)

    pixel_values = torch.cat(pixel_values_list, dim=0)

    with torch.no_grad():
      responses = self.model.batch_chat(
        tokenizer=self.tokenizer,
        pixel_values=pixel_values,
        num_patches_list=num_patches_list,
        questions=texts,
        generation_config=self.generation_config
      )

    return responses
