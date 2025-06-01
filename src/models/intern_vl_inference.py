import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from src.data.message_formats import InternVLMessageFormat
from src.utils.utils import flatten, is_mps, is_cuda
from src.utils.logger import get_logger
from src.utils.internVL_image_utils import load_image, get_num_patches
from src.models.base_inference import BaseInferenceEngine
from typing import List, Dict


logger = get_logger(__name__)

# ToDo: Implement way easier LMDeploy Pipeline inference for InternVL model as described in documentation (linux only)

class InternVLInferenceEngine(BaseInferenceEngine):
  def __init__(self, model_path: str = "OpenGVLab/InternVL3-2B", device: torch.device = None):
    super().__init__(model_path, device)
    self.model = None
    self.tokenizer = None
    self.message_formatter = InternVLMessageFormat()

  def load_model(self) -> None:

    torch_dtype = torch.float32 if is_mps() else torch.float16

    self.model = AutoModel.from_pretrained(
      self.model_path,
      torch_dtype=torch_dtype,
      low_cpu_mem_usage=True,
      use_flash_attn=is_cuda(),
      load_in_8bit=is_cuda(),
      trust_remote_code=True,
    ).to(self.device).eval()

    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    self.generation_config = dict(max_new_tokens=128, do_sample=False)

    logger.info(f"{self.model_path} loaded and ready.")

  def predict_batch(self, messages: List[List[Dict]]):

    flat_messages = flatten(messages)

    texts = [msg["text"] for msg in flat_messages]

    pixel_values_list = []
    num_patches_list = []

    for msg in flat_messages:
      pixel_tensor = load_image(msg["image_path"]).to(self.device).to(torch.float32)
      num_patches_list.append(get_num_patches(pixel_tensor))
      pixel_values_list.append(pixel_tensor)

    pixel_values = torch.cat(pixel_values_list, dim=0)

    responses = self.model.batch_chat(
      tokenizer=self.tokenizer,
      pixel_values=pixel_values,
      num_patches_list=num_patches_list,
      questions=texts,
      generation_config=self.generation_config
    )

    return responses
