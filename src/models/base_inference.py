from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import torch
from src.utils.utils import get_device, is_cuda, is_mps

class BaseInferenceEngine(ABC):
    def __init__(self, model_path: str, revision: Optional[str] = None, device: Optional[str] = None):
      self.model_path = model_path
      self.device = device if device is not None else get_device()
      self.message_formatter = None
      self.quantization_config = None
      self.torch_dtype = torch.float32 if is_mps() else torch.float16
      self.revision = revision

      if is_cuda():
          torch.cuda.empty_cache()

    def configure_quantization(self, use_4bit: bool = False, torch_dtype: Optional[torch.dtype] = None):
        self.torch_dtype = torch_dtype # override init dtype if provided
        if is_cuda():
            from transformers import BitsAndBytesConfig
            if use_4bit:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=self.torch_dtype,
                )
            else:
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=self.torch_dtype,
                )
        return None

    # separating model loading for now, for testing purposes - may remove later
    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def predict_batch(self, messages: List[List[Dict]]) -> list[str]:
        pass
