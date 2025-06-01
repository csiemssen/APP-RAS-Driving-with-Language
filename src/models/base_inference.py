from abc import ABC, abstractmethod
from typing import List, Dict
import torch
from src.utils.utils import get_device

class BaseInferenceEngine(ABC):
    def __init__(self, model_path: str, device: torch.device = None):
      self.model_path = model_path
      self.device = device if device is not None else get_device()
      self.message_formatter = None

    # seperating model loading for now, for testing purposes - may remove later
    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def predict_batch(self, messages: List[List[Dict]]) -> list[str]:
        pass
