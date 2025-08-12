from abc import abstractmethod
from typing import Dict, List

from src.models.base_inference import BaseInferenceEngine


class RemoteInferenceEngine(BaseInferenceEngine):
    def load_model(self) -> None:
        pass

    @abstractmethod
    def predict_batch(self, messages: List[List[Dict]]) -> list[str]:
        pass
