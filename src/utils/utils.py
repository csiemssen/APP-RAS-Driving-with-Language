from typing import Any
import torch

def flatten(list_of_lists: list[list[Any]]) -> list[Any]:
    return [x for l in list_of_lists for x in l]

def remove_nones(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}

def sanitize_model_name(model_path: str) -> str:
    return model_path.replace("/", "_")

def get_device() -> str:
  if torch.cuda.is_available():
    return "cuda"
  elif torch.backends.mps.is_available():
    return "mps"
  else:
    return "cpu"

def is_mps() -> bool:
  return torch.backends.mps.is_available()

def is_cuda() -> bool:
  return torch.cuda.is_available()
