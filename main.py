import torch

from src.eval.eval_models import evaluate_model
from src.models.intern_vl_inference import InternVLInferenceEngine
from src.utils.utils import is_cuda

if __name__ == "__main__":
    # Todo
    if is_cuda():
        engine = InternVLInferenceEngine(
            "OpenGVLab/InternVL3-2B",
            use_4bit=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        engine = InternVLInferenceEngine("OpenGVLab/InternVL3-2B")

    evaluate_model(
        engine=engine,
        dataset_split="val",
        batch_size=2,
    )
