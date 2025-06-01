from src.eval.eval_models import evaluate_model
from src.models.intern_vl_inference import InternVLInferenceEngine

# Todo
evaluate_model(
    engine=InternVLInferenceEngine("OpenGVLab/InternVL3-2B"),
    dataset_split="val",
    batch_size=2,
)
