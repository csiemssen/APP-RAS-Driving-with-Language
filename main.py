from src.eval.eval_models import evaluate_model
from src.models.intern_vl_inference import InternVLInferenceEngine

if __name__ == '__main__':
  # todo
  evaluate_model(
      engine=InternVLInferenceEngine("OpenGVLab/InternVL3-2B"),
      dataset_split="val",
      batch_size=2,
  )
