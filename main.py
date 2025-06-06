from src.eval.eval_models import evaluate_model
from src.models.gemma_inference import GemmaInferenceEngine
from src.utils.utils import is_cuda

if __name__ == "__main__":
    if is_cuda():
        engine = GemmaInferenceEngine(use_4bit=True)
    else:
        engine = GemmaInferenceEngine()

    evaluate_model(
        engine=engine,
        dataset_split="val",
        batch_size=30,
    )
