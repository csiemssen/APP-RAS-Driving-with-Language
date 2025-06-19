from argparse import ArgumentParser

from src.eval.eval_models import evaluate_model
from src.models.intern_vl_inference import InternVLInferenceEngine
from src.utils.utils import is_cuda
from src.train.train_qwen import train

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--train", 
        help="Set to finetune the current model",
        action="store_true",
    )
    parser.add_argument(
        "--eval",
        help="Set to evaluate the current model",
        action="store_true"
    )
    args = parser.parse_args()

    if args.train:
        train()
    elif args.eval:
        if is_cuda():
            engine = InternVLInferenceEngine(use_4bit=True)
        else:
            engine = InternVLInferenceEngine()

        evaluate_model(
            engine=engine,
            dataset_split="val",
            batch_size=30,
        )
    else:
        parser.print_help()
