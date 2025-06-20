from argparse import ArgumentParser

from src.eval.eval_models import evaluate_model
from src.models.qwen_vl_inference import QwenVLInferenceEngine
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
    parser.add_argument(
        "--approach",
        help="The name of the current approach (used for naming of the resulting files).",
        choices=["front_cam", "image_grid"],
        required=True,
    )
    parser.add_argument(
        "--test_set_size",
        help="Number of samples to test the current approach and pipeline with.",
        default=None,
    )
    args = parser.parse_args()

    # This should be continually added to so we can just pass this dict every time a new approach is added
    kwargs = {}
    if args.approach == "image_grid":
        kwargs["use_grid"] = True

    if args.train:
            train(
                args.approach,
                args.test_set_size,
                **kwargs,
            )
    elif args.eval:
        if is_cuda():
            engine = QwenVLInferenceEngine(use_4bit=True)
        else:
            engine = QwenVLInferenceEngine()

        evaluate_model(
            engine=engine,
            dataset_split="val",
            batch_size=30,
            test_set_size=args.test_set_size,
            **kwargs,
        )
    else:
        parser.print_help()
