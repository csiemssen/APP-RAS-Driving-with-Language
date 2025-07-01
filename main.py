from argparse import ArgumentParser

from src.eval.eval_models import evaluate_model
from src.models.qwen_vl_inference import QwenVLInferenceEngine
from src.train.train_qwen import train
from src.utils.approach import get_approach_kwargs, get_approach_name
from src.utils.logger import get_logger
from src.utils.utils import is_cuda

logger = get_logger(__name__)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--train",
        help="Set to finetune the current model",
        action="store_true",
    )
    parser.add_argument(
        "--eval", help="Set to evaluate the current model", action="store_true"
    )
    parser.add_argument(
        "--approach",
        help="The name of the current approach (used for naming of the resulting files).",
        choices=["front_cam", "image_grid", "descriptor_qas", "reasoning"],
        nargs="+",  # Allow multiple approaches to be specified
        required=True,
    )
    parser.add_argument(
        "--test_set_size",
        help="Number of samples to test the current approach and pipeline with.",
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        help="The batch size to use for training / evaluation.",
        default="1",
    )
    args = parser.parse_args()

    # add more approaches in get_approach_kwargs
    kwargs = get_approach_kwargs(args.approach)

    approach_name = get_approach_name(args.approach)

    logger.info(f"Running with approach: {approach_name}")

    if args.train:
        train(
            approach_name=approach_name,
            batch_size=args.batch_size,
            test_set_size=args.test_set_size,
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
            batch_size=args.batch_size,
            test_set_size=args.test_set_size,
            **kwargs,
        )
    else:
        parser.print_help()
