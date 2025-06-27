from argparse import ArgumentParser

from src.eval.eval_models import evaluate_model
from src.models.qwen_vl_inference import QwenVLInferenceEngine
from src.train.train_qwen import train
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
        choices=["front_cam", "image_grid", "descriptor_qas"],
        nargs="+",  # Allow multiple approaches to be specified
        required=True,
    )
    parser.add_argument(
        "--test_set_size",
        help="Number of samples to test the current approach and pipeline with.",
        default=None,
    )
    parser.add_argument(
        "--resize_factor",
        help="Resize factor to apply to the images. Original size is (1600 x 900). Currently only applied if using image_grid approach.",
        default="1.0",
    )
    args = parser.parse_args()

    approach_kwargs_map = {
        "image_grid": {"use_grid": True},
        "descriptor_qas": {"use_augmented": True},
        # Add more approaches here as needed
    }

    resize_factor = float(args.resize_factor)

    kwargs = {}
    for approach in args.approach:
        if approach in approach_kwargs_map:
            kwargs.update(approach_kwargs_map[approach])

    approach_name = "_".join(args.approach)

    logger.info(f"Running with approach: {approach_name}")

    if args.train:
        train(
            approach_name=approach_name,
            test_set_size=args.test_set_size,
            resize_factor=resize_factor,
            **kwargs,
        )
    elif args.eval:
        if is_cuda():
            engine = QwenVLInferenceEngine(use_4bit=True, resize_factor=resize_factor)
        else:
            engine = QwenVLInferenceEngine(resize_factor=resize_factor)

        evaluate_model(
            engine=engine,
            dataset_split="val",
            batch_size=30,
            test_set_size=args.test_set_size,
            resize_factor=resize_factor,
            **kwargs,
        )
    else:
        parser.print_help()
