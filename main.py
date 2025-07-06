import os
from argparse import ArgumentParser

from src.constants import model_dir
from src.eval.eval_models import evaluate_model
from src.models.qwen_vl_inference import QwenVLInferenceEngine
from src.train.train_qwen import train
from src.utils.approach import get_approach_kwargs, get_approach_name
from src.utils.logger import get_logger
from src.utils.utils import get_resize_image_size, is_cuda

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
        choices=[
            "front_cam",
            "image_grid",
            "descriptor_qas",
            "reasoning",
            "system_prompt",
        ],
        nargs="+",  # Allow multiple approaches to be specified
        required=True,
    )
    parser.add_argument(
        "--test_set_size",
        help="Number of samples to test the current approach and pipeline with.",
        default=None,
    )
    parser.add_argument(
        "--dataset_split",
        help="The dataset split to use for training / evaluation.",
        type=str,
        choices=["train", "val"],
        default="val",
    )
    parser.add_argument(
        "--resize_factor",
        help="Resize factor to apply to the images. Original size is (1600 x 900). Currently only applied if using image_grid approach.",
        default="0.5",
    )
    parser.add_argument(
        "--batch_size",
        help="The batch size to use for training / evaluation.",
        default="1",
    )
    parser.add_argument(
        "--model_path",
        help="The path of the finetuned model to use.",
        default=None,
    )
    args = parser.parse_args()

    model_path = None
    if args.model_path:
        model_path = os.path.join(model_dir, args.model_path)

    # add more approaches in get_approach_kwargs
    kwargs = get_approach_kwargs(args.approach)

    approach_name = get_approach_name(args.approach)

    resize_factor = float(args.resize_factor)

    logger.info(f"Running with approach: {approach_name}")

    if args.train:
        train(
            approach_name=approach_name,
            batch_size=args.batch_size,
            test_set_size=args.test_set_size,
            resize_factor=resize_factor,
            **kwargs,
        )
    elif args.eval:
        resize_image_size = get_resize_image_size(
            resize_factor=resize_factor, grid="image_grid" in args.approach
        )
        logger.debug(f"Using resize image size: {resize_image_size}")
        if is_cuda():
            engine = QwenVLInferenceEngine(
                model_path=model_path,
                use_4bit=True,
                resize_image_size=resize_image_size,
            )
        else:
            engine = QwenVLInferenceEngine(resize_image_size=resize_image_size)

        evaluate_model(
            engine=engine,
            dataset_split=args.dataset_split,
            batch_size=args.batch_size,
            test_set_size=args.test_set_size,
            approach_name=approach_name,
            **kwargs,
        )
    else:
        parser.print_help()
