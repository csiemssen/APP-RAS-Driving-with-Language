import os
from argparse import ArgumentParser

from src.constants import model_dir
from src.eval.eval_models import evaluate_model
from src.models.anthropic_inference import AnthropicInferenceEngine
from src.models.gemini_inference import GeminiInferenceEngine
from src.models.gemma_inference import GemmaInferenceEngine
from src.models.intern_vl_inference import InternVLInferenceEngine
from src.models.openai_inference import OpenAIInferenceEngine
from src.models.qwen_vl_inference import QwenVLInferenceEngine
from src.train.train_qwen import train
from src.utils.approach import get_approach_kwargs, get_approach_name
from src.utils.logger import get_logger
from src.utils.utils import get_resize_image_size

logger = get_logger(__name__)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--provider",
        help="The inference provider/model to use.",
        choices=[
            "openai",
            "anthropic",
            "gemini",
            "gemma",
            "intern_vl",
            "qwen",
        ],
        default="qwen",
    )
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
            "add_kois",
            "add_bev",
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
        choices=["train", "val", "test"],
        default="val",
    )
    parser.add_argument(
        "--resize_factor",
        help="Resize factor to apply to the images. Original size is (1600 x 900). Currently only applied if using image_grid approach.",
        default="1",
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

    approach_name = "resize_factor=" + str(args.resize_factor)
    if len(args.approach) > 1:
        approach_name += "_" + get_approach_name(args.approach)

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
            resize_factor=resize_factor,
            grid="image_grid" in args.approach,
        )
        logger.debug(f"Using resize image size: {resize_image_size}")

        engine = None
        if args.provider == "openai":
            engine = OpenAIInferenceEngine(model=args.model_path)
        elif args.provider == "anthropic":
            engine = AnthropicInferenceEngine(model=args.model_path)
        elif args.provider == "gemini":
            engine = GeminiInferenceEngine(model=args.model_path)
        elif args.provider == "gemma":
            engine = GemmaInferenceEngine(model_path=model_path)
        elif args.provider == "intern_vl":
            engine = InternVLInferenceEngine(model_path=model_path)
        elif args.provider == "qwen":
            engine = QwenVLInferenceEngine(
                model_path=model_path, resize_image_size=resize_image_size
            )
        else:
            raise ValueError(f"Unknown provider: {args.provider}")

        evaluate_model(
            engine=engine,
            dataset_split=args.dataset_split,
            batch_size=args.batch_size,
            test_set_size=args.test_set_size,
            approach_name=approach_name,
            resize_factor=resize_factor,
            **kwargs,
        )
    else:
        parser.print_help()
