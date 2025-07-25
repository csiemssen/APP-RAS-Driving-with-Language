import os
from argparse import ArgumentParser

from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration

from src.constants import model_dir, model_output_dir
from src.models.qwen_vl_inference import QwenVLInferenceEngine


parser = ArgumentParser()
parser.add_argument(
    "--adapter",
    help="Name of the directory the adapter is in.",
    required=True,
)
args = parser.parse_args()

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct"
)

model = PeftModel.from_pretrained(
    model=model,
    model_id=os.path.join(model_output_dir, args.adapter),
)
test = model.merge_and_unload()
test.save_pretrained(os.path.join(model_dir, args.adapter))
