import os
from argparse import ArgumentParser

from peft import PeftModel

from src.constants import model_dir, model_output_dir
from src.models.qwen_vl_inference import QwenVLInferenceEngine


parser = ArgumentParser()
parser.add_argument(
    "--adapter",
    help="Name of the directory the adapter is in.",
    required=True,
)
args = parser.parse_args()

engine = QwenVLInferenceEngine(use_4bit=True)
engine.load_model()

model = PeftModel.from_pretrained(
    model=engine.model,
    model_id=os.path.join(model_output_dir, args.adapter),
)
model = model.merge_and_unload()
model.save_pretrained(os.path.join(model_dir, args.adapter))
