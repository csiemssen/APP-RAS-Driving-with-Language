# Further adopted from https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-finetune/qwenvl/train/train_qwen.py

# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import transformers
import pandas as pd
from peft import LoraConfig
from transformers import Trainer
from qwen_vl_utils import process_vision_info

from src.constants import model_output_dir, model_log_dir
from src.data.basic_dataset import DriveLMImageDataset
from src.models.qwen_vl_inference import QwenVLInferenceEngine
from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None


def log_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model):
    """
    Set grad to True for LLM part of the model only.
    """
    for n, p in model.visual.named_parameters():
        p.requires_grad = False
    for n, p in model.visual.merger.named_parameters():
        p.requires_grad = False
    for n, p in model.model.named_parameters():
        p.requires_grad = True
    model.lm_head.requires_grad = True


# TODO: Look into the deepspeed config
# TODO: We will have to look into fixing the resolution -> This should likely allways be 1600 x 900
#       -> Seemingly important for fine tuning
# TODO: Test full pipeline with tiny DS
# TODO: Look through the warnings
# TODO: Check whether we can optimize further.
def train(approach_name: str):
    engine = QwenVLInferenceEngine(use_4bit=True, training=True)

    def collator(batch: Any):
        texts = [
            engine.processor.apply_chat_template(
                data["messages"], tokenize=False
            )
            for data in batch
        ]
        image_inputs, video_inputs = process_vision_info(
            [data["messages"][0] for data in batch]
        )

        batch = engine.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            padding_side="left",
        )

        labels = batch["input_ids"].clone()
        labels[labels == engine.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch


    dataset = [
        message for message, _, _, _, _ in
        DriveLMImageDataset(engine.training_message_formatter, split="train")
    ]

    engine.load_model()

    if hasattr(engine.model, "enable_input_require_grads"):
        engine.model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        engine.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    set_model(engine.model)

    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )
    engine.model.add_adapter(lora_config)
    engine.model.config.use_cache = False
    engine.model.gradient_checkpointing_enable()

    log_trainable_parameters(engine.model)

    trainer = Trainer(
        model=engine.model, 
        processing_class=engine.processor.tokenizer,
        args=TrainingArguments(
            remove_unused_columns=False,
            bf16=True,
            tf32=True,
            
        ),
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_state()

    name = approach_name + datetime.now()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=model_output_dir / name)
    pd.DataFrame(trainer.state.log_history).to_csv(
        model_log_dir / name + ".csv"
    )
