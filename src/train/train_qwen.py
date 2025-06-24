# Further adopted from https://github.com/QwenLM/Qwen2.5-VL

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
from transformers.trainer import ALL_LAYERNORM_LAYERS, get_parameter_names
from qwen_vl_utils import process_vision_info

from src.constants import model_output_dir, model_log_dir
from src.data.basic_dataset import DriveLMImageDataset
from src.models.qwen_vl_inference import QwenVLInferenceEngine
from src.utils.logger import get_logger
from src.utils.utils import create_subset_for_testing


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


def create_optimizer(self):

    opt_model = self.model

    if self.optimizer is None:
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
            projector_parameters = [
                name for name, _ in opt_model.named_parameters() if "merger" in name
            ]
            if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                vision_tower_parameters = [
                    name for name, _ in opt_model.named_parameters() if "visual" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer


# TODO: Look into the deepspeed config
def train(
        approach_name: str, 
        test_set_size: Optional[int | None]=None,
        use_grid: bool=False,
    ):
    name = approach_name + datetime.now().strftime("%H:%M:%S-%m-%d-%Y%")
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

    dataset = DriveLMImageDataset(
        engine.training_message_formatter, 
        split="train",
        use_grid=use_grid,
    )
    if test_set_size is not None:
        dataset = create_subset_for_testing(dataset, int(test_set_size))
    dataset = [
        message for message, _, _, _, _ in dataset
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
            output_dir=model_output_dir / name,
            num_train_epochs=1,
        ),
        train_dataset=dataset,
        data_collator=collator,
    )
    Trainer.create_optimizer = create_optimizer
    trainer.train()

    logger.info(f"Done training. Saving the model to: {model_output_dir / name}")

    trainer.save_state()

    pd.DataFrame(trainer.state.log_history).to_csv(
        model_log_dir / (name + ".csv")
    )
