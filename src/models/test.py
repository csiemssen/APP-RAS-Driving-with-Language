import json
import os.path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

from src.constants import data_dir
from src.data.basic_dataset import DriveLMImageDataset, simple_dict_collate
from src.data.message_formats import QwenMessageFormat


def run_inference():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        quantization_config=nf4_config,
        device_map="cuda",
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # NOTE: This still does not produces the correct result
    #       We will have to figure out whether we can use batch processing here or not
    ds = DriveLMImageDataset(QwenMessageFormat())
    test_set = Subset(ds, np.arange(4))

    # TODO: Experiment with the batch size to fit as much as we can into memory!
    dl = DataLoader(test_set, batch_size=2, collate_fn=simple_dict_collate)

    results = []
    for messages, questions, labels, q_ids, qa_types in dl:
        # Preparation for inference
        texts = [
            processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            ) for message in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            padding_side="left",
        )
        inputs = inputs.to("cuda")

        # TODO: Build output.json as expected by the test server!

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print("> Messages:\n", messages)
        print("> Preds:\n", output_text)
        print("> Labels:\n", labels)
        print("> QIDs:\n", q_ids)

        results.extend([
            {
                "id": q_ids[i],
                "question": questions[i],
                "answer": output_text[i]
            } for i in range(len(output_text))
        ])

    # TODO: Input the proper info here once we have it
    json.dump(
        {
            "method": "test",
            "team": "test",
            "authors": ["test"],
            "email": "test",
            "institution": "test",
            "country": "test",
            "results": results,
        },
        open(os.path.join(data_dir, "output/submission.json"), "w"),
    )
