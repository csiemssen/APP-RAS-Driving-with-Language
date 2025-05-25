import torch
from torch.utils.data import DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

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
    dl = DataLoader(ds, batch_size=3, collate_fn=simple_dict_collate)
    messages = next(iter(dl))
    #print(messages)

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

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("> Message: \n", messages)
    print("> Pred: \n", output_text)
    # print("Label:", a)
